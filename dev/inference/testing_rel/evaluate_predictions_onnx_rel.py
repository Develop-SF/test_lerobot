#!/usr/bin/env python3
"""
Evaluate ONNX+TensorRT optimized model predictions against ground truth from rosbag data.
Supports Relative Action Representation for Approach Real Bing 20Hz.

This script uses ONNX Runtime with TensorRT backend for optimized inference,
comparing model predictions with actual joint trajectories from the rosbag.

Usage:
    python evaluate_predictions_onnx_rel.py \
        --checkpoint /path/to/checkpoint \
        --onnx-dir ./onnx_models \
        --rosbag /path/to/rosbag \
        --plot \
        --save-plot evaluation_results_onnx_rel.png
"""

# ROS2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available. Cannot evaluate with rosbag data.")


import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import deque
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# ONNX Runtime imports
import onnxruntime as ort
import torch
import torchvision
import cv2

# Diffusers scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class ONNXTensorRTInference:
    """
    ONNX Runtime with TensorRT backend inference for Approach Real Bing 20Hz model.
    
    This class handles:
    1. Loading ONNX models (RGB Encoder, UNet) with TensorRT optimization.
    2. Image preprocessing (Crop, Rotate, Resize, Normalize).
    3. State normalization and relative action handling.
    4. Denoising loop using DDIM scheduler.
    5. Action unnormalization and conversion from relative to absolute.
    """
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda"):
        """
        Initialize the ONNX+TensorRT inference system.
        
        Args:
            checkpoint_path: Path to original model checkpoint (used for loading normalization stats).
            onnx_dir: Path to directory containing ONNX models (unet.onnx, rgb_encoder.onnx) and config.
            device: Device for inference ("cuda" or "cpu").
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.onnx_dir = Path(onnx_dir)
        
        # Load ONNX configuration
        config_path = self.onnx_dir / "onnx_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load relative action settings
        self.use_relative_actions = self.config.get("use_relative_actions", False)
        self.arm_dim = self.config.get("arm_dim", 6)
        
        # Load image_features from the original model config to determine camera order
        self._load_image_features_config()
        
        print(f"Loaded ONNX config:")
        print(f"  Horizon: {self.config['horizon']}")
        print(f"  N obs steps: {self.config['n_obs_steps']}")
        print(f"  N action steps: {self.config['n_action_steps']}")
        print(f"  Action dim: {self.config['action_dim']}")
        print(f"  State dim: {self.config['state_dim']}")
        print(f"  Inference steps: {self.config['num_inference_steps']}")
        print(f"  Scheduler: {self.config['noise_scheduler_type']}")
        print(f"  Relative actions: {self.use_relative_actions}")
        if self.use_relative_actions:
            print(f"    Arm dim: {self.arm_dim}")
        
        # Setup ONNX Runtime sessions with TensorRT
        self._setup_onnx_sessions()
        
        # Setup noise scheduler (DDIM with 16 steps)
        self._setup_scheduler()
        
        # Setup normalization
        self._setup_normalization()
        
        # Image preprocessing parameters (from lerobot_inference.py)
        # Head camera (Top view): Crop (260, 135, 178, 224) then Rotate 90° Clockwise
        # Left Arm Camera: Resize to (224, 178)
        self.crop_box = (260, 135, 178, 224)  # (x, y, w, h) for top view
        self.target_size = (224, 178)  # (width, height) - final size for both cameras
        
        # Setup center crop if needed (for ONNX encoder)
        if self.config.get('crop_shape'):
            self.center_crop = torchvision.transforms.CenterCrop(self.config['crop_shape'])
        else:
            self.center_crop = None

        # Left arm joint names
        self.left_arm_joints = [
            'la_shoulder_pan_joint',
            'la_shoulder_lift_joint',
            'la_elbow_joint',
            'la_wrist_1_joint',
            'la_wrist_2_joint',
            'la_wrist_3_joint'
        ]
        
        # Observation and action queues (matching DiffusionPolicy.reset())
        self._queues = {
            "action": deque(maxlen=self.config['n_action_steps']),
            "observation.state": deque(maxlen=self.config['n_obs_steps']),
            "observation.images": deque(maxlen=self.config['n_obs_steps']),
        }
        
        print(f"\nImage preprocessing:")
        print(f"  - Head camera: Crop {self.crop_box} → Rotate 90° CW → {self.target_size}")
        print(f"  - Left arm camera: Resize to {self.target_size}")
    
    def reset(self):
        """Clear observation and action queues."""
        self._queues["action"].clear()
        self._queues["observation.state"].clear()
        self._queues["observation.images"].clear()

    def _setup_onnx_sessions(self):
        """Setup ONNX Runtime sessions with TensorRT provider."""
        print("\nInitializing ONNX Runtime sessions with TensorRT...")
        
        # Check available providers
        available_providers = ort.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # Configure providers
        providers = []
        if 'TensorrtExecutionProvider' in available_providers and self.device == "cuda":
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(self.onnx_dir / 'trt_engines'),
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
            print("  ✓ TensorRT enabled with FP16")
        elif 'CUDAExecutionProvider' in available_providers and self.device == "cuda":
            providers.append('CUDAExecutionProvider')
            print("  ℹ️  TensorRT not available, using CUDA")
        
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load RGB encoder
        encoder_path = self.onnx_dir / "rgb_encoder.onnx"
        self.rgb_encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_options,
            providers=providers
        )
        print(f"  ✓ RGB encoder loaded from {encoder_path}")
        print(f"    Using: {self.rgb_encoder_session.get_providers()[0]}")
        
        # Load UNet
        unet_path = self.onnx_dir / "unet.onnx"
        self.unet_session = ort.InferenceSession(
            str(unet_path),
            sess_options=sess_options,
            providers=providers
        )
        print(f"  ✓ UNet loaded from {unet_path}")
        print(f"    Using: {self.unet_session.get_providers()[0]}")
    
    def _setup_scheduler(self):
        """Setup DDIM scheduler with 16 steps."""
        scheduler_kwargs = {
            "num_train_timesteps": self.config['num_train_timesteps'],
            "beta_start": self.config['beta_start'],
            "beta_end": self.config['beta_end'],
            "beta_schedule": self.config['beta_schedule'],
            "clip_sample": self.config['clip_sample'],
            "clip_sample_range": self.config['clip_sample_range'],
            "prediction_type": self.config['prediction_type'],
        }
        
        self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
        print(f"  ✓ Scheduler: {self.config['noise_scheduler_type']} with {self.config['num_inference_steps']} steps")
    
    def _load_image_features_config(self):
        """Load the image_features list from the original model config."""
        # Resolve checkpoint path
        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "last"
        else:
            checkpoint_path = self.checkpoint_path
        
        model_path = checkpoint_path / "pretrained_model"
        config_path = model_path / "config.json"
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # HARDCODED ORDER TO MATCH PYTORCH VERSION
        self.image_features = [
            'observation.images.sync_left_arm_cam',
            'observation.images.sync_head_cam'
        ]
        print(f"  ✓ Loaded image features order (HARDCODED): {self.image_features}")
        
        # Map known camera roles to their config keys
        self.camera_key_map = {}
        for key in self.image_features:
            if 'head' in key:
                self.camera_key_map['head'] = key
            elif 'left' in key:
                self.camera_key_map['left'] = key
        
        print(f"  ✓ Camera key map: {self.camera_key_map}")
    
    def _setup_normalization(self):
        """Setup normalization from policy's normalize_inputs/normalize_targets."""
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
        # Resolve checkpoint path
        if not (self.checkpoint_path / "pretrained_model").exists():
            checkpoint_path = self.checkpoint_path / "last"
        else:
            checkpoint_path = self.checkpoint_path
        
        model_path = checkpoint_path / "pretrained_model"
        
        # Load the policy to extract normalization stats
        print(f"  Loading policy to extract normalization stats...")
        # IMPORTANT: Load with use_relative_actions=True if needed, to ensure correct stats loading
        policy = DiffusionPolicy.from_pretrained(
            str(model_path),
            use_relative_actions=self.use_relative_actions,
            arm_dim=self.arm_dim
        )
        
        self.norm_stats = {}
        self.unnorm_stats = {}
        
        # Extract input normalization from normalize_inputs
        if hasattr(policy, 'normalize_inputs') and policy.normalize_inputs is not None:
            # Iterate through all buffer parameters in normalize_inputs
            for buffer_name in dir(policy.normalize_inputs):
                if buffer_name.startswith('buffer_'):
                    # Extract the original name (e.g., 'buffer_observation_state' -> 'observation.state')
                    feature_name = buffer_name.replace('buffer_', '').replace('_', '.')
                    buffer = getattr(policy.normalize_inputs, buffer_name)
                    
                    # Check what normalization parameters exist
                    stats = {}
                    if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                        stats['mode'] = 'mean_std'
                        stats['mean'] = buffer['mean'].cpu().numpy()
                        stats['std'] = buffer['std'].cpu().numpy()
                        print(f"  ✓ Loaded normalization for {feature_name} (mode: mean_std)")
                    elif hasattr(buffer, 'min') and hasattr(buffer, 'max'):
                        stats['mode'] = 'min_max'
                        stats['min'] = buffer['min'].cpu().numpy()
                        stats['max'] = buffer['max'].cpu().numpy()
                        print(f"  ✓ Loaded normalization for {feature_name} (mode: min_max)")
                    
                    if stats:
                        self.norm_stats[feature_name] = stats
        
        # Extract output unnormalization from normalize_targets (used for unnormalization)
        if hasattr(policy, 'normalize_targets') and policy.normalize_targets is not None:
            # Iterate through all buffer parameters in normalize_targets
            for buffer_name in dir(policy.normalize_targets):
                if buffer_name.startswith('buffer_'):
                    # Extract the original name (e.g., 'buffer_action' -> 'action')
                    feature_name = buffer_name.replace('buffer_', '').replace('_', '.')
                    buffer = getattr(policy.normalize_targets, buffer_name)
                    
                    # Check what normalization parameters exist
                    stats = {}
                    if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                        stats['mode'] = 'mean_std'
                        stats['mean'] = buffer['mean'].cpu().numpy()
                        stats['std'] = buffer['std'].cpu().numpy()
                        print(f"   Loaded unnormalization for {feature_name} (mode: mean_std)")
                    elif hasattr(buffer, 'min') and hasattr(buffer, 'max'):
                        stats['mode'] = 'min_max'
                        stats['min'] = buffer['min'].cpu().numpy()
                        stats['max'] = buffer['max'].cpu().numpy()
                        print(f"   Loaded unnormalization for {feature_name} (mode: min_max)")
                    
                    if stats:
                        self.unnorm_stats[feature_name] = stats
        
        # Clean up policy to free memory
        del policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"  Normalization stats keys: {list(self.norm_stats.keys())}")
        print(f"  Unnormalization stats keys: {list(self.unnorm_stats.keys())}")
    
    def decode_compressed_image_msg(self, compressed_msg, is_top_view: bool = False) -> np.ndarray:
        """Decode ROS CompressedImage message to RGB array with preprocessing."""
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise ValueError("Failed to decode compressed image")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        if is_top_view:
            # Head camera: Crop then rotate
            x, y, w, h = self.crop_box
            image_cropped = image_rgb[y:y+h, x:x+w]
            image_processed = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Left arm camera: Resize
            image_processed = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        
        return image_processed
    
    def extract_joint_positions_msg(self, joint_state_msg) -> np.ndarray:
        """Extract left arm joint positions from ROS JointState message."""
        joint_names = list(joint_state_msg.name)
        positions = np.array(joint_state_msg.position, dtype=np.float32)
        
        left_arm_indices = []
        for joint_name in self.left_arm_joints:
            if joint_name in joint_names:
                left_arm_indices.append(joint_names.index(joint_name))
        
        if len(left_arm_indices) != 6:
            raise ValueError(f"Expected 6 left arm joints, found {len(left_arm_indices)}")
        
        return positions[left_arm_indices]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input (CHW format, normalized)."""
        # Normalize to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Convert HWC -> CHW
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        return image_chw
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state observation."""
        if "observation.state" in self.norm_stats:
            stats = self.norm_stats["observation.state"]
            if stats.get("mode") == "min_max":
                min_val = stats["min"]
                max_val = stats["max"]
                state = (state - min_val) / (max_val - min_val + 1e-8)
                state = state * 2 - 1  # Scale to [-1, 1]
            elif stats.get("mode") == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                state = (state - mean) / (std + 1e-8)
        return state
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image observation."""
        # Try different possible keys (with both underscores and periods)
        possible_keys = [
            "observation.image",
            "observation.images.sync_left_arm_cam",
            "observation.images.sync.left.arm.cam",
            "observation.images.sync_head_cam",
            "observation.images.sync.head.cam"
        ]
        
        for key in possible_keys:
            if key in self.norm_stats:
                stats = self.norm_stats[key]
                if stats.get("mode") == "mean_std":
                    mean = stats["mean"].reshape(3, 1, 1)
                    std = stats["std"].reshape(3, 1, 1)
                    image = (image - mean) / (std + 1e-8)
                break
        return image
    
    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize action."""
        if "action" in self.unnorm_stats:
            stats = self.unnorm_stats["action"]
            if stats.get("mode") == "min_max":
                # Action is in [-1, 1], scale back to original range
                action = (action + 1) / 2  # [-1, 1] -> [0, 1]
                min_val = stats["min"]
                max_val = stats["max"]
                action = action * (max_val - min_val) + min_val
            elif stats.get("mode") == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                action = action * (std + 1e-8) + mean
        return action
    
    def _encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images using ONNX RGB encoder.
        """
        # Convert to numpy if tensor
        if torch.is_tensor(images):
            images = images.cpu().numpy()

        # images shape: (batch, n_cameras, C, H, W)
        batch_size, n_cameras = images.shape[:2]
        
        # Flatten batch and camera dims
        # (B, N, C, H, W) -> (B*N, C, H, W)
        images_flat = images.reshape(-1, *images.shape[2:])

        # Step 1: Apply center crop using torchvision if specified
        if self.center_crop is not None:
            # Convert to tensor for cropping (supports batch)
            images_tensor = torch.from_numpy(images_flat)
            images_cropped = self.center_crop(images_tensor)
            images_flat = images_cropped.numpy()
        
        # Step 3: Run ONNX encoder (core neural network only)
        features_flat = self.rgb_encoder_session.run(
            ['features'],
            {'image': images_flat}
        )[0]
        # Output: (B*N, feature_dim)
        
        # Step 4: Reshape back to separate batch and cameras
        # (B*N, feature_dim) -> (B, N*feature_dim)
        features = features_flat.reshape(batch_size, -1)
        
        return features
    
    def predict(self, left_image, head_image, joint_state) -> np.ndarray:
        """
        Run inference on preprocessed inputs.
        
        This method:
        1. Preprocesses images (HWC -> CHW, Normalize).
        2. Updates observation queues.
        3. Prepares batches for inference.
        4. Handles relative state conversion (Absolute -> Relative).
        5. Runs the diffusion inference loop.
        6. Converts predicted relative actions back to absolute actions.
        
        Args:
            left_image: Left arm camera image (H, W, 3) RGB.
            head_image: Head camera image (H, W, 3) RGB.
            joint_state: Joint positions (6,).
            
        Returns:
            Action array (6,) representing the next target joint position.
        """
        # Preprocess images (HWC -> CHW, [0,1])
        left_processed = self.preprocess_image(left_image)
        head_processed = self.preprocess_image(head_image)
        
        # Normalize images
        left_normalized = self._normalize_image(left_processed)
        head_normalized = self._normalize_image(head_processed)
        
        # Map normalized images to their config keys
        batch_images = {}
        if 'left' in self.camera_key_map:
            batch_images[self.camera_key_map['left']] = left_normalized
        if 'head' in self.camera_key_map:
            batch_images[self.camera_key_map['head']] = head_normalized
            
        # Stack in the exact order defined by the model config
        images_stacked = np.stack([batch_images[key] for key in self.image_features], axis=0)
        
        # CRITICAL FIX: Keep state unnormalized for queue (to allow correct relative conversion)
        # This matches the fixed PyTorch pipeline
        state_stored = joint_state
        
        # Populate queues
        self._queues["observation.state"].append(state_stored)
        self._queues["observation.images"].append(images_stacked)
        
        # Generate actions if queue is empty
        if len(self._queues["action"]) == 0:
            # Prepare observation batches from queues
            state_list = list(self._queues["observation.state"])
            images_list = list(self._queues["observation.images"])
            
            # Pad with first observation if we don't have enough yet
            while len(state_list) < self.config['n_obs_steps']:
                state_list.insert(0, state_list[0] if state_list else state_stored)
                images_list.insert(0, images_list[0] if images_list else images_stacked)
            
            # Only keep the last n_obs_steps
            state_list = state_list[-self.config['n_obs_steps']:]
            images_list = images_list[-self.config['n_obs_steps']:]
            
            # Create batches: (1, n_obs_steps, ...)
            state_batch = np.stack(state_list, axis=0)[np.newaxis, ...]
            images_batch = np.stack(images_list, axis=0)[np.newaxis, ...]
            
            # Handle Relative Observations (PD2.2)
            if self.use_relative_actions:
                # 1. Convert unnormalized absolute state to unnormalized relative state
                # Reference is the current state (last in sequence)
                current_state = state_batch[:, -1:, :self.arm_dim]
                # Subtract current state from all steps
                state_batch[:, :, :self.arm_dim] = state_batch[:, :, :self.arm_dim] - current_state
                
                # 2. Normalize the relative state
                state_batch = self._normalize_state(state_batch)
            else:
                # Absolute mode: Just normalize the absolute state
                state_batch = self._normalize_state(state_batch)
            
            batch_size = 1
            n_obs_steps = state_batch.shape[1]
            
            # Encode images
            batch_size, n_obs_steps, n_cameras, C, H, W = images_batch.shape
            images_reshaped = images_batch.reshape(batch_size * n_obs_steps, n_cameras, C, H, W)
            features_reshaped = self._encode_images(images_reshaped)
            img_features = features_reshaped.reshape(batch_size, n_obs_steps, -1)
            
            # Prepare global conditioning
            global_cond_unflat = np.concatenate([state_batch, img_features], axis=2)
            global_cond = global_cond_unflat.reshape(batch_size, -1).astype(np.float32)
            
            # Initialize noise
            noise = np.random.randn(
                batch_size,
                self.config['horizon'],
                self.config['action_dim']
            ).astype(np.float32)
            
            # Denoising loop
            sample = noise.copy()
            
            # Set timesteps
            num_inference_steps = self.config.get('num_inference_steps', self.config['num_train_timesteps'])
            self.noise_scheduler.set_timesteps(num_inference_steps)
            
            for t in self.noise_scheduler.timesteps:
                timestep = np.array([t.item()], dtype=np.int64)
                timestep = np.repeat(timestep, batch_size)
                
                # Run ONNX UNet
                model_output = self.unet_session.run(
                    ['noise_pred'],
                    {
                        'sample': sample,
                        'timestep': timestep,
                        'global_cond': global_cond
                    }
                )[0]
                
                # Scheduler step
                sample = self.noise_scheduler.step(
                    torch.from_numpy(model_output),
                    t,
                    torch.from_numpy(sample)
                ).prev_sample.numpy()
            
            actions = sample
            
            # Extract action chunk
            start = self.config['n_obs_steps'] - 1
            end = start + self.config['n_action_steps']
            action_chunk = actions[:, start:end]
            
            # Process the entire chunk immediately (Unnormalize + Relative->Absolute)
            # action_chunk is (1, n_action_steps, action_dim)
            
            # 1. Unnormalize (Vectorized)
            # _unnormalize_action supports broadcasting (1, T, D) vs (D,)
            action_chunk_unnorm = self._unnormalize_action(action_chunk)
            
            # 2. Convert to Absolute (if relative)
            if self.use_relative_actions:
                # Use the state captured AT INFERENCE TIME (state_stored)
                # state_stored is (state_dim,)
                # Add it to the relative actions
                # Broadcasting: (1, T, arm_dim) + (arm_dim,)
                action_chunk_unnorm[:, :, :self.arm_dim] = action_chunk_unnorm[:, :, :self.arm_dim] + state_stored[:self.arm_dim]
            
            # Add to queue (now storing Absolute Unnormalized actions)
            # action_chunk_unnorm[0] is (n_action_steps, action_dim)
            # extend adds each row (action_dim,) to the deque
            self._queues["action"].extend(action_chunk_unnorm[0])
        
        # Pop next action (Already Absolute & Unnormalized)
        action = self._queues["action"].popleft()
        
        return action

    def predict_from_ros_messages(self, left_compressed_msg, head_compressed_msg, joint_state_msg) -> np.ndarray:
        """
        Run inference from ROS messages using ONNX+TensorRT.
        
        Returns:
            Action array (6,) for position output
        """
        # Decode images
        left_image = self.decode_compressed_image_msg(left_compressed_msg, is_top_view=False)
        head_image = self.decode_compressed_image_msg(head_compressed_msg, is_top_view=True)
        
        # Extract joint state
        joint_state = self.extract_joint_positions_msg(joint_state_msg)
        
        return self.predict(left_image, head_image, joint_state)


class PredictionEvaluator:
    """
    Evaluate model predictions against ground truth using ONNX+TensorRT.
    
    This class:
    1. Loads synchronized data from rosbags.
    2. Runs inference on the loaded data.
    3. Compares predicted actions with ground truth actions (from joint trajectory commands).
    4. Calculates and reports metrics (MAE, RMSE, Max Error).
    """
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda"):
        """
        Initialize evaluator with ONNX+TensorRT inference.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            onnx_dir: Path to ONNX models directory.
            device: Device for inference.
        """
        self.inference = ONNXTensorRTInference(checkpoint_path, onnx_dir, device)
        
        # Topic names
        self.topics = {
            'left_image': '/sync/emily01/left_arm/color/image_raw/compressed',
            'head_image': '/sync/emily01/head/color/image_raw/compressed',
            'joint_state': '/sync/joint_states',
            'action_command': '/sync/la_trajectory_controller/joint_trajectory'
        }
        
        # Left arm joint names
        self.left_arm_joints = self.inference.left_arm_joints
    
    def load_rosbag_data(self, rosbag_path: str, max_samples: int = 100) -> Dict[str, List]:
        """Load synchronized data from rosbag."""
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS2 not available")
        
        storage_id = 'mcap' if str(rosbag_path).endswith('.mcap') else 'sqlite3'
        storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        messages = {topic: [] for topic in self.topics.values()}
        
        while reader.has_next():
            min_topic_count = min(len(msgs) for msgs in messages.values())
            if min_topic_count >= max_samples:
                break
            
            topic, data, timestamp = reader.read_next()
            
            if topic in messages:
                if 'image_raw/compressed' in topic:
                    msg_type = get_message('sensor_msgs/msg/CompressedImage')
                elif 'joint_states' in topic:
                    msg_type = get_message('sensor_msgs/msg/JointState')
                elif 'joint_trajectory' in topic:
                    msg_type = get_message('trajectory_msgs/msg/JointTrajectory')
                else:
                    continue
                
                msg = deserialize_message(data, msg_type)
                messages[topic].append(msg)
        
        return messages
    
    def extract_ground_truth_positions(self, joint_msg) -> np.ndarray:
        """Extract left arm positions from joint state message."""
        return self.inference.extract_joint_positions_msg(joint_msg)
    
    def extract_ground_truth_from_command(self, command_msg) -> np.ndarray:
        """Extract command positions from JointTrajectory message."""
        if not hasattr(command_msg, 'points') or not command_msg.points:
            raise ValueError("JointTrajectory message has no trajectory points")
        
        first_point = command_msg.points[0]
        
        if not hasattr(first_point, 'positions') or not first_point.positions:
            raise ValueError("Trajectory point has no position commands")
            
        # Extract positions
        positions = np.array(first_point.positions, dtype=np.float32)
        
        return positions
    
    def evaluate(self, rosbag_path: str, num_samples: int = 50) -> Dict:
        """Evaluate model predictions against ground truth."""
        print(f"Loading data from: {rosbag_path}")
        
        messages = self.load_rosbag_data(rosbag_path, num_samples)
        
        left_count = len(messages[self.topics['left_image']])
        head_count = len(messages[self.topics['head_image']])
        joint_count = len(messages[self.topics['joint_state']])
        action_count = len(messages[self.topics['action_command']])
        
        print(f"Loaded messages - Left: {left_count}, Head: {head_count}, Joints: {joint_count}, Actions: {action_count}")
        
        min_count = min(left_count, head_count, joint_count, action_count)
        if min_count == 0:
            raise ValueError("No synchronized messages found")
        
        num_samples = min(num_samples, min_count)
        
        predictions = []
        ground_truth = []
        inference_times = []
        
        print(f"Evaluating {num_samples} samples with ONNX+TensorRT...")
        print(f"Using DDIM scheduler with {self.inference.config['num_inference_steps']} steps")
        
        for i in range(num_samples):
            try:
                left_msg = messages[self.topics['left_image']][i]
                head_msg = messages[self.topics['head_image']][i]
                joint_msg = messages[self.topics['joint_state']][i]
                action_msg = messages[self.topics['action_command']][i]
                
                # Time inference
                start_time = time.perf_counter()
                pred = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
                inference_time = time.perf_counter() - start_time
                inference_times.append(inference_time)
                
                # Extract ground truth from action command (not current state)
                gt_positions = self.extract_ground_truth_from_command(action_msg)
                
                predictions.append(pred)
                ground_truth.append(gt_positions)
                
                if (i + 1) % 10 == 0:
                    avg_time = np.mean(inference_times) * 1000
                    print(f"Processed {i + 1}/{num_samples} samples - Avg inference: {avg_time:.2f} ms")
                
            except Exception as e:
                print(f"Sample {i+1} failed: {e}")
                continue
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        results = self.calculate_metrics(predictions, ground_truth)
        results['inference_times'] = inference_times
        
        return results
    
    def calculate_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        mae = np.mean(np.abs(predictions - ground_truth), axis=0)
        mae_overall = np.mean(mae)
        
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=0))
        rmse_overall = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        
        max_error = np.max(np.abs(predictions - ground_truth), axis=0)
        max_error_overall = np.max(np.abs(predictions - ground_truth))
        
        errors = predictions - ground_truth
        std_error = np.std(errors, axis=0)
        std_error_overall = np.std(errors)
        
        return {
            'num_samples': len(predictions),
            'mae_per_joint': mae,
            'mae_overall': mae_overall,
            'rmse_per_joint': rmse,
            'rmse_overall': rmse_overall,
            'max_error_per_joint': max_error,
            'max_error_overall': max_error_overall,
            'std_error_per_joint': std_error,
            'std_error_overall': std_error_overall,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS - ONNX+TensorRT PIPELINE (DDIM 16 steps)")
        print("="*60)
        print(f"Number of samples: {results['num_samples']}")
        
        if 'inference_times' in results:
            inference_times = np.array(results['inference_times']) * 1000
            print(f"\nInference Performance:")
            print(f"  Mean: {np.mean(inference_times):.2f} ms")
            print(f"  Std:  {np.std(inference_times):.2f} ms")
            print(f"  Min:  {np.min(inference_times):.2f} ms")
            print(f"  Max:  {np.max(inference_times):.2f} ms")
        
        print(f"\nOverall Metrics:")
        print(f"  MAE:  {results['mae_overall']:.4f} rad ({np.rad2deg(results['mae_overall']):.2f}°)")
        print(f"  RMSE: {results['rmse_overall']:.4f} rad ({np.rad2deg(results['rmse_overall']):.2f}°)")
        print(f"  Max Error: {results['max_error_overall']:.4f} rad ({np.rad2deg(results['max_error_overall']):.2f}°)")
        print(f"  Std Error: {results['std_error_overall']:.4f} rad ({np.rad2deg(results['std_error_overall']):.2f}°)")
        
        print(f"\nPer-Joint Metrics:")
        for i, joint_name in enumerate(self.left_arm_joints):
            print(f"\n  {joint_name}:")
            print(f"    MAE:  {results['mae_per_joint'][i]:.4f} rad ({np.rad2deg(results['mae_per_joint'][i]):.2f}°)")
            print(f"    RMSE: {results['rmse_per_joint'][i]:.4f} rad ({np.rad2deg(results['rmse_per_joint'][i]):.2f}°)")
            print(f"    Max:  {results['max_error_per_joint'][i]:.4f} rad ({np.rad2deg(results['max_error_per_joint'][i]):.2f}°)")
            print(f"    Std:  {results['std_error_per_joint'][i]:.4f} rad ({np.rad2deg(results['std_error_per_joint'][i]):.2f}°)")
        
        print("\nOptimization Applied:")
        print(f"  - Backend: ONNX Runtime with TensorRT")
        print(f"  - Scheduler: DDIM with {self.inference.config['num_inference_steps']} inference steps")
        print(f"  - Precision: FP16 (TensorRT)")
        print("="*60)
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot evaluation results."""
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Prediction vs Ground Truth - ONNX+TensorRT Pipeline', fontsize=16)
        
        for i, (ax, joint_name) in enumerate(zip(axes.flat, self.left_arm_joints)):
            # Plot predictions vs ground truth
            ax.plot(ground_truth[:, i], label='Ground Truth', linewidth=2)
            ax.plot(predictions[:, i], label='Prediction', linewidth=2, alpha=0.7)
            ax.set_title(f'{joint_name}')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Position (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add error statistics to plot
            mae = results['mae_per_joint'][i]
            rmse = results['rmse_per_joint'][i]
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX+TensorRT predictions (Relative Action Support)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to original model checkpoint directory'
    )
    parser.add_argument(
        '--onnx-dir',
        type=str,
        default='onnx_models',
        help='Path to ONNX models directory'
    )
    parser.add_argument(
        '--rosbag',
        type=str,
        required=True,
        help='Path to rosbag directory for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots of results'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save plot (if --plot is enabled)'
    )
    
    args = parser.parse_args()
    
    if not ROS_AVAILABLE:
        print("Error: ROS2 not available. Cannot evaluate with rosbag data.")
        return
    
    try:
        print("="*60)
        print("APPROACH REAL BING 20HZ - ONNX+TENSORRT EVALUATION (RELATIVE)")
        print("="*60)
        
        evaluator = PredictionEvaluator(
            args.checkpoint,
            args.onnx_dir,
            args.device
        )
        
        results = evaluator.evaluate(args.rosbag, args.num_samples)
        evaluator.print_results(results)
        
        if args.plot:
            save_path = args.save_plot
            if not save_path:
                save_path = str(Path(args.onnx_dir) / 'evaluation_results_onnx_rel.png')
            evaluator.plot_results(results, save_path)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
