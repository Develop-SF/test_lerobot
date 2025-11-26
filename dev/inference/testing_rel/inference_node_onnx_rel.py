#!/usr/bin/env python3
"""
ROS2 Inference Node for LeRobot Deployment - Approach Plate (ONNX+TensorRT)
Relative Action Mode

This node subscribes to sensor topics, runs inference using ONNX Runtime with TensorRT,
and publishes actions.

Includes image preprocessing matching the training pipeline:
- Top view (head camera): Crop (260, 135, 178, 224) → Rotate 90° CW → 224×178
- Left arm camera: Resize to 224×178

Supports Relative Action Representation (PD2.1 + PD2.2)
"""

import sys
import time
import threading
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import deque

import numpy as np
import cv2
import torch
import torchvision
import onnxruntime as ort
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header


class ONNXTensorRTInference:
    """ONNX Runtime with TensorRT backend inference for Approach Real Bing 20Hz model (Relative)."""
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda"):
        """
        Initialize the ONNX+TensorRT inference system.
        
        Args:
            checkpoint_path: Path to original model checkpoint (for normalization stats)
            onnx_dir: Path to ONNX models directory
            device: Device for inference ("cuda" or "cpu")
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
        
        # Image preprocessing parameters (from training pipeline/lerobot_inference.py)
        self.crop_box = (260, 135, 178, 224)  # (x, y, w, h) for top view (head camera)
        self.target_size = (224, 178)  # (width, height) for both cameras
        
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
        
        # Metadata for ROS node
        self.input_mode = "vision_pos" # Assuming vision + position based on model structure
        self.output_mode = "pos_only" # Assuming position only output
    
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
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
        else:
            checkpoint_path = self.checkpoint_path
        
        model_path = checkpoint_path / "pretrained_model"
        config_path = model_path / "config.json"
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # HARDCODED ORDER TO MATCH PYTORCH VERSION
        # PyTorch uses insertion order of the dict, which happens to be [left, head] for this model
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
            checkpoint_path = self.checkpoint_path / "output" / "checkpoints" / "last"
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
                        print(f"  ✓ Loaded unnormalization for {feature_name} (mode: mean_std)")
                    elif hasattr(buffer, 'min') and hasattr(buffer, 'max'):
                        stats['mode'] = 'min_max'
                        stats['min'] = buffer['min'].cpu().numpy()
                        stats['max'] = buffer['max'].cpu().numpy()
                        print(f"  ✓ Loaded unnormalization for {feature_name} (mode: min_max)")
                    
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
            # Head camera (top view): Crop FIRST, then rotate
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
        
        Args:
            left_image: Left arm camera image (H, W, 3) RGB - should be 224x178
            head_image: Head camera image (H, W, 3) RGB - should be 224x178
            joint_state: Joint positions (6,)
            
        Returns:
            Action array (6,)
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
            
            # Set timesteps if not already set by external scheduler management
            # Use configured inference steps if available, otherwise train steps
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


class InferenceNode(Node):
    """ROS2 node for LeRobot inference deployment with image preprocessing (ONNX+TensorRT)."""
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda", inference_frequency: float = 20.0, mode: str = "continuous"):
        """
        Initialize the ROS inference node.
        
        Args:
            checkpoint_path: Path to model checkpoint
            onnx_dir: Path to ONNX models directory
            device: Device for inference
            inference_frequency: Continuous inference frequency (Hz)
            mode: Inference mode - 'continuous' or 'triggered'
        """
        super().__init__('lerobot_inference_node_onnx_rel')
        
        # Initialize inference system
        self.inference = ONNXTensorRTInference(checkpoint_path, onnx_dir, device)
        
        # Configuration
        self.mode = mode
        self.inference_frequency = inference_frequency
        self.inference_interval = 1.0 / inference_frequency
        
        # Message storage (latest messages from each topic)
        self.latest_messages = {
            'left_image': None,
            'head_image': None,
            'joint_state': None
        }
        self.message_lock = threading.Lock()
        
        # Inference control
        self.inference_running = False
        self.inference_thread = None
        
        # Triggered mode variables
        self.last_inference_time = 0.0
        self.min_inference_interval = 1.0 / 30.0  # Max 30 Hz for triggered mode
        
        # Setup QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Setup subscribers based on model input mode
        self.left_image_sub = self.create_subscription(
            CompressedImage,
            '/sync/emily01/left_arm/color/image_raw/compressed',
            self.left_image_callback,
            sensor_qos
        )
        
        self.head_image_sub = self.create_subscription(
            CompressedImage,
            '/sync/emily01/head/color/image_raw/compressed',
            self.head_image_callback,
            sensor_qos
        )
        
        # Only subscribe to joint states if needed by the model
        self.joint_state_sub = None
        if self.inference.input_mode != "vision_only":
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/sync/joint_states',
                self.joint_state_callback,
                sensor_qos
            )
            self.get_logger().info(f"Subscribed to joint states for input mode: {self.inference.input_mode}")
        else:
            self.get_logger().info("Running in vision_only mode - no joint state subscription")
        
        # Setup publisher
        self.action_pub = self.create_publisher(
            JointTrajectory,
            '/left_arm/joint_trajectory',
            control_qos
        )
        
        self.get_logger().info("LeRobot ONNX Inference Node ready (Relative)")
        self.get_logger().info(f"Image preprocessing: Top view crop {self.inference.crop_box} + rotate, Left arm resize to {self.inference.target_size}")
    
    def left_image_callback(self, msg: CompressedImage):
        """Handle left arm camera messages."""
        with self.message_lock:
            self.latest_messages['left_image'] = msg
        
        # In triggered mode, trigger inference from left camera updates
        if self.mode == 'triggered':
            self.trigger_inference()
    
    def head_image_callback(self, msg: CompressedImage):
        """Handle head camera messages."""
        with self.message_lock:
            self.latest_messages['head_image'] = msg
        # Don't trigger inference from head camera to avoid conflicts
    
    def joint_state_callback(self, msg: JointState):
        """Handle joint state messages."""
        with self.message_lock:
            self.latest_messages['joint_state'] = msg
        # Don't trigger inference from joint states to avoid conflicts
    
    def trigger_inference(self):
        """Trigger inference with rate limiting (for triggered mode)."""
        current_time = time.time()
        
        # Rate limiting to prevent excessive inference
        if current_time - self.last_inference_time < self.min_inference_interval:
            return
        
        # Get latest messages
        with self.message_lock:
            # Check for required messages based on model input mode
            required_messages = ['left_image', 'head_image']
            if self.inference.input_mode != "vision_only":
                required_messages.append('joint_state')
            
            # Wait until we have all required messages
            if any(self.latest_messages[msg_type] is None for msg_type in required_messages):
                return
            
            # Copy messages for processing
            left_msg = self.latest_messages['left_image']
            head_msg = self.latest_messages['head_image']
            joint_msg = self.latest_messages['joint_state'] if self.inference.input_mode != "vision_only" else None
        
        # Run inference in separate thread to avoid blocking callbacks
        if not self.inference_running:
            threading.Thread(
                target=self.run_inference_threaded,
                args=(left_msg, head_msg, joint_msg),
                daemon=True
            ).start()
    
    def start_inference_loop(self):
        """Start the continuous inference loop (for continuous mode only)."""
        if self.mode != 'continuous':
            self.get_logger().warn("Inference loop only available in continuous mode")
            return
            
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.get_logger().warn("Inference loop already running")
            return
        
        self.inference_running = True
        self.inference_thread = threading.Thread(target=self.continuous_inference_loop, daemon=True)
        self.inference_thread.start()
        self.get_logger().info(f"Started continuous inference loop at {self.inference_frequency} Hz")
    
    def stop_inference_loop(self):
        """Stop the continuous inference loop."""
        self.inference_running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2.0)
        self.get_logger().info("Stopped continuous inference loop")
    
    def continuous_inference_loop(self):
        """Continuous inference loop that runs at specified frequency."""
        # Wait 1 second before starting as requested
        self.get_logger().info("Waiting 1 second before starting inference loop...")
        time.sleep(1.0)
        
        self.get_logger().info(f"Starting continuous inference at {self.inference_frequency} Hz")
        
        while self.inference_running:
            loop_start_time = time.time()
            
            try:
                # Get latest messages
                with self.message_lock:
                    # Check for required messages based on model input mode
                    required_messages = ['left_image', 'head_image']
                    if self.inference.input_mode != "vision_only":
                        required_messages.append('joint_state')
                    
                    # Check if all required messages are available
                    if all(self.latest_messages[msg_type] is not None for msg_type in required_messages):
                        # Copy messages for processing
                        left_msg = self.latest_messages['left_image']
                        head_msg = self.latest_messages['head_image'] 
                        joint_msg = self.latest_messages['joint_state'] if self.inference.input_mode != "vision_only" else None
                        
                        # Run inference
                        self.run_inference_synchronous(left_msg, head_msg, joint_msg)
                    else:
                        missing_msgs = [msg_type for msg_type in required_messages if self.latest_messages[msg_type] is None]
                        self.get_logger().debug(f"Waiting for messages: {missing_msgs}")
                        
            except Exception as e:
                self.get_logger().error(f"Inference loop error: {e}")
            
            # Sleep to maintain frequency
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, self.inference_interval - elapsed_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.get_logger().warn(f"Inference loop running slower than {self.inference_frequency} Hz")
    
    def run_inference_threaded(self, left_msg: CompressedImage, head_msg: CompressedImage, joint_msg: Optional[JointState]):
        """Run inference in a separate thread (for triggered mode)."""
        self.inference_running = True
        
        try:
            # Process ROS messages and run inference
            action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
            
            # Publish action as JointTrajectory
            self.publish_action(action)
            
        except Exception as e:
            self.get_logger().error(f"Triggered inference failed: {e}")
        
        finally:
            self.last_inference_time = time.time()
            self.inference_running = False
    
    def run_inference_synchronous(self, left_msg: CompressedImage, head_msg: CompressedImage, joint_msg: Optional[JointState]):
        """Run inference synchronously with the provided messages."""
        try:
            # Process ROS messages and run inference
            action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
            
            # Publish action as JointTrajectory
            self.publish_action(action)
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
    
    def publish_action(self, action: np.ndarray):
        """Publish action as JointTrajectory message."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.header = Header()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = "base_link"
        
        # Set joint names for left arm
        trajectory_msg.joint_names = [
            'la_shoulder_pan_joint',
            'la_shoulder_lift_joint', 
            'la_elbow_joint',
            'la_wrist_1_joint',
            'la_wrist_2_joint',
            'la_wrist_3_joint'
        ]
        
        # Handle action based on model output mode
        if self.inference.output_mode == "pos_only":
            # Position-only output
            if len(action) != 6:
                raise ValueError(f"Expected 6D action for pos_only mode, got {len(action)}D")
            positions = action.astype(np.float64).tolist()
            velocities = [0.0] * 6  # Zero velocities for position-only control
            self.get_logger().debug(f"Position-only action: {positions}")
        elif self.inference.output_mode == "pos_vel":
            # Position + velocity output
            if len(action) != 12:
                raise ValueError(f"Expected 12D action for pos_vel mode, got {len(action)}D")
            positions = action[:6].astype(np.float64).tolist()
            velocities = action[6:].astype(np.float64).tolist()
            self.get_logger().debug(f"Position+velocity action - Pos: {positions}, Vel: {velocities}")
        else:
            raise ValueError(f"Unknown output mode: {self.inference.output_mode}")
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = velocities
        point.accelerations = []  # Empty for position/velocity control
        point.effort = []  # Empty for position/velocity control
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds
        
        trajectory_msg.points = [point]
        self.action_pub.publish(trajectory_msg)


def main():
    """Main entry point for the ROS node."""
    
    parser = argparse.ArgumentParser(description='LeRobot ROS2 Inference Node - Approach Plate (ONNX+TensorRT) - Relative')
    parser.add_argument(
        '--checkpoint', 
        type=str,
        required=True,
        help='Path to model checkpoint directory (for normalization stats)'
    )
    parser.add_argument(
        '--onnx-dir',
        type=str,
        required=True,
        help='Path to ONNX models directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--frequency',
        type=float,
        default=20.0,
        help='Continuous inference frequency (Hz) - used in continuous mode'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='continuous',
        choices=['continuous', 'triggered'],
        help='Inference mode: continuous (fixed frequency) or triggered (by topic updates)'
    )
    
    # Parse known args to allow ROS args
    parsed_args, unknown = parser.parse_known_args()
    
    # Initialize ROS
    rclpy.init(args=unknown)
    
    try:
        # Create and run the node
        node = InferenceNode(
            checkpoint_path=parsed_args.checkpoint,
            onnx_dir=parsed_args.onnx_dir,
            device=parsed_args.device,
            inference_frequency=parsed_args.frequency,
            mode=parsed_args.mode
        )
        
        print("LeRobot ONNX Inference Node started (Approach Plate - Relative)")
        print("Subscribing to:")
        print("  - /sync/emily01/left_arm/color/image_raw/compressed")
        print("  - /sync/emily01/head/color/image_raw/compressed")
        if node.inference.input_mode != "vision_only":
            print("  - /sync/joint_states")
        print("Publishing to:")
        print("  - /left_arm/joint_trajectory")
        print(f"Inference mode: {parsed_args.mode}")
        print(f"Model modes - Input: {node.inference.input_mode}, Output: {node.inference.output_mode}")
        print(f"Image preprocessing:")
        print(f"  - Top view: Crop {node.inference.crop_box} → Rotate 90° CW → {node.inference.target_size}")
        print(f"  - Left arm: Resize to {node.inference.target_size}")
        
        if parsed_args.mode == 'continuous':
            print(f"Inference frequency: {parsed_args.frequency} Hz")
            # Start the continuous inference loop
            node.start_inference_loop()
        else:
            print("Inference triggered by left camera topic updates (max 30 Hz)")
        
        print("Press Ctrl+C to stop")
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("Stopping node...")
    except Exception as e:
        print(f"Node error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'node' in locals():
            node.stop_inference_loop()
            node.destroy_node()
        
        rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
