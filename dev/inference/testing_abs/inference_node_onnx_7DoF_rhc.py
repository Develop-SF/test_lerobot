#!/usr/bin/env python3
"""
ROS2 Inference Node for LeRobot Deployment - Approach Plate (ONNX+TensorRT)
Absolute Action Mode

This node subscribes to sensor topics, runs inference using ONNX Runtime with TensorRT,
and publishes actions.

Includes image preprocessing matching the training pipeline:
- All cameras: Resize to 320×180 (no cropping or rotation for 7DoF model)
"""

import sys
import time
import threading
from threading import Lock, Thread
import argparse
import json
import datetime
import os
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
from control_msgs.msg import GripperCommand


class ONNXTensorRTInference:
    """ONNX Runtime with TensorRT backend inference for Approach Real Bing 20Hz model (Absolute)."""
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda", debug_dir: Optional[Path] = None,
                 chunk_size_threshold: float = 0.5, aggregate_fn_name: str = "weighted_average",
                 aggregate_weight_decay: float = 0.01, control_frequency: float = 20.0):
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
        self.debug_dir = debug_dir
        self.debug_step = 0
        
        # Async inference configuration (LeRobot-style)
        self.chunk_size_threshold = chunk_size_threshold
        self.aggregate_fn_name = aggregate_fn_name
        self.aggregate_weight_decay = aggregate_weight_decay
        self.control_frequency = control_frequency
        self.control_dt = 1.0 / control_frequency
        
        # Load ONNX configuration
        config_path = self.onnx_dir / "onnx_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
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
        
        # Setup ONNX Runtime sessions with TensorRT
        self._setup_onnx_sessions()
        
        # Setup noise scheduler (DDIM with 16 steps)
        self._setup_scheduler()
        
        # Setup normalization
        self._setup_normalization()
        
        # Image preprocessing parameters - 7DOF model uses 180x320 images without cropping
        self.crop_box = None  # No crop for 7DOF
        self.target_size = (320, 180)  # (width, height) - model expects 180x320
        
        # Setup center crop if needed (for ONNX encoder)
        if self.config.get('crop_shape'):
            self.center_crop = torchvision.transforms.CenterCrop(self.config['crop_shape'])
        else:
            self.center_crop = None

        # Right arm joint names (7DOF configuration) - MUST MATCH DATASET CONVERSION ORDER
        # Dataset order from rosbag_to_lerobot_rosbag2_7DoF.py: gripper at index 5, wrist_3 at index 6
        self.right_arm_joints = [
            'ra_shoulder_pan_joint',
            'ra_shoulder_lift_joint',
            'ra_elbow_joint',
            'ra_wrist_1_joint',
            'ra_wrist_2_joint',
            'ra_robotiq_85_left_knuckle_joint',  # gripper (index 5 in dataset)
            'ra_wrist_3_joint'                    # wrist_3 (index 6 in dataset)
        ]
        
        # Observation and action queues (matching DiffusionPolicy.reset())
        self._queues = {
            "action": deque(maxlen=self.config['n_action_steps']),
            "observation.state": deque(maxlen=self.config['n_obs_steps']),
            "observation.images": deque(maxlen=self.config['n_obs_steps']),
        }
        
        # Async inference state
        self.replan_trigger_size = int(self.config['n_action_steps'] * (1 - self.chunk_size_threshold))
        self.pending_chunk = None  # Stores result from background inference
        self.inference_lock = Lock()  # Thread-safe access to pending_chunk
        self.inference_thread = None  # Background inference thread
        self.inference_times = deque(maxlen=10)  # Rolling window of inference durations
        self.starvation_count = 0  # Track consecutive queue starvations
        
        print(f"\nImage preprocessing:")
        print(f"  - All cameras: Resize to {self.target_size}")
        print(f"\nAsync inference configuration:")
        print(f"  - Chunk size threshold: {self.chunk_size_threshold} ({self.replan_trigger_size}/{self.config['n_action_steps']} actions)")
        print(f"  - Aggregation function: {self.aggregate_fn_name}")
        print(f"  - Control frequency: {self.control_frequency} Hz (dt={self.control_dt:.4f}s)")
        
        # Metadata for ROS node
        self.input_mode = "vision_pos" # Assuming vision + position based on model structure
        self.output_mode = "pos_only" # 7DOF outputs position only (7 joints)
    
    def reset(self):
        """Clear observation and action queues."""
        self._queues["action"].clear()
        self._queues["observation.state"].clear()
        self._queues["observation.images"].clear()
        with self.inference_lock:
            self.pending_chunk = None
        self.starvation_count = 0

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
        # 7DOF model uses front and head cameras
        self.image_features = [
            'observation.images.sync_front_cam',
            'observation.images.sync_head_cam'
        ]
        print(f"  ✓ Loaded image features order (HARDCODED): {self.image_features}")
        
        # Map known camera roles to their config keys
        self.camera_key_map = {}
        for key in self.image_features:
            if 'head' in key:
                self.camera_key_map['head'] = key
            elif 'front' in key or 'left' in key:
                self.camera_key_map['front'] = key  # 'front' key for front camera
        
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
        policy = DiffusionPolicy.from_pretrained(str(model_path))
        
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
        
        # 7DOF model: simply resize to target size (no cropping or rotation)
        image_processed = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        
        return image_processed
    
    def extract_joint_positions_msg(self, joint_state_msg) -> np.ndarray:
        """Extract right arm joint positions (7 joints) from ROS JointState message."""
        joint_names = list(joint_state_msg.name)
        positions = np.array(joint_state_msg.position, dtype=np.float32)
        
        # Extract positions in correct 7DOF order matching dataset
        ordered_positions = []
        for joint_name in self.right_arm_joints:
            if joint_name in joint_names:
                idx = joint_names.index(joint_name)
                ordered_positions.append(positions[idx])
            else:
                # If a joint is missing, use 0 (shouldn't happen for valid data)
                print(f"Warning: Joint {joint_name} not found in joint_states")
                ordered_positions.append(0.0)
        
        if len(ordered_positions) != 7:
            raise ValueError(f"Expected 7 right arm joints (6 arm + gripper), found {len(ordered_positions)}")
        
        return np.array(ordered_positions, dtype=np.float32)
    
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
        return state
    
    def _normalize_image(self, image: np.ndarray, camera_key: str) -> np.ndarray:
        """
        Normalize image observation using camera-specific stats.
        
        Args:
            image: Image array (C, H, W)
            camera_key: Camera key from config (e.g., 'observation.images.sync_front_cam')
        
        Returns:
            Normalized image
        """
        # Try the exact key first
        if camera_key in self.norm_stats:
            stats = self.norm_stats[camera_key]
            if stats.get("mode") == "mean_std":
                mean = stats["mean"].reshape(3, 1, 1)
                std = stats["std"].reshape(3, 1, 1)
                image = (image - mean) / (std + 1e-8)
                return image
        
        # Try converting underscores to dots
        alt_key = camera_key.replace('_', '.')
        if alt_key in self.norm_stats:
            stats = self.norm_stats[alt_key]
            if stats.get("mode") == "mean_std":
                mean = stats["mean"].reshape(3, 1, 1)
                std = stats["std"].reshape(3, 1, 1)
                image = (image - mean) / (std + 1e-8)
                return image
        
        # Try converting dots to underscores
        alt_key = camera_key.replace('.', '_')
        if alt_key in self.norm_stats:
            stats = self.norm_stats[alt_key]
            if stats.get("mode") == "mean_std":
                mean = stats["mean"].reshape(3, 1, 1)
                std = stats["std"].reshape(3, 1, 1)
                image = (image - mean) / (std + 1e-8)
                return image
        
        print(f"Warning: No normalization stats found for {camera_key}")
        print(f"Available keys: {list(self.norm_stats.keys())}")
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
        return action
    
    def _aggregate_actions(self, old_actions: np.ndarray, new_actions: np.ndarray, method: str = "weighted_average") -> np.ndarray:
        """Aggregate overlapping action chunks using specified method.
        
        Args:
            old_actions: Remaining actions in queue (N, action_dim)
            new_actions: Newly predicted actions (M, action_dim) where M >= N
            method: Aggregation method ('weighted_average', 'replace', 'keep', 'mean')
            
        Returns:
            Aggregated actions for overlap region (N, action_dim)
        """
        overlap_len = len(old_actions)
        if overlap_len == 0:
            return np.array([], dtype=np.float32).reshape(0, new_actions.shape[1])
        
        # Extract overlapping portion from new actions
        new_overlap = new_actions[:overlap_len]
        
        if method == "weighted_average":
            # Exponential weighting: older predictions get less weight
            # Weight decay increases over time to favor newer predictions
            weights_old = np.exp(-self.aggregate_weight_decay * np.arange(overlap_len))[:, np.newaxis]
            weights_new = 1.0 - weights_old
            aggregated = (weights_old * old_actions + weights_new * new_overlap) / (weights_old + weights_new)
        elif method == "replace":
            aggregated = new_overlap
        elif method == "keep":
            aggregated = old_actions
        elif method == "mean":
            aggregated = (old_actions + new_overlap) / 2.0
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated
    
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
    
    def predict(self, front_image, head_image, joint_state) -> np.ndarray:
        """
        Run inference on preprocessed inputs.
        
        Args:
            front_image: Front camera image (H, W, 3) RGB - should be 320x180
            head_image: Head camera image (H, W, 3) RGB - should be 320x180
            joint_state: Joint positions (7,) including gripper
            
        Returns:
            Action array (7,) - 6 arm joints + gripper
        """
        # Preprocess images (HWC -> CHW, [0,1])
        front_processed = self.preprocess_image(front_image)
        head_processed = self.preprocess_image(head_image)
        
        # Map images to their config keys and normalize with camera-specific stats
        batch_images = {}
        if 'front' in self.camera_key_map:
            front_key = self.camera_key_map['front']
            front_normalized = self._normalize_image(front_processed, front_key)
            batch_images[front_key] = front_normalized
        if 'head' in self.camera_key_map:
            head_key = self.camera_key_map['head']
            head_normalized = self._normalize_image(head_processed, head_key)
            batch_images[head_key] = head_normalized
            
        # Stack in the exact order defined by the model config
        images_stacked = np.stack([batch_images[key] for key in self.image_features], axis=0)
        
        # Normalize joint state
        state_normalized = self._normalize_state(joint_state)
        
        # Populate queues
        self._queues["observation.state"].append(state_normalized)
        self._queues["observation.images"].append(images_stacked)
        
        # Check if we need to merge pending inference result
        with self.inference_lock:
            has_pending = self.pending_chunk is not None
        
        if has_pending:
            # Check if background thread completed
            if self.inference_thread is not None and not self.inference_thread.is_alive():
                self._merge_action_chunks()
        
        # Adaptive replanning: trigger inference when queue is low (LeRobot-style)
        queue_size = len(self._queues["action"])
        should_start_inference = (
            queue_size <= self.replan_trigger_size and 
            self.inference_thread is None  # No inference already running
        )
        
        if should_start_inference:
            # Prepare observation batches from queues
            state_list = list(self._queues["observation.state"])
            images_list = list(self._queues["observation.images"])
            
            # Pad with first observation if we don't have enough yet
            while len(state_list) < self.config['n_obs_steps']:
                state_list.insert(0, state_list[0] if state_list else state_normalized)
                images_list.insert(0, images_list[0] if images_list else images_stacked)
            
            # Only keep the last n_obs_steps
            state_list = state_list[-self.config['n_obs_steps']:]
            images_list = images_list[-self.config['n_obs_steps']:]
            
            # Create batch (1, n_obs_steps, ...)
            state_batch = np.stack(state_list, axis=0)[np.newaxis, ...]
            images_batch = np.stack(images_list, axis=0)[np.newaxis, ...]
            
            # Start background inference
            self._start_background_inference(images_batch, state_batch)
            
            if self.debug_dir:
                print(f"[Async] Triggered inference at queue_size={queue_size}/{self.config['n_action_steps']} (threshold={self.replan_trigger_size})")
        
        # Handle queue starvation
        if queue_size == 0:
            self.starvation_count += 1
            print(f"[WARNING] Queue starvation! Count: {self.starvation_count}")
            
            # If starving repeatedly, wait for inference to complete (blocking fallback)
            if self.starvation_count >= 3:
                print(f"[WARNING] Excessive starvation detected, switching to blocking mode temporarily")
                if self.inference_thread is not None:
                    self.inference_thread.join(timeout=2.0)  # Wait up to 2 seconds
                    if self.inference_thread.is_alive():
                        print(f"[ERROR] Inference thread timeout!")
                    else:
                        self._merge_action_chunks()
                
                # If still empty after waiting, we have a serious problem
                if len(self._queues["action"]) == 0:
                    raise RuntimeError("Action queue starvation: inference too slow for control frequency")
            else:
                # Repeat last action if available, otherwise block and wait
                if len(self._queues["observation.state"]) > 0:
                    # Return zero-velocity action (stay in place)
                    return np.zeros(self.config['action_dim'], dtype=np.float32)
                else:
                    raise RuntimeError("No actions available and no previous state to fallback")
        else:
            # Reset starvation counter on successful pop
            self.starvation_count = 0

        # Pop next action (Already Absolute & Unnormalized)
        action = self._queues["action"].popleft()
        
        return action

    def _run_inference(self, images_batch, state_batch):
        """
        Core inference logic: Preprocessing -> Encoding -> Denoising -> Postprocessing.
        Args:
            images_batch: (B, T, N, C, H, W)
            state_batch: (B, T, D)
        Returns:
            action_chunk_unnorm: (B, T_action, D_action)
        """
        inference_start = time.time()
        
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
        
        # Unnormalize the entire chunk immediately (Vectorized)
        # action_chunk is (1, n_action_steps, action_dim)
        action_chunk_unnorm = self._unnormalize_action(action_chunk)
        
        # Record inference time
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        avg_inference_time = np.mean(list(self.inference_times))
        print(f"  Inference time: {inference_time:.4f}s | Avg (last 10): {avg_inference_time:.4f}s")
        
        return action_chunk_unnorm
    
    def _start_background_inference(self, images_batch: np.ndarray, state_batch: np.ndarray):
        """Start inference in background thread.
        
        Args:
            images_batch: Batched images (B, T, N, C, H, W)
            state_batch: Batched states (B, T, D)
        """
        def _inference_worker():
            start_time = time.perf_counter()
            try:
                action_chunk = self._run_inference(images_batch, state_batch)
                inference_duration = time.perf_counter() - start_time
                
                # Store result and timing
                with self.inference_lock:
                    self.pending_chunk = {
                        'actions': action_chunk,
                        'duration': inference_duration,
                        'timestamp': time.time()
                    }
                    self.inference_times.append(inference_duration)
            except Exception as e:
                print(f"Background inference error: {e}")
                traceback.print_exc()
                with self.inference_lock:
                    self.pending_chunk = None
        
        self.inference_thread = Thread(target=_inference_worker, daemon=True)
        self.inference_thread.start()
    
    def _merge_action_chunks(self):
        """Merge pending chunk with current action queue using weighted average."""
        with self.inference_lock:
            if self.pending_chunk is None:
                return
            
            new_actions = self.pending_chunk['actions'][0]  # (n_action_steps, action_dim)
            inference_duration = self.pending_chunk['duration']
            
            # Calculate how many actions remain in queue
            old_actions = np.array(list(self._queues["action"]), dtype=np.float32)  # (remaining, action_dim)
            
            # Aggregate overlapping region
            if len(old_actions) > 0:
                aggregated_overlap = self._aggregate_actions(old_actions, new_actions, self.aggregate_fn_name)
                # Determine non-overlapping portion from new chunk
                non_overlap_start = len(old_actions)
                fresh_actions = new_actions[non_overlap_start:]
                # Combine: aggregated overlap + fresh actions
                merged_actions = np.vstack([aggregated_overlap, fresh_actions])
            else:
                # No overlap, use all new actions
                merged_actions = new_actions
            
            # Replace queue contents
            self._queues["action"].clear()
            for action in merged_actions:
                self._queues["action"].append(action)
            
            # Debug logging
            if self.debug_dir:
                overlap_len = len(old_actions)
                avg_inference_steps = inference_duration / self.control_dt
                debug_msg = (
                    f"[Async Merge] Queue: {len(old_actions)}→{len(merged_actions)} | "
                    f"Overlap: {overlap_len} | Inference: {inference_duration*1000:.1f}ms ({avg_inference_steps:.1f} steps) | "
                    f"Method: {self.aggregate_fn_name}"
                )
                print(debug_msg)
            
            # Clear pending chunk
            self.pending_chunk = None
            self.inference_thread = None

    def _warmup(self, n_steps: int = 5):
        """Run inference with dummy inputs to warm up the engine."""
        print(f"Running warmup for {n_steps} steps...")
        
        # Create dummy inputs matching the expected shapes
        # images_batch: (B, T, N, C, H, W)
        # state_batch: (B, T, D)
        
        batch_size = 1
        n_obs_steps = self.config['n_obs_steps']
        n_cameras = 2  # Front, Head
        C = 3
        # target_size is (320, 180) (width, height)
        # preprocess_image does: transpose(image_normalized, (2, 0, 1)) -> (C, H, W)
        # So H=180, W=320
        H, W = self.target_size[1], self.target_size[0]
        state_dim = self.config['state_dim']
        
        dummy_images = np.random.rand(batch_size, n_obs_steps, n_cameras, C, H, W).astype(np.float32)
        dummy_state = np.random.rand(batch_size, n_obs_steps, state_dim).astype(np.float32)
        
        for i in range(n_steps):
            _ = self._run_inference(dummy_images, dummy_state)
            print(f"  Warmup step {i+1}/{n_steps} complete")
        
        print("Warmup complete!")


    def predict_from_ros_messages(self, front_compressed_msg, head_compressed_msg, joint_state_msg) -> np.ndarray:
        """
        Run inference from ROS messages using ONNX+TensorRT.
        
        Returns:
            Action array (7,) for 7DOF position output (6 arm joints + gripper)
        """
        # Decode images
        front_image = self.decode_compressed_image_msg(front_compressed_msg, is_top_view=False)
        head_image = self.decode_compressed_image_msg(head_compressed_msg, is_top_view=True)
        
        # Extract joint state (7 joints including gripper)
        joint_state = self.extract_joint_positions_msg(joint_state_msg)
        
        return self.predict(front_image, head_image, joint_state)


class InferenceNode(Node):
    """ROS2 node for LeRobot inference deployment with image preprocessing (ONNX+TensorRT)."""
    
    def __init__(self, checkpoint_path: str, onnx_dir: str, device: str = "cuda", inference_frequency: float = 20.0, 
                 mode: str = "continuous", debug: bool = False, chunk_size_threshold: float = 0.5,
                 aggregate_fn_name: str = "weighted_average", aggregate_weight_decay: float = 0.01):
        """
        Initialize the ROS inference node.
        
        Args:
            checkpoint_path: Path to model checkpoint
            onnx_dir: Path to ONNX models directory
            device: Device for inference
            inference_frequency: Continuous inference frequency (Hz)
            mode: Inference mode - 'continuous' or 'triggered'
            chunk_size_threshold: Fraction of action queue empty before replanning (0.5 = 50%)
            aggregate_fn_name: Method for aggregating overlapping actions
            aggregate_weight_decay: Decay rate for weighted average aggregation
        """
        super().__init__('lerobot_inference_node_onnx')
        
        # Configuration
        self.mode = mode
        self.debug = debug
        self.inference_frequency = inference_frequency
        self.inference_interval = 1.0 / inference_frequency
        
        # Setup debug directory
        self.debug_dir = None
        self.timing_log_path = None
        if self.debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = Path(f"debug_inference_{timestamp}")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.timing_log_path = self.debug_dir / "timing_log.txt"
            with open(self.timing_log_path, "w") as f:
                f.write("timestamp,metric,value_ms\n")
            self.get_logger().info(f"Debug mode enabled. Saving data to {self.debug_dir}")

        # Initialize inference system with async parameters
        self.inference = ONNXTensorRTInference(
            checkpoint_path, onnx_dir, device, 
            debug_dir=self.debug_dir,
            chunk_size_threshold=chunk_size_threshold,
            aggregate_fn_name=aggregate_fn_name,
            aggregate_weight_decay=aggregate_weight_decay,
            control_frequency=inference_frequency
        )
        
        # Run warmup
        self.get_logger().info("Warming up inference engine...")
        self.inference._warmup()
        self.get_logger().info("Warmup complete.")
        
        # Gripper state tracking for hysteresis filter (prevents jitter)
        # 0.0 = Open, 0.8 = Closed
        self.current_gripper_state = 0.0  # Start in open state
        self.gripper_upper_threshold = 0.75  # Above this = command close
        self.gripper_lower_threshold = 0.25  # Below this = command open
        
        # Message storage (latest messages from each topic)
        self.latest_messages = {
            'front_image': None,
            'head_image': None,
            'joint_state': None
        }
        self.latest_message_times = {
            'front_image': 0.0,
            'head_image': 0.0,
            'joint_state': 0.0
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
        self.front_image_sub = self.create_subscription(
            CompressedImage,
            '/sync/emily01/front/color/image_raw/compressed',
            self.front_image_callback,
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
                '/sync/isaac_joint_states',
                self.joint_state_callback,
                sensor_qos
            )
            self.get_logger().info(f"Subscribed to joint states for input mode: {self.inference.input_mode}")
        else:
            self.get_logger().info("Running in vision_only mode - no joint state subscription")
        
        # Setup publisher - 7DOF uses right arm
        self.action_pub = self.create_publisher(
            JointTrajectory,
            '/ra_trajectory_controller/joint_trajectory',
            control_qos
        )
        
        # Setup gripper publisher
        self.gripper_pub = self.create_publisher(
            GripperCommand,
            '/right_gripper_cmd',
            control_qos
        )
        
        self.get_logger().info("LeRobot ONNX Inference Node ready (7DOF)")
        self.get_logger().info(f"Image preprocessing: No crop, simple resize to {self.inference.target_size}")
    
    def front_image_callback(self, msg: CompressedImage):
        """Handle front camera messages."""
        with self.message_lock:
            self.latest_messages['front_image'] = msg
            self.latest_message_times['front_image'] = time.time()
        
        # In triggered mode, trigger inference from front camera updates
        if self.mode == 'triggered':
            self.trigger_inference()
    
    def head_image_callback(self, msg: CompressedImage):
        """Handle head camera messages."""
        with self.message_lock:
            self.latest_messages['head_image'] = msg
            self.latest_message_times['head_image'] = time.time()
        # Don't trigger inference from head camera to avoid conflicts
    
    def joint_state_callback(self, msg: JointState):
        """Handle joint state messages."""
        with self.message_lock:
            self.latest_messages['joint_state'] = msg
            self.latest_message_times['joint_state'] = time.time()
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
            required_messages = ['front_image', 'head_image']
            if self.inference.input_mode != "vision_only":
                required_messages.append('joint_state')
            
            # Wait until we have all required messages
            if any(self.latest_messages[msg_type] is None for msg_type in required_messages):
                return
            
            # Copy messages for processing
            front_msg = self.latest_messages['front_image']
            head_msg = self.latest_messages['head_image']
            joint_msg = self.latest_messages['joint_state'] if self.inference.input_mode != "vision_only" else None
        
        # Run inference in separate thread to avoid blocking callbacks
        if not self.inference_running:
            threading.Thread(
                target=self.run_inference_threaded,
                args=(front_msg, head_msg, joint_msg),
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
                    required_messages = ['front_image', 'head_image']
                    if self.inference.input_mode != "vision_only":
                        required_messages.append('joint_state')
                    
                    # Check if all required messages are available
                    if all(self.latest_messages[msg_type] is not None for msg_type in required_messages):
                        # Copy messages for processing
                        front_msg = self.latest_messages['front_image']
                        head_msg = self.latest_messages['head_image'] 
                        joint_msg = self.latest_messages['joint_state'] if self.inference.input_mode != "vision_only" else None
                        
                        # Run inference
                        if self.debug:
                            # Calculate latency: now - min(message_times)
                            # Only consider times for required messages
                            msg_times = [self.latest_message_times['front_image'], self.latest_message_times['head_image']]
                            if self.inference.input_mode != "vision_only":
                                msg_times.append(self.latest_message_times['joint_state'])
                            
                            latency = time.time() - min(msg_times)
                            latency_ms = latency * 1000
                            self.get_logger().info(f"[DEBUG] Message Latency (Receive -> Inference Start): {latency_ms:.2f} ms")
                            
                            # Log to file
                            if self.timing_log_path:
                                try:
                                    with open(self.timing_log_path, "a") as f:
                                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                                        f.write(f"{timestamp},message_latency_ms,{latency_ms:.4f}\n")
                                except Exception as e:
                                    self.get_logger().error(f"Failed to write to timing log: {e}")

                        self.run_inference_synchronous(front_msg, head_msg, joint_msg)
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
    
    def run_inference_threaded(self, front_msg: CompressedImage, head_msg: CompressedImage, joint_msg: Optional[JointState]):
        """Run inference in a separate thread (for triggered mode)."""
        self.inference_running = True
        
        try:
            # Process ROS messages and run inference
            action = self.inference.predict_from_ros_messages(front_msg, head_msg, joint_msg)
            
            # Publish action as JointTrajectory
            self.publish_action(action)
            
        except Exception as e:
            self.get_logger().error(f"Triggered inference failed: {e}")
        
        finally:
            self.last_inference_time = time.time()
            self.inference_running = False
    
    def run_inference_synchronous(self, front_msg: CompressedImage, head_msg: CompressedImage, joint_msg: Optional[JointState]):
        """Run inference synchronously with the provided messages."""
        try:
            # Process ROS messages and run inference
            if self.debug:
                start_time = time.perf_counter()
                
            action = self.inference.predict_from_ros_messages(front_msg, head_msg, joint_msg)
            
            if self.debug:
                inference_time = time.perf_counter() - start_time
                inference_time_ms = inference_time * 1000
                self.get_logger().info(f"[DEBUG] Inference Time (Predict -> Publish): {inference_time_ms:.2f} ms")
                
                # Log to file
                if self.timing_log_path:
                    try:
                        with open(self.timing_log_path, "a") as f:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            f.write(f"{timestamp},inference_time_ms,{inference_time_ms:.4f}\n")
                    except Exception as e:
                        self.get_logger().error(f"Failed to write to timing log: {e}")
            
            # Publish action as JointTrajectory
            self.publish_action(action)
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
    
    def publish_action(self, action: np.ndarray):
        """Publish action as JointTrajectory message for 7DOF (6 arm + gripper)."""
        if action is None:
            return
        
        # Extract positions based on model output mode
        if self.inference.output_mode == "pos_only":
            # Position-only output (7 joints)
            if len(action) != 7:
                raise ValueError(f"Expected 7D action for 7DOF pos_only mode, got {len(action)}D")
            positions = action.astype(np.float64)
            self.get_logger().debug(f"Position-only action (7DOF): {positions}")
        elif self.inference.output_mode == "pos_vel":
            # Position + velocity output (14 total: 7 pos + 7 vel)
            if len(action) != 14:
                raise ValueError(f"Expected 14D action for 7DOF pos_vel mode, got {len(action)}D")
            positions = action[:7].astype(np.float64)
            velocities = action[7:].astype(np.float64)
            self.get_logger().debug(f"Position+velocity action (7DOF) - Pos: {positions}, Vel: {velocities}")
        else:
            raise ValueError(f"Unknown output mode: {self.inference.output_mode}")
        
        # Extract gripper value (index 5 in model output)
        # Model output order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, GRIPPER, wrist_3]
        raw_gripper_value = float(positions[5])
        
        # Apply hysteresis filter to prevent jitter (Schmitt trigger)
        # Gripper convention: 0.0 = Open, 0.8 = Closed
        # Only change state if model is confident (outside hysteresis band)
        if raw_gripper_value > self.gripper_upper_threshold:
            # Strong signal to close
            self.current_gripper_state = 0.8
        elif raw_gripper_value < self.gripper_lower_threshold:
            # Strong signal to open
            self.current_gripper_state = 0.0
        # else: maintain previous state (ignore noisy predictions in 0.25-0.75 range)
        
        gripper_value = self.current_gripper_state
        
        # Debug log if raw differs from filtered
        if abs(raw_gripper_value - gripper_value) > 0.1:
            self.get_logger().debug(
                f"Gripper hysteresis: raw={raw_gripper_value:.3f} → filtered={gripper_value:.3f} "
                f"(state={'CLOSED' if gripper_value > 0.5 else 'OPEN'})"
            )
        
        # Create and publish gripper command
        gripper_msg = GripperCommand()
        gripper_msg.position = gripper_value
        gripper_msg.max_effort = 100.0  # Default max effort
        self.gripper_pub.publish(gripper_msg)
        self.get_logger().debug(f"Published gripper command: {gripper_value:.4f}")
        
        # Prepare arm trajectory (6 joints, removing gripper at index 5)
        # Controller expects: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        arm_positions = np.concatenate([positions[:5], positions[6:7]]).tolist()
        
        trajectory_msg = JointTrajectory()
        trajectory_msg.header = Header()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = "base_link"
        
        # Set joint names for right arm (6 joints without gripper)
        trajectory_msg.joint_names = [
            'ra_shoulder_pan_joint',
            'ra_shoulder_lift_joint', 
            'ra_elbow_joint',
            'ra_wrist_1_joint',
            'ra_wrist_2_joint',
            'ra_wrist_3_joint'
        ]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = arm_positions
        # point.velocities = velocities  # Could add velocity support if needed
        point.accelerations = []  # Empty for position/velocity control
        point.effort = []  # Empty for position/velocity control
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds
        
        trajectory_msg.points = [point]
        self.action_pub.publish(trajectory_msg)


def main():
    """Main entry point for the ROS node."""
    
    parser = argparse.ArgumentParser(description='LeRobot ROS2 Inference Node - Approach Plate (ONNX+TensorRT)')
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
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode to log timing and save inference data'
    )
    parser.add_argument(
        '--chunk-size-threshold',
        type=float,
        default=0.5,
        help='Trigger replanning when queue is this fraction empty (0.5 = 50%% empty). LeRobot async parameter.'
    )
    parser.add_argument(
        '--aggregate-fn',
        type=str,
        default='weighted_average',
        choices=['weighted_average', 'replace', 'keep', 'mean'],
        help='Method for aggregating overlapping action chunks. LeRobot async parameter.'
    )
    parser.add_argument(
        '--aggregate-weight-decay',
        type=float,
        default=0.01,
        help='Exponential decay rate for weighted average (higher = favor newer predictions more)'
    )
    
    # Parse known args to allow ROS args
    parsed_args, unknown = parser.parse_known_args()
    
    # Initialize ROS
    rclpy.init(args=unknown)
    
    try:
        # Create and run the node with async inference parameters
        node = InferenceNode(
            checkpoint_path=parsed_args.checkpoint,
            onnx_dir=parsed_args.onnx_dir,
            device=parsed_args.device,
            inference_frequency=parsed_args.frequency,
            mode=parsed_args.mode,
            debug=parsed_args.debug,
            chunk_size_threshold=parsed_args.chunk_size_threshold,
            aggregate_fn_name=parsed_args.aggregate_fn,
            aggregate_weight_decay=parsed_args.aggregate_weight_decay
        )
        
        print("LeRobot ONNX Inference Node started (7DOF Pick & Place)")
        print("Subscribing to:")
        print("  - /sync/emily01/front/color/image_raw/compressed")
        print("  - /sync/emily01/head/color/image_raw/compressed")
        if node.inference.input_mode != "vision_only":
            print("  - /sync/joint_states")
        print("Publishing to:")
        print("  - /right_arm/joint_trajectory (6 arm joints)")
        print("  - /sns_right_gripper_cmd (gripper)")
        print(f"Inference mode: {parsed_args.mode}")
        print(f"Model modes - Input: {node.inference.input_mode}, Output: {node.inference.output_mode}")
        print(f"Image preprocessing:")
        print(f"  - Front camera: Resize to {node.inference.target_size}")
        print(f"  - Head camera: Resize to {node.inference.target_size}")
        
        if parsed_args.mode == 'continuous':
            print(f"Inference frequency: {parsed_args.frequency} Hz")
            # Start the continuous inference loop
            node.start_inference_loop()
        else:
            print("Inference triggered by front camera topic updates (max 30 Hz)")
        
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
