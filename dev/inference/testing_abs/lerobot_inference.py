#!/usr/bin/env python3
"""
LeRobot Inference Module for Approach Plate Dataset

Provides inference for trained LeRobot models with automatic mode detection.
Includes image preprocessing matching the training pipeline:
- Top view (head camera): Crop (260, 135, 178, 224) → Rotate 90° CW → 224×178
- Left arm camera: Resize to 224×178
"""

import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch
from PIL import Image

# LeRobot imports
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


class LeRobotInference:
    """LeRobot inference with automatic input/output mode detection, image preprocessing, and relative action support."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        use_relative_actions: bool = False,
        arm_dim: int = 6,
    ):
        """
        Initialize the inference system.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            device: Device for inference ("cuda" or "cpu")
            use_relative_actions: If True, the model was trained with relative actions (PD2.1 + PD2.2)
            arm_dim: Number of arm joints (excluding gripper). Used for relative action conversion.
        """
        self.device = device
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.use_relative_actions = use_relative_actions
        self.arm_dim = arm_dim
        
        # Load model
        model_path = os.path.join(self.checkpoint_path, "pretrained_model")
        
        # Load policy with relative action support if needed
        if use_relative_actions:
            # Import relative action utilities
            from lerobot.common.utils.relative_actions import (
                convert_actions_to_absolute,
                convert_observation_state_to_relative,
                get_current_arm_state,
            )
            
            # Load policy with relative action support
            # The policy will automatically load identity normalization stats if they were saved during training
            self.policy = DiffusionPolicy.from_pretrained(
                model_path,
                use_relative_actions=use_relative_actions,
                arm_dim=arm_dim,
            )
            
            # Verify identity normalization is being used
            print("\nNormalization verification:")
            if hasattr(self.policy, 'normalize_inputs') and hasattr(self.policy.normalize_inputs, 'dataset_stats'):
                stats = self.policy.normalize_inputs.dataset_stats
                if 'action' in stats:
                    action_max = stats['action']['max']
                    action_min = stats['action']['min']
                    action_mean = stats['action']['mean']
                    # Check if using identity normalization (max=1, min=-1, mean=0)
                    if torch.allclose(action_max, torch.ones_like(action_max)) and \
                       torch.allclose(action_min, -torch.ones_like(action_min)) and \
                       torch.allclose(action_mean, torch.zeros_like(action_mean)):
                        print("  ✓ Identity normalization detected for actions")
                        print("    - Scale: 1.0, Offset: 0.0 (effectively no normalization)")
                    else:
                        print("  ⚠ Non-identity normalization detected!")
                        print(f"    - Action max: {action_max[:3]}")
                        print(f"    - Action min: {action_min[:3]}")
                        print(f"    - This may indicate training/inference mismatch")
        else:
            self.policy = DiffusionPolicy.from_pretrained(model_path)
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract model configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Get model parameters
        self.expected_image_size = tuple(self.config["input_features"]["observation.images.sync_left_arm_cam"]["shape"][1:])  # (193, 237)
        
        # Auto-detect input/output modes from config
        self.input_mode, self.state_dim = self._detect_input_mode()
        self.output_mode, self.action_dim = self._detect_output_mode()
        
        print(f"Auto-detected modes - Input: {self.input_mode}, Output: {self.output_mode}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Expected image size: {self.expected_image_size}")
        print(f"Relative actions: {'ENABLED' if use_relative_actions else 'DISABLED'}")
        if use_relative_actions:
            print(f"  - Arm dimension: {arm_dim}")
            print(f"  - PD2.1: Actions relative to current position")
            print(f"  - PD2.2: Observations relative to current position")
        
        # Image preprocessing parameters (matching training pipeline)
        self.crop_box = (260, 135, 178, 224)  # (x, y, w, h) for top view
        self.target_size = (224, 178)  # (width, height) - final size for both cameras
        
        print(f"Image preprocessing:")
        print(f"  - Top view: Crop {self.crop_box} → Rotate 90° CW → {self.target_size}")
        print(f"  - Left arm: Resize to {self.target_size}")
        
        # Left arm joint names for extraction
        self.left_arm_joints = [
            'la_shoulder_pan_joint',
            'la_shoulder_lift_joint', 
            'la_elbow_joint',
            'la_wrist_1_joint',
            'la_wrist_2_joint',
            'la_wrist_3_joint'
        ]
    
    def _detect_input_mode(self) -> tuple[str, int]:
        """
        Auto-detect input mode from model config.
        
        Returns:
            Tuple of (input_mode, state_dim)
        """
        input_features = self.config["input_features"]
        
        # Check if observation.state exists
        if "observation.state" not in input_features:
            return "vision_only", 0
        
        state_dim = input_features["observation.state"]["shape"][0]
        
        # Determine mode based on state dimension
        # 6 joints = position only, 12 joints = position + velocity
        if state_dim == 6:
            return "vision_pos", state_dim
        elif state_dim == 12:
            return "vision_pos_vel", state_dim
        else:
            raise ValueError(f"Unexpected state dimension: {state_dim}. Expected 0, 6, or 12.")
    
    def _detect_output_mode(self) -> tuple[str, int]:
        """
        Auto-detect output mode from model config.
        
        Returns:
            Tuple of (output_mode, action_dim)
        """
        output_features = self.config["output_features"]
        action_dim = output_features["action"]["shape"][0]
        
        # Determine mode based on action dimension
        # 6 = position only, 12 = position + velocity
        if action_dim == 6:
            return "pos_only", action_dim
        elif action_dim == 12:
            return "pos_vel", action_dim
        else:
            raise ValueError(f"Unexpected action dimension: {action_dim}. Expected 6 or 12.")
    
    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        """Resolve checkpoint path to the actual checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        # Check if it's already pointing to a checkpoint
        if (checkpoint_path / "pretrained_model").exists():
            return str(checkpoint_path)
        
        # Look for output/checkpoints structure
        checkpoints_dir = checkpoint_path / "output" / "checkpoints"
        if checkpoints_dir.exists():
            # Use 'last' checkpoint if available
            last_checkpoint = checkpoints_dir / "last"
            if last_checkpoint.exists():
                return str(last_checkpoint)
            
            # Otherwise find the latest checkpoint
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_dirs:
                latest = max(checkpoint_dirs, key=lambda x: int(x.name))
                return str(latest)
        
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_path}")
    
    def decode_compressed_image_msg(self, compressed_msg, is_top_view: bool = False) -> np.ndarray:
        """
        Decode ROS CompressedImage message to RGB array with preprocessing.
        
        Args:
            compressed_msg: ROS CompressedImage message
            is_top_view: If True, applies crop and rotation for top view (head camera)
                        If False, applies resize for left arm camera
            
        Returns:
            RGB image array (H, W, 3) - 193×237 for both cameras
        """
        # Decode compressed image data
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise ValueError("Failed to decode compressed image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply camera-specific preprocessing
        if is_top_view:
            # Top view (head camera): Crop FIRST, then rotate
            x, y, w, h = self.crop_box
            image_cropped = image_rgb[y:y+h, x:x+w]  # Crop to 193×237 (height=237, width=193)
            # Rotate 90° clockwise → 237×193 (height=193, width=237)
            image_processed = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Left arm camera: Resize to 237×193
            # cv2.resize takes (width, height)
            image_processed = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        
        return image_processed
    
    def extract_joint_positions_msg(self, joint_state_msg) -> np.ndarray:
        """
        Extract left arm joint state from ROS JointState message based on input mode.
        
        Args:
            joint_state_msg: ROS JointState message
            
        Returns:
            Joint state array based on input mode:
            - vision_only: raises error (should not be called)
            - vision_pos: positions only (6,)
            - vision_pos_vel: positions + velocities (12,)
        """
        if self.input_mode == "vision_only":
            raise ValueError("extract_joint_positions_msg should not be called for vision_only mode")
        
        # Extract all joint data
        joint_names = list(joint_state_msg.name)
        positions = np.array(joint_state_msg.position, dtype=np.float32)
        velocities = np.array(joint_state_msg.velocity, dtype=np.float32) if hasattr(joint_state_msg, 'velocity') and joint_state_msg.velocity else np.zeros_like(positions)
        
        # Find indices of left arm joints
        left_arm_indices = []
        for joint_name in self.left_arm_joints:
            if joint_name in joint_names:
                left_arm_indices.append(joint_names.index(joint_name))
        
        if len(left_arm_indices) != 6:
            raise ValueError(f"Expected 6 left arm joints, found {len(left_arm_indices)}")
        
        # Extract left arm data
        left_arm_positions = positions[left_arm_indices]
        left_arm_velocities = velocities[left_arm_indices]
        
        # Return based on input mode
        if self.input_mode == "vision_pos":
            return left_arm_positions.astype(np.float32)
        elif self.input_mode == "vision_pos_vel":
            # Combine position and velocity
            state = np.concatenate([left_arm_positions, left_arm_velocities])
            return state.astype(np.float32)
        else:
            raise ValueError(f"Unexpected input mode: {self.input_mode}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        Note: Image should already be at the correct size (193×237) from decode_compressed_image_msg.
        
        Args:
            image: RGB image array (H, W, 3) - should be 193×237
            
        Returns:
            Preprocessed image array (3, H, W)
        """
        # Verify image is at expected size
        height, width = self.expected_image_size
        if image.shape[:2] != (height, width):
            print(f"Warning: Image size {image.shape[:2]} doesn't match expected {(height, width)}. Resizing...")
            image = cv2.resize(image, (width, height))
        
        # Convert to PIL and back to ensure proper format
        pil_image = Image.fromarray(image)
        image_array = np.array(pil_image)
        
        # Normalize to [0, 1] and convert to CHW format
        image_normalized = image_array.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
        
        return image_chw
    
    def predict(self, left_image: np.ndarray, head_image: np.ndarray, joint_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run inference on preprocessed inputs.
        
        Args:
            left_image: Left arm camera image (H, W, 3) RGB - should be 193×237
            head_image: Head camera image (H, W, 3) RGB - should be 193×237
            joint_state: Optional joint state based on input mode:
                - vision_only: None or ignored
                - vision_pos: positions only (6,)
                - vision_pos_vel: positions + velocities (12,)
            
        Returns:
            Action array based on output mode:
            - pos_only: positions only (6,)
            - pos_vel: positions + velocities (12,)
        """
        # Preprocess images
        left_processed = self.preprocess_image(left_image)
        head_processed = self.preprocess_image(head_image)
        
        # Prepare model input
        with torch.no_grad():
            # Convert to tensors
            left_tensor = torch.from_numpy(left_processed).float().unsqueeze(0).to(self.device)  # (1, 3, H, W)
            head_tensor = torch.from_numpy(head_processed).float().unsqueeze(0).to(self.device)  # (1, 3, H, W)
            
            # Create observation dictionary
            observation = {
                "observation.images.sync_left_arm_cam": left_tensor,
                "observation.images.sync_head_cam": head_tensor
            }
            
            # Add state if required by input mode
            if self.input_mode != "vision_only":
                if joint_state is None:
                    raise ValueError(f"joint_state required for input mode {self.input_mode}")
                
                # Validate joint state dimension
                if joint_state.shape[0] != self.state_dim:
                    raise ValueError(f"Expected joint_state dimension {self.state_dim}, got {joint_state.shape[0]}")
                
                state_tensor = torch.from_numpy(joint_state).float().unsqueeze(0).to(self.device)
                observation["observation.state"] = state_tensor
            
            # Run inference
            action_tensor = self.policy.select_action(observation)
            action = action_tensor.cpu().numpy().flatten()
            
            return action
    
    def predict_from_ros_messages(self, left_compressed_msg, head_compressed_msg, joint_state_msg=None) -> np.ndarray:
        """
        Run inference directly from ROS messages.
        
        Args:
            left_compressed_msg: ROS CompressedImage from left arm camera
            head_compressed_msg: ROS CompressedImage from head camera (top view)
            joint_state_msg: Optional ROS JointState message (required for non-vision_only modes)
            
        Returns:
            Action array based on output mode:
            - pos_only: positions only (6,)
            - pos_vel: positions + velocities (12,)
        """
        # Decode ROS messages with appropriate preprocessing
        left_image = self.decode_compressed_image_msg(left_compressed_msg, is_top_view=False)
        head_image = self.decode_compressed_image_msg(head_compressed_msg, is_top_view=True)
        
        # Extract joint state if needed
        joint_state = None
        if self.input_mode != "vision_only":
            if joint_state_msg is None:
                raise ValueError(f"joint_state_msg required for input mode {self.input_mode}")
            joint_state = self.extract_joint_positions_msg(joint_state_msg)
        
        # Run inference
        return self.predict(left_image, head_image, joint_state)


if __name__ == "__main__":
    """Test the inference module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LeRobot inference module")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument(
        "--use_relative_actions",
        action="store_true",
        help="Enable relative action mode (must match training mode)"
    )
    parser.add_argument(
        "--arm_dim",
        type=int,
        default=6,
        help="Number of arm joints (excluding gripper). Default: 6"
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    print("Initializing inference system...")
    inference = LeRobotInference(
        args.checkpoint_path,
        args.device,
        use_relative_actions=args.use_relative_actions,
        arm_dim=args.arm_dim,
    )
    
    print("\nInference system ready!")
    print(f"Input mode: {inference.input_mode}")
    print(f"Output mode: {inference.output_mode}")
    print(f"Expected image size: {inference.expected_image_size}")
    print(f"Crop box (top view): {inference.crop_box}")
    print(f"Target size (both cameras): {inference.target_size}")
