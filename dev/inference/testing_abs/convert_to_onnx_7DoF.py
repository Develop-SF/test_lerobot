#!/usr/bin/env python3
"""
Convert Approach Real Bing 20Hz LeRobot Diffusion Policy to ONNX format.

This script exports the trained diffusion policy models (RGB encoder and UNet)
to ONNX format for optimized inference with ONNX Runtime and TensorRT.

Usage:
    python convert_to_onnx.py \
        --checkpoint /path/to/checkpoint \
        --output-dir ./onnx_models \
        --opset-version 17
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
from safetensors.torch import load_file

# Add lerobot package to path
# Assuming script is in: workspace/approach_real_bing_20hz_lerobot/testing_abs/convert_to_onnx.py
# And lerobot is in: workspace/lerobot/
script_dir = Path(__file__).parent
workspace_dir = script_dir.parent.parent
lerobot_path = workspace_dir / "lerobot"

if lerobot_path.exists():
    sys.path.insert(0, str(workspace_dir))
    print(f"Added to Python path: {workspace_dir}")
else:
    print(f"Warning: lerobot package not found at {lerobot_path}")
    print("Attempting to import from installed package...")

# Import from older lerobot version
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


def get_model_device(model):
    """Get device of model from its parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


class RGBEncoderCoreONNX(torch.nn.Module):
    """
    ONNX-compatible RGB encoder wrapper.
    
    This wrapper exports only the core neural network components:
    - Backbone (ResNet)
    - Spatial Softmax
    
    Image preprocessing (center crop) is handled in Python before ONNX inference.
    Input shape: (batch * n_cameras, C, H, W) - already flattened and cropped
    Output shape: (batch * n_cameras, feature_dim)
    """
    
    def __init__(self, rgb_encoder):
        super().__init__()
        self.backbone = rgb_encoder.backbone
        self.pool = rgb_encoder.pool
        self.out = rgb_encoder.out
        self.relu = rgb_encoder.relu
    
    def forward(self, x):
        # Skip cropping - assume input is already properly sized
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x



def export_rgb_encoder(policy, output_dir: Path, opset_version: int = 17):
    """
    Export RGB encoder to ONNX.
    
    This function:
    1. Locates the RGB encoder within the policy structure.
    2. Wraps it in `RGBEncoderCoreONNX` to export only the core network (ResNet + Spatial Softmax).
    3. Handles the fact that image cropping and reshaping are done in Python preprocessing.
    4. Exports the model to `rgb_encoder.onnx`.
    
    Args:
        policy: The loaded LeRobot policy.
        output_dir: Directory to save the ONNX model.
        opset_version: ONNX opset version (default: 17).
        
    Returns:
        Path to the exported ONNX file.
    """
    print("\n" + "="*80)
    print("EXPORTING RGB ENCODER")
    print("="*80)
    
    # In older lerobot, RGB encoder might be in different location
    # Try different possible locations
    if hasattr(policy, 'rgb_encoder'):
        rgb_encoder = policy.rgb_encoder
    elif hasattr(policy, 'vision_encoder'):
        rgb_encoder = policy.vision_encoder
    elif hasattr(policy, 'diffusion') and hasattr(policy.diffusion, 'rgb_encoder'):
        rgb_encoder = policy.diffusion.rgb_encoder
    elif hasattr(policy, 'diffusion') and hasattr(policy.diffusion, 'vision_encoder'):
        rgb_encoder = policy.diffusion.vision_encoder
    else:
        # List available attributes to help debug
        print("Available policy attributes:")
        for attr in dir(policy):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        if hasattr(policy, 'diffusion'):
            print("Available diffusion attributes:")
            for attr in dir(policy.diffusion):
                if not attr.startswith('_'):
                    print(f"  - {attr}")
        raise AttributeError("Could not find RGB encoder in policy. Please check model structure.")
    
    # Wrap encoder to export only core components (no crop, no reshape)
    # Preprocessing (crop) and reshaping will be done in Python
    print(f"\nCreating ONNX wrapper (core components only)")
    print(f"  - Crop will be handled in Python preprocessing")
    print(f"  - Reshape will be handled in Python code")
    rgb_encoder_core = RGBEncoderCoreONNX(rgb_encoder)
    rgb_encoder_core.eval()
    
    # Get configuration (convert to dict if it's a config object)
    config = policy.config
    if hasattr(config, '__dict__'):
        # Convert config object to dictionary
        config_dict = vars(config)
    else:
        config_dict = config
    
    # Determine number of cameras from config
    n_cameras = 0
    for key in config_dict.get("input_features", {}):
        if "images" in key:
            n_cameras += 1
    
    print(f"Number of cameras detected: {n_cameras}")
    
    # Create dummy input for RGB encoder
    # Shape: (batch=1, n_cameras, C, H, W)
    # After cropping: (batch, n_cameras, C, crop_h, crop_w)
    crop_shape = config_dict.get("crop_shape", None)
    if crop_shape:
        crop_h, crop_w = crop_shape
    else:
        # Use original image shape
        input_features = config_dict.get("input_features", {})
        # Try to find a shape in input features
        img_shape = None
        for key, val in input_features.items():
            if "images" in key:
                # Handle both dict and object-based configs
                if isinstance(val, dict) and "shape" in val:
                    img_shape = val["shape"]
                elif hasattr(val, "shape"):
                    img_shape = val.shape
                break
        
        if img_shape:
            # shape is usually [C, H, W]
            crop_h, crop_w = img_shape[1], img_shape[2]
        else:
             # Fallback to hardcoded values if not found (matching lerobot_inference.py target size)
             # Note: lerobot_inference.py uses (224, 178) which is (W, H)
             # PyTorch usually expects (H, W)
             crop_h, crop_w = 178, 224 

    # Get device from model parameters
    device = get_model_device(rgb_encoder)
    
    # ONNX model expects flattened input: (batch * n_cameras, C, H, W)
    # The reshaping and cropping is done in Python preprocessing
    dummy_image_flat = torch.randn(1 * n_cameras, 3, crop_h, crop_w, device=device)
    
    print(f"\nDummy input for ONNX export:")
    print(f"  Shape: {dummy_image_flat.shape} (already flattened)")
    print(f"  Crop dimensions: {crop_h} x {crop_w}")
    print(f"  Note: Python preprocessing will:")
    print(f"    1. Apply center crop to each camera")
    print(f"    2. Flatten (B, N, C, H, W) -> (B*N, C, H, W)")
    print(f"    3. Pass to ONNX encoder")
    print(f"    4. Reshape output (B*N, features) -> (B, N*features)")
    
    # Export to ONNX
    output_path = output_dir / "rgb_encoder.onnx"
    
    torch.onnx.export(
        rgb_encoder_core,
        dummy_image_flat,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch_x_cameras'},  # Batch and cameras are flattened
            'features': {0: 'batch_x_cameras'}
        }
    )
    
    print(f"✓ RGB encoder exported to {output_path}")
    return str(output_path)


def export_unet(policy, output_dir: Path, opset_version: int = 17):
    """
    Export UNet denoising model to ONNX.
    
    This function:
    1. Locates the UNet within the policy structure.
    2. Determines input/output dimensions from the policy config.
    3. Creates dummy inputs (sample, timestep, global_cond) matching the model signature.
    4. Exports the model to `unet.onnx`.
    
    Args:
        policy: The loaded LeRobot policy.
        output_dir: Directory to save the ONNX model.
        opset_version: ONNX opset version (default: 17).
        
    Returns:
        Path to the exported ONNX file.
    """
    print("\n" + "="*80)
    print("EXPORTING UNET")
    print("="*80)
    
    # Access UNet from diffusion module
    if not hasattr(policy, 'diffusion'):
        raise AttributeError("Policy does not have 'diffusion' attribute")
    
    if not hasattr(policy.diffusion, 'unet'):
        print("Available diffusion attributes:")
        for attr in dir(policy.diffusion):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        raise AttributeError("Diffusion module does not have 'unet' attribute")
    
    unet = policy.diffusion.unet
    unet.eval()
    
    # Get configuration (convert to dict if it's a config object)
    config = policy.config
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    horizon = getattr(config, "horizon", 16)
    
    # Access output features - handle both dict and object
    if hasattr(config, "output_features"):
        output_features = config.output_features
        action_dim = 6  # Default value

        # Try dictionary access first (handles dotted keys better)
        if isinstance(output_features, dict):
            # Check for 'action' key
            action_config = output_features.get("action", {})
            if isinstance(action_config, dict):
                action_dim = action_config.get("shape", [6])[0]
            elif hasattr(action_config, "shape") and action_config.shape is not None:
                # Handle PolicyFeature objects with shape attribute
                action_dim = action_config.shape[0]
        # Then try object access
        elif hasattr(output_features, "action"):
            action_feature = output_features.action
            if hasattr(action_feature, "shape") and action_feature.shape is not None:
                action_dim = action_feature.shape[0]
    else:
        action_dim = 6  # default
    
    # Calculate global conditioning dimension
    # State features
    if hasattr(config, "input_features"):
        input_features = config.input_features
        
        # Determine n_cameras and state_dim
        # Handle both dict and object-based configs gracefully
        keys = []
        if isinstance(input_features, dict):
            keys = list(input_features.keys())
        elif hasattr(input_features, "__dict__"):
            keys = list(vars(input_features).keys())
        
        n_cameras = sum(1 for key in keys if "images" in key)
        
        # Try different possible paths for state dimension
        state_dim = 6 # Default fallback
        
        # 1. Try dotted key (common in dict-based configs)
        if isinstance(input_features, dict) and "observation.state" in input_features:
            state_feature = input_features["observation.state"]
            if isinstance(state_feature, dict):
                state_dim = state_feature.get("shape", [6])[0]
            elif hasattr(state_feature, "shape"):
                state_dim = state_feature.shape[0]
        # 2. Try nested object access
        elif hasattr(input_features, "observation") and hasattr(input_features.observation, "state"):
            state_dim = input_features.observation.state.shape[0]
        # 3. Try dotted attribute access
        elif "observation.state" in keys:
            state_feature = getattr(input_features, "observation.state")
            state_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else 6
    else:
        n_cameras = 2  # default
        state_dim = 6  # default
    
    n_obs_steps = getattr(config, "n_obs_steps", 2)
    
    # Get RGB encoder output dimension
    # For ResNet18 with spatial softmax: feature_dim = spatial_softmax_num_keypoints * 2 * n_cameras
    spatial_softmax_keypoints = getattr(config, "spatial_softmax_num_keypoints", 32)
    img_feature_dim_per_camera = spatial_softmax_keypoints * 2
    total_img_feature_dim = img_feature_dim_per_camera * n_cameras
    
    # Global conditioning: (state_dim * n_obs_steps) + (img_feature_dim * n_obs_steps)
    global_cond_dim = (state_dim * n_obs_steps) + (total_img_feature_dim * n_obs_steps)
    
    print(f"Configuration:")
    print(f"  Horizon: {horizon}")
    print(f"  Action dim: {action_dim}")
    print(f"  State dim: {state_dim}")
    print(f"  N obs steps: {n_obs_steps}")
    print(f"  N cameras: {n_cameras}")
    print(f"  Image feature dim per camera: {img_feature_dim_per_camera}")
    print(f"  Total image feature dim: {total_img_feature_dim}")
    print(f"  Global conditioning dim: {global_cond_dim}")
    
    # Create dummy inputs
    batch_size = 1
    device = get_model_device(unet)
    dummy_sample = torch.randn(batch_size, horizon, action_dim, device=device)
    dummy_timestep = torch.randint(0, 100, (batch_size,), device=device, dtype=torch.long)
    dummy_global_cond = torch.randn(batch_size, global_cond_dim, device=device)
    
    print(f"\nDummy input shapes:")
    print(f"  Sample: {dummy_sample.shape}")
    print(f"  Timestep: {dummy_timestep.shape}")
    print(f"  Global cond: {dummy_global_cond.shape}")
    
    # Export to ONNX
    output_path = output_dir / "unet.onnx"
    
    torch.onnx.export(
        unet,
        (dummy_sample, dummy_timestep, dummy_global_cond),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['sample', 'timestep', 'global_cond'],
        output_names=['noise_pred'],
        dynamic_axes={
            'sample': {0: 'batch'},
            'timestep': {0: 'batch'},
            'global_cond': {0: 'batch'},
            'noise_pred': {0: 'batch'}
        }
    )
    
    print(f"✓ UNet exported to {output_path}")
    return str(output_path)


def save_config(policy, output_dir: Path):
    """
    Save ONNX inference configuration to JSON.
    
    This extracts key parameters from the policy config (horizon, dimensions, scheduler settings)
    and saves them to `onnx_config.json` for use by the inference node.
    
    Args:
        policy: The loaded LeRobot policy.
        output_dir: Directory to save the config file.
        
    Returns:
        Path to the saved config file.
    """
    print("\n" + "="*80)
    print("SAVING CONFIGURATION")
    print("="*80)
    
    # Get configuration (convert to dict if it's a config object)
    config = policy.config
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    # Extract relevant config - use getattr for objects
    # Count cameras
    if hasattr(config, "input_features"):
        input_features = config.input_features
        # Check for dict first (more common case)
        if isinstance(input_features, dict):
            n_cameras = sum(1 for key in input_features if "images" in key)
            if "observation.state" in input_features:
                state_feature = input_features["observation.state"]
                if isinstance(state_feature, dict):
                    state_dim = state_feature.get("shape", [6])[0]
                elif hasattr(state_feature, "shape") and state_feature.shape is not None:
                    # Handle PolicyFeature objects with shape attribute
                    state_dim = state_feature.shape[0]
                else:
                    state_dim = 6
            else:
                state_dim = 6
        elif hasattr(input_features, "__dict__"):
            n_cameras = sum(1 for key in vars(input_features) if "images" in key)
            # Get state dim
            if hasattr(input_features, "observation") and hasattr(input_features.observation, "state"):
                state_dim = input_features.observation.state.shape[0]
            else:
                state_dim = 6
        else:
            n_cameras = 2
            state_dim = 6
    else:
        n_cameras = 2
        state_dim = 6
    
    # Get action dim
    if hasattr(config, "output_features"):
        output_features = config.output_features
        action_dim = 6  # Default value

        # Try dictionary access first (handles dotted keys better)
        if isinstance(output_features, dict):
            # Check for 'action' key
            action_config = output_features.get("action", {})
            if isinstance(action_config, dict):
                action_dim = action_config.get("shape", [6])[0]
            elif hasattr(action_config, "shape") and action_config.shape is not None:
                # Handle PolicyFeature objects with shape attribute
                action_dim = action_config.shape[0]
        # Then try object access
        elif hasattr(output_features, "action"):
            action_feature = output_features.action
            if hasattr(action_feature, "shape") and action_feature.shape is not None:
                action_dim = action_feature.shape[0]
    else:
        action_dim = 6
    
    spatial_softmax_keypoints = getattr(config, "spatial_softmax_num_keypoints", 32)
    img_feature_dim = spatial_softmax_keypoints * 2 * n_cameras
    
    # Get crop shape
    crop_shape = getattr(config, "crop_shape", None)
    if crop_shape and hasattr(crop_shape, "__iter__"):
        crop_shape = list(crop_shape)
    
    onnx_config = {
        "horizon": getattr(config, "horizon", 16),
        "n_obs_steps": getattr(config, "n_obs_steps", 2),
        "n_action_steps": getattr(config, "n_action_steps", 8),
        "action_dim": action_dim,
        "state_dim": state_dim,
        "n_cameras": n_cameras,
        "image_features": True,
        "img_feature_dim": img_feature_dim,
        "crop_shape": crop_shape,
        "vision_backbone": getattr(config, "vision_backbone", "resnet18"),
        "spatial_softmax_num_keypoints": spatial_softmax_keypoints,
        
        # Scheduler config
        "noise_scheduler_type": "DDIM",  # Default to DDIM for inference
        "num_train_timesteps": getattr(config, "num_train_timesteps", 100),
        "num_inference_steps": 16,  # Default DDIM 16 steps
        "beta_schedule": getattr(config, "beta_schedule", "squaredcos_cap_v2"),
        "beta_start": getattr(config, "beta_start", 0.0001),
        "beta_end": getattr(config, "beta_end", 0.02),
        "prediction_type": getattr(config, "prediction_type", "epsilon"),
        "clip_sample": getattr(config, "clip_sample", True),
        "clip_sample_range": getattr(config, "clip_sample_range", 1.0),
    }
    
    output_path = output_dir / "onnx_config.json"
    with open(output_path, 'w') as f:
        json.dump(onnx_config, f, indent=2)
    
    print(f"✓ Configuration saved to {output_path}")
    print(f"\nConfiguration summary:")
    for key, value in onnx_config.items():
        print(f"  {key}: {value}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert Approach Real Bing 20Hz model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="approach_real_bing_20hz_lerobot/testing_abs/onnx_models",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model loading"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("APPROACH REAL BING 20HZ MODEL - ONNX CONVERSION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset version: {args.opset_version}")
    print(f"Device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not (checkpoint_path / "pretrained_model").exists():
        # Look for output/checkpoints/last structure
        checkpoint_path = checkpoint_path / "output" / "checkpoints" / "last"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint at {args.checkpoint}")
    
    model_path = checkpoint_path / "pretrained_model"
    print(f"\nLoading model from: {model_path}")
    
    # Load policy
    policy = DiffusionPolicy.from_pretrained(str(model_path))
    policy.to(args.device)
    policy.eval()
    
    # Get config as dict
    if hasattr(policy.config, '__dict__'):
        config_dict = vars(policy.config)
    else:
        config_dict = policy.config
    
    print("✓ Policy loaded successfully")
    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})
    print(f"  Input features: {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")
    
    # Export models
    with torch.no_grad():
        rgb_encoder_path = export_rgb_encoder(policy, output_dir, args.opset_version)
        unet_path = export_unet(policy, output_dir, args.opset_version)
        config_path = save_config(policy, output_dir)
    
    # Verify ONNX models
    print("\n" + "="*80)
    print("VERIFYING ONNX MODELS")
    print("="*80)
    
    try:
        import onnx
        
        # Verify RGB encoder
        print("\nVerifying RGB encoder...")
        onnx_model = onnx.load(rgb_encoder_path)
        onnx.checker.check_model(onnx_model)
        print("✓ RGB encoder ONNX model is valid")
        
        # Verify UNet
        print("\nVerifying UNet...")
        onnx_model = onnx.load(unet_path)
        onnx.checker.check_model(onnx_model)
        print("✓ UNet ONNX model is valid")
        
    except ImportError:
        print("⚠️  ONNX package not found, skipping verification")
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {rgb_encoder_path}")
    print(f"  - {unet_path}")
    print(f"  - {config_path}")
    print(f"\nNext steps:")
    print(f"  1. Run evaluation with ONNX+TensorRT:")
    print(f"     python evaluate_predictions_onnx.py ")
    print(f"         --checkpoint {args.checkpoint} ")
    print(f"         --onnx-dir {args.output_dir} ")
    print(f"         --rosbag /path/to/rosbag ")
    print(f"         --plot")
    print("="*80)


if __name__ == "__main__":
    main()
