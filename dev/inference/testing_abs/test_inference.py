#!/usr/bin/env python3
"""
Test script for LeRobot inference with image preprocessing validation.

This script tests the inference module and validates that image preprocessing
matches the training pipeline.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from lerobot_inference import LeRobotInference


def create_test_image(height: int, width: int) -> np.ndarray:
    """Create a test RGB image with a pattern."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = int(255 * i / height)  # Red gradient
            image[i, j, 1] = int(255 * j / width)   # Green gradient
            image[i, j, 2] = 128                     # Constant blue
    
    return image


def test_image_preprocessing():
    """Test image preprocessing pipeline."""
    print("=" * 60)
    print("Testing Image Preprocessing Pipeline")
    print("=" * 60)
    
    # Create test images at typical ROS camera resolution
    print("\n1. Creating test images...")
    original_height, original_width = 480, 640
    test_image = create_test_image(original_height, original_width)
    print(f"   Original image size: {test_image.shape}")
    
    # Test top view preprocessing (crop + rotate)
    print("\n2. Testing top view preprocessing (crop + rotate)...")
    crop_box = (247, 122, 193, 237)
    x, y, w, h = crop_box
    
    # Crop
    cropped = test_image[y:y+h, x:x+w]
    print(f"   After crop: {cropped.shape} (expected: (237, 193, 3))")
    
    # Rotate 90° clockwise
    rotated = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    print(f"   After rotate: {rotated.shape} (expected: (193, 237, 3))")
    
    # Test left arm preprocessing (resize)
    print("\n3. Testing left arm preprocessing (resize)...")
    resized = cv2.resize(test_image, (237, 193), interpolation=cv2.INTER_AREA)
    print(f"   After resize: {resized.shape} (expected: (193, 237, 3))")
    
    # Verify both cameras output same size
    print("\n4. Verifying both cameras output same size...")
    if rotated.shape == resized.shape:
        print(f"   ✓ Both cameras output: {rotated.shape}")
    else:
        print(f"   ✗ Size mismatch! Top view: {rotated.shape}, Left arm: {resized.shape}")
        return False
    
    return True


def test_inference_module(checkpoint_path: str, device: str = "cuda"):
    """Test the inference module."""
    print("\n" + "=" * 60)
    print("Testing Inference Module")
    print("=" * 60)
    
    try:
        # Initialize inference
        print("\n1. Initializing inference system...")
        inference = LeRobotInference(checkpoint_path, device)
        
        print(f"   ✓ Model loaded successfully")
        print(f"   - Input mode: {inference.input_mode}")
        print(f"   - Output mode: {inference.output_mode}")
        print(f"   - Expected image size: {inference.expected_image_size}")
        print(f"   - Crop box: {inference.crop_box}")
        print(f"   - Target size: {inference.target_size}")
        
        # Create test images
        print("\n2. Creating test images...")
        height, width = 480, 640
        left_image = create_test_image(height, width)
        head_image = create_test_image(height, width)
        print(f"   ✓ Created test images: {left_image.shape}")
        
        # Test image preprocessing
        print("\n3. Testing image preprocessing...")
        
        # Simulate ROS CompressedImage message
        class MockCompressedImage:
            def __init__(self, image):
                # Encode image to JPEG
                _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.data = encoded.tobytes()
        
        left_msg = MockCompressedImage(left_image)
        head_msg = MockCompressedImage(head_image)
        
        # Decode and preprocess
        left_processed = inference.decode_compressed_image_msg(left_msg, is_top_view=False)
        head_processed = inference.decode_compressed_image_msg(head_msg, is_top_view=True)
        
        print(f"   ✓ Left arm processed: {left_processed.shape}")
        print(f"   ✓ Head cam processed: {head_processed.shape}")
        
        # Verify sizes match
        if left_processed.shape != head_processed.shape:
            print(f"   ✗ Size mismatch!")
            return False
        
        if left_processed.shape[:2] != inference.expected_image_size:
            print(f"   ✗ Size doesn't match expected: {inference.expected_image_size}")
            return False
        
        # Test inference (without joint state for vision_only mode)
        print("\n4. Testing inference...")
        
        if inference.input_mode == "vision_only":
            action = inference.predict(left_processed, head_processed)
        else:
            # Create dummy joint state
            if inference.input_mode == "vision_pos":
                joint_state = np.zeros(6, dtype=np.float32)
            else:  # vision_pos_vel
                joint_state = np.zeros(12, dtype=np.float32)
            
            action = inference.predict(left_processed, head_processed, joint_state)
        
        print(f"   ✓ Inference successful")
        print(f"   - Action shape: {action.shape}")
        print(f"   - Action values: {action}")
        
        # Verify action dimension
        expected_action_dim = inference.action_dim
        if action.shape[0] != expected_action_dim:
            print(f"   ✗ Action dimension mismatch! Expected {expected_action_dim}, got {action.shape[0]}")
            return False
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test LeRobot inference module")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--skip-preprocessing-test",
        action="store_true",
        help="Skip standalone preprocessing test"
    )
    
    args = parser.parse_args()
    
    # Run tests
    success = True
    
    if not args.skip_preprocessing_test:
        if not test_image_preprocessing():
            success = False
    
    if not test_inference_module(args.checkpoint, args.device):
        success = False
    
    # Print summary
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
