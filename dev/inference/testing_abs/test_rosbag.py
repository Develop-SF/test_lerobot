#!/usr/bin/env python3
"""
Test LeRobot inference with rosbag data for approach_plate pipeline.

This script loads data directly from rosbag files and tests the inference pipeline
with the cropped/rotated image preprocessing.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ROS2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available. Cannot test with rosbag data.")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from lerobot_inference import LeRobotInference


class RosbagTester:
    """Test inference system with rosbag data."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the tester.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
        """
        self.inference = LeRobotInference(checkpoint_path, device)
        
        # Topic names to extract
        self.topics = {
            'left_image': '/sync/emily01/left_arm/color/image_raw/compressed',
            'head_image': '/sync/emily01/head/color/image_raw/compressed',
            'joint_state': '/sync/joint_states'
        }
    
    def load_rosbag_messages(self, rosbag_path: str, max_messages: int = 50) -> Dict[str, List]:
        """
        Load messages from rosbag file.
        
        Args:
            rosbag_path: Path to rosbag directory
            max_messages: Maximum number of messages to load
            
        Returns:
            Dictionary with lists of messages for each topic
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS2 not available")
        
        # Setup rosbag reader
        storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Collect messages by topic
        messages = {topic: [] for topic in self.topics.values()}
        
        message_count = 0
        while reader.has_next() and message_count < max_messages * len(self.topics):
            topic, data, timestamp = reader.read_next()
            
            if topic in messages:
                # Determine message type
                if 'image_raw/compressed' in topic:
                    msg_type = get_message('sensor_msgs/msg/CompressedImage')
                elif 'joint_states' in topic:
                    msg_type = get_message('sensor_msgs/msg/JointState')
                else:
                    continue
                
                # Deserialize and store message
                msg = deserialize_message(data, msg_type)
                messages[topic].append(msg)
            
            message_count += 1
        
        return messages
    
    def test_rosbag(self, rosbag_path: str, num_tests: int = 5) -> Dict[str, Any]:
        """
        Test inference with rosbag data.
        
        Args:
            rosbag_path: Path to rosbag directory
            num_tests: Number of inference tests to run
            
        Returns:
            Test results dictionary
        """
        print(f"Loading messages from: {rosbag_path}")
        
        # Load messages
        messages = self.load_rosbag_messages(rosbag_path)
        
        # Check message counts
        left_count = len(messages[self.topics['left_image']])
        head_count = len(messages[self.topics['head_image']])
        joint_count = len(messages[self.topics['joint_state']])
        
        print(f"Loaded messages - Left: {left_count}, Head: {head_count}, Joints: {joint_count}")
        
        # Find minimum count for synchronized testing
        min_count = min(left_count, head_count, joint_count)
        if min_count == 0:
            raise ValueError("No synchronized messages found")
        
        # Run inference tests
        num_tests = min(num_tests, min_count)
        results = {
            'total_tests': num_tests,
            'successful_tests': 0,
            'failed_tests': 0,
            'inference_times': [],
            'actions': []
        }
        
        print(f"Running {num_tests} inference tests...")
        print(f"Image preprocessing: Head camera cropped (247,122,193,237) + rotated 90° CW → 237x193")
        print(f"                     Left camera resized → 237x193")
        
        for i in range(num_tests):
            try:
                # Get synchronized messages
                left_msg = messages[self.topics['left_image']][i]
                head_msg = messages[self.topics['head_image']][i]
                joint_msg = messages[self.topics['joint_state']][i]
                
                # Run inference (handle vision_only mode)
                start_time = time.time()
                if self.inference.input_mode == "vision_only":
                    action = self.inference.predict_from_ros_messages(left_msg, head_msg, None)
                else:
                    action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
                inference_time = time.time() - start_time
                
                # Record results
                results['successful_tests'] += 1
                results['inference_times'].append(inference_time)
                results['actions'].append(action)
                
                # Display action based on output mode
                if self.inference.output_mode == "pos_only" and len(action) == 6:
                    pos_sample = action[:3]
                    print(f"Test {i+1}: SUCCESS - Pos: {pos_sample} Time: {inference_time*1000:.1f}ms")
                elif self.inference.output_mode == "pos_vel" and len(action) == 12:
                    pos_sample = action[:3]
                    vel_sample = action[6:9]
                    print(f"Test {i+1}: SUCCESS - Pos: {pos_sample} Vel: {vel_sample} Time: {inference_time*1000:.1f}ms")
                else:
                    print(f"Test {i+1}: SUCCESS - Action ({len(action)}D): {action[:3]} Time: {inference_time*1000:.1f}ms")
                
            except Exception as e:
                results['failed_tests'] += 1
                print(f"Test {i+1}: FAILED - {e}")
        
        # Calculate statistics
        if results['inference_times']:
            results['avg_inference_time'] = np.mean(results['inference_times'])
            results['min_inference_time'] = np.min(results['inference_times'])
            results['max_inference_time'] = np.max(results['inference_times'])
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "="*50)
        print("TEST SUMMARY - APPROACH PLATE PIPELINE")
        print("="*50)
        print(f"Total tests: {results['total_tests']}")
        print(f"Successful: {results['successful_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success rate: {(results['successful_tests']/results['total_tests'])*100:.1f}%")
        
        if results['inference_times']:
            print(f"Average inference time: {results['avg_inference_time']*1000:.1f}ms")
            print(f"Min inference time: {results['min_inference_time']*1000:.1f}ms")
            print(f"Max inference time: {results['max_inference_time']*1000:.1f}ms")
        
        print("\nImage Processing Applied:")
        print("  - Head camera: Crop (247,122,193,237) + Rotate 90° CW → 237x193")
        print("  - Left camera: Resize → 237x193")
        print("="*50)


def main():
    """Main function for rosbag testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LeRobot inference with rosbag data (approach_plate pipeline)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot',
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--rosbag',
        type=str,
        required=True,
        help='Path to rosbag directory (e.g., /path/to/approach_plate/sync_approach_plate_...)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=5,
        help='Number of inference tests to run'
    )
    
    args = parser.parse_args()
    
    if not ROS_AVAILABLE:
        print("Error: ROS2 not available. Cannot test with rosbag data.")
        return
    
    try:
        # Create tester and run tests
        print("="*50)
        print("APPROACH PLATE PIPELINE - ROSBAG TESTING")
        print("="*50)
        tester = RosbagTester(args.checkpoint, args.device)
        results = tester.test_rosbag(args.rosbag, args.num_tests)
        tester.print_summary(results)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
