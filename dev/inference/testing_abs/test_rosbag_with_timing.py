#!/usr/bin/env python3
"""
Test LeRobot inference with rosbag data with detailed timing instrumentation.

This script extends test_rosbag.py to add detailed timing for:
- Total inference time
- generate_actions() method time
- conditional_sample() method time
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


def patch_policy_for_timing(inference_obj):
    """Patch the policy's diffusion model to add timing instrumentation."""
    original_generate_actions = inference_obj.policy.diffusion.generate_actions
    original_conditional_sample = inference_obj.policy.diffusion.conditional_sample
    
    timing_data = {
        'generate_actions_times': [],
        'conditional_sample_times': []
    }
    
    def timed_generate_actions(batch):
        start = time.perf_counter()
        result = original_generate_actions(batch)
        elapsed = time.perf_counter() - start
        timing_data['generate_actions_times'].append(elapsed)
        return result
    
    def timed_conditional_sample(batch_size, global_cond=None, generator=None):
        start = time.perf_counter()
        result = original_conditional_sample(batch_size, global_cond, generator)
        elapsed = time.perf_counter() - start
        timing_data['conditional_sample_times'].append(elapsed)
        return result
    
    inference_obj.policy.diffusion.generate_actions = timed_generate_actions
    inference_obj.policy.diffusion.conditional_sample = timed_conditional_sample
    
    return timing_data


class RosbagTesterWithTiming:
    """Test inference system with rosbag data and detailed timing."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the tester.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
        """
        self.inference = LeRobotInference(checkpoint_path, device)
        
        # Add timing instrumentation
        self.timing_data = patch_policy_for_timing(self.inference)
        
        # Get scheduler info
        scheduler_type = self.inference.policy.diffusion.noise_scheduler.__class__.__name__
        num_steps = self.inference.policy.diffusion.num_inference_steps
        print(f"\nScheduler: {scheduler_type}")
        print(f"Inference steps: {num_steps}")
        
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
            'total_inference_times': [],
            'actions': []
        }
        
        print(f"Running {num_tests} inference tests...")
        print(f"Image preprocessing: Head camera cropped (247,122,193,237) + rotated 90° CW → 237x193")
        print(f"                     Left camera resized → 237x193")
        
        # Reset timing data
        self.timing_data['generate_actions_times'] = []
        self.timing_data['conditional_sample_times'] = []
        
        for i in range(num_tests):
            try:
                # Get synchronized messages
                left_msg = messages[self.topics['left_image']][i]
                head_msg = messages[self.topics['head_image']][i]
                joint_msg = messages[self.topics['joint_state']][i]
                
                # Run inference (handle vision_only mode)
                start_time = time.perf_counter()
                if self.inference.input_mode == "vision_only":
                    action = self.inference.predict_from_ros_messages(left_msg, head_msg, None)
                else:
                    action = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
                total_time = time.perf_counter() - start_time
                
                # Get timing for this iteration
                gen_time = self.timing_data['generate_actions_times'][-1] if self.timing_data['generate_actions_times'] else 0
                sample_time = self.timing_data['conditional_sample_times'][-1] if self.timing_data['conditional_sample_times'] else 0
                
                # Record results
                results['successful_tests'] += 1
                results['total_inference_times'].append(total_time)
                results['actions'].append(action)
                
                # Display action based on output mode with detailed timing
                if self.inference.output_mode == "pos_only" and len(action) == 6:
                    pos_sample = action[:3]
                    print(f"Test {i+1}: SUCCESS - Pos: {pos_sample}")
                    print(f"         Total: {total_time*1000:.1f}ms | generate_actions: {gen_time*1000:.1f}ms | conditional_sample: {sample_time*1000:.1f}ms")
                elif self.inference.output_mode == "pos_vel" and len(action) == 12:
                    pos_sample = action[:3]
                    vel_sample = action[6:9]
                    print(f"Test {i+1}: SUCCESS - Pos: {pos_sample} Vel: {vel_sample}")
                    print(f"         Total: {total_time*1000:.1f}ms | generate_actions: {gen_time*1000:.1f}ms | conditional_sample: {sample_time*1000:.1f}ms")
                else:
                    print(f"Test {i+1}: SUCCESS - Action ({len(action)}D): {action[:3]}")
                    print(f"         Total: {total_time*1000:.1f}ms | generate_actions: {gen_time*1000:.1f}ms | conditional_sample: {sample_time*1000:.1f}ms")
                
            except Exception as e:
                results['failed_tests'] += 1
                print(f"Test {i+1}: FAILED - {e}")
        
        # Calculate statistics
        if results['total_inference_times']:
            results['total_inference_stats'] = {
                'mean': np.mean(results['total_inference_times']) * 1000,
                'std': np.std(results['total_inference_times']) * 1000,
                'min': np.min(results['total_inference_times']) * 1000,
                'max': np.max(results['total_inference_times']) * 1000,
            }
        
        if self.timing_data['generate_actions_times']:
            results['generate_actions_stats'] = {
                'mean': np.mean(self.timing_data['generate_actions_times']) * 1000,
                'std': np.std(self.timing_data['generate_actions_times']) * 1000,
                'min': np.min(self.timing_data['generate_actions_times']) * 1000,
                'max': np.max(self.timing_data['generate_actions_times']) * 1000,
            }
        
        if self.timing_data['conditional_sample_times']:
            results['conditional_sample_stats'] = {
                'mean': np.mean(self.timing_data['conditional_sample_times']) * 1000,
                'std': np.std(self.timing_data['conditional_sample_times']) * 1000,
                'min': np.min(self.timing_data['conditional_sample_times']) * 1000,
                'max': np.max(self.timing_data['conditional_sample_times']) * 1000,
            }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary with detailed timing."""
        print("\n" + "="*70)
        print("TEST SUMMARY - APPROACH PLATE PIPELINE (WITH TIMING)")
        print("="*70)
        print(f"Total tests: {results['total_tests']}")
        print(f"Successful: {results['successful_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success rate: {(results['successful_tests']/results['total_tests'])*100:.1f}%")
        
        if 'total_inference_stats' in results:
            stats = results['total_inference_stats']
            print(f"\nTotal inference time: {stats['mean']:.1f} ± {stats['std']:.1f} ms")
            print(f"  Min: {stats['min']:.1f} ms, Max: {stats['max']:.1f} ms")
        
        if 'generate_actions_stats' in results:
            stats = results['generate_actions_stats']
            print(f"\ngenerate_actions() time: {stats['mean']:.1f} ± {stats['std']:.1f} ms")
            print(f"  Min: {stats['min']:.1f} ms, Max: {stats['max']:.1f} ms")
        
        if 'conditional_sample_stats' in results:
            stats = results['conditional_sample_stats']
            print(f"\nconditional_sample() time: {stats['mean']:.1f} ± {stats['std']:.1f} ms")
            print(f"  Min: {stats['min']:.1f} ms, Max: {stats['max']:.1f} ms")
        
        print("\nImage Processing Applied:")
        print("  - Head camera: Crop (247,122,193,237) + Rotate 90° CW → 237x193")
        print("  - Left camera: Resize → 237x193")
        print("="*70)


def main():
    """Main function for rosbag testing with timing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LeRobot inference with rosbag data (with detailed timing)')
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
        print("="*70)
        print("APPROACH PLATE PIPELINE - ROSBAG TESTING WITH TIMING")
        print("="*70)
        tester = RosbagTesterWithTiming(args.checkpoint, args.device)
        results = tester.test_rosbag(args.rosbag, args.num_tests)
        tester.print_summary(results)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
