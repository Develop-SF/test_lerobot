#!/usr/bin/env python3
"""
Evaluate model predictions against ground truth from rosbag data.

This script compares model predictions with actual joint trajectories
from the rosbag to assess model performance.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ROS2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available. Cannot evaluate with rosbag data.")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from lerobot_inference import LeRobotInference


class PredictionEvaluator:
    """Evaluate model predictions against ground truth."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        use_relative_actions: bool = False,
        arm_dim: int = 6,
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            use_relative_actions: If True, model was trained with relative actions (PD2.1 + PD2.2)
            arm_dim: Number of arm joints (excluding gripper)
        """
        self.inference = LeRobotInference(
            checkpoint_path,
            device,
            use_relative_actions=use_relative_actions,
            arm_dim=arm_dim,
        )
        
        # Topic names
        self.topics = {
            'left_image': '/sync/emily01/left_arm/color/image_raw/compressed',
            'head_image': '/sync/emily01/head/color/image_raw/compressed',
            'joint_state': '/sync/joint_states',
            'action_command': '/sync/la_trajectory_controller/joint_trajectory'
        }
        
        # Left arm joint names for extraction
        self.left_arm_joints = [
            'la_shoulder_pan_joint',
            'la_shoulder_lift_joint',
            'la_elbow_joint',
            'la_wrist_1_joint',
            'la_wrist_2_joint',
            'la_wrist_3_joint'
        ]
    
    def load_rosbag_data(self, rosbag_path: str, max_samples: int = 100) -> Dict[str, List]:
        """
        Load synchronized data from rosbag.
        
        Args:
            rosbag_path: Path to rosbag directory
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with synchronized messages
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
        
        # Track messages per topic to stop when we have enough
        while reader.has_next():
            # Check if we have enough messages for all topics
            min_topic_count = min(len(msgs) for msgs in messages.values())
            if min_topic_count >= max_samples:
                break
            
            topic, data, timestamp = reader.read_next()
            
            if topic in messages:
                # Determine message type
                if 'image_raw/compressed' in topic:
                    msg_type = get_message('sensor_msgs/msg/CompressedImage')
                elif 'joint_states' in topic:
                    msg_type = get_message('sensor_msgs/msg/JointState')
                elif 'joint_trajectory' in topic:
                    msg_type = get_message('trajectory_msgs/msg/JointTrajectory')
                else:
                    continue
                
                # Deserialize and store message
                msg = deserialize_message(data, msg_type)
                messages[topic].append(msg)
        
        return messages
    
    def extract_ground_truth_positions(self, joint_msg) -> np.ndarray:
        """Extract left arm positions from joint state message."""
        joint_names = list(joint_msg.name)
        positions = np.array(joint_msg.position, dtype=np.float32)
        
        # Find indices of left arm joints
        left_arm_indices = []
        for joint_name in self.left_arm_joints:
            if joint_name in joint_names:
                left_arm_indices.append(joint_names.index(joint_name))
        
        if len(left_arm_indices) != 6:
            raise ValueError(f"Expected 6 left arm joints, found {len(left_arm_indices)}")
        
        return positions[left_arm_indices]
    
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
        """
        Evaluate model predictions against ground truth.
        
        Args:
            rosbag_path: Path to rosbag directory
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Loading data from: {rosbag_path}")
        
        # Load messages
        messages = self.load_rosbag_data(rosbag_path, num_samples)
        
        # Check message counts
        left_count = len(messages[self.topics['left_image']])
        head_count = len(messages[self.topics['head_image']])
        joint_count = len(messages[self.topics['joint_state']])
        action_count = len(messages[self.topics['action_command']])
        
        print(f"Loaded messages - Left: {left_count}, Head: {head_count}, Joints: {joint_count}, Actions: {action_count}")
        
        # Find minimum count for synchronized evaluation
        min_count = min(left_count, head_count, joint_count, action_count)
        if min_count == 0:
            raise ValueError("No synchronized messages found")
        
        num_samples = min(num_samples, min_count)
        
        # Storage for predictions and ground truth
        predictions = []
        ground_truth = []
        
        print(f"Evaluating {num_samples} samples...")
        print(f"Image preprocessing: Head camera cropped + rotated, Left camera resized → {self.inference.target_size[0]}x{self.inference.target_size[1]}")
        
        for i in range(num_samples):
            try:
                # Get synchronized messages
                left_msg = messages[self.topics['left_image']][i]
                head_msg = messages[self.topics['head_image']][i]
                joint_msg = messages[self.topics['joint_state']][i]
                action_msg = messages[self.topics['action_command']][i]
                
                # Get prediction
                if self.inference.input_mode == "vision_only":
                    pred = self.inference.predict_from_ros_messages(left_msg, head_msg, None)
                else:
                    pred = self.inference.predict_from_ros_messages(left_msg, head_msg, joint_msg)
                
                # Extract ground truth (next state positions)
                # For evaluation, we compare predicted positions with actual positions
                # gt_positions = self.extract_ground_truth_positions(joint_msg)
                gt_positions = self.extract_ground_truth_from_command(action_msg)
                
                # Store results (only positions for comparison)
                if self.inference.output_mode == "pos_only":
                    predictions.append(pred)  # 6D positions
                elif self.inference.output_mode == "pos_vel":
                    predictions.append(pred[:6])  # Extract positions from 12D output
                
                ground_truth.append(gt_positions)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples")
                
            except Exception as e:
                print(f"Sample {i+1} failed: {e}")
                continue
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        results = self.calculate_metrics(predictions, ground_truth)
        
        return results
    
    def calculate_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted joint positions (N, 6)
            ground_truth: Ground truth joint positions (N, 6)
            
        Returns:
            Dictionary of metrics
        """
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - ground_truth), axis=0)
        mae_overall = np.mean(mae)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=0))
        rmse_overall = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        
        # Maximum error
        max_error = np.max(np.abs(predictions - ground_truth), axis=0)
        max_error_overall = np.max(np.abs(predictions - ground_truth))
        
        # Standard deviation of errors
        errors = predictions - ground_truth
        std_error = np.std(errors, axis=0)
        std_error_overall = np.std(errors)
        
        results = {
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
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS - APPROACH PLATE PIPELINE")
        print("="*60)
        print(f"Number of samples: {results['num_samples']}")
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
        
        print("\nImage Processing Applied:")
        print(f"  - Head camera: Crop {self.inference.crop_box} + Rotate 90° CW → {self.inference.target_size[0]}x{self.inference.target_size[1]}")
        print(f"  - Left camera: Resize → {self.inference.target_size[0]}x{self.inference.target_size[1]}")
        print("="*60)
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save the plot
        """
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Prediction vs Ground Truth - Approach Plate Pipeline', fontsize=16)
        
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
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate LeRobot predictions (approach_plate pipeline)')
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
        default=300,
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
    parser.add_argument(
        '--use_relative_actions',
        action='store_true',
        help='Enable relative action mode (must match training mode). Model predicts relative actions.'
    )
    parser.add_argument(
        '--arm_dim',
        type=int,
        default=6,
        help='Number of arm joints (excluding gripper). Default: 6'
    )
    
    args = parser.parse_args()
    
    if not ROS_AVAILABLE:
        print("Error: ROS2 not available. Cannot evaluate with rosbag data.")
        return
    
    try:
        # Create evaluator and run evaluation
        print("="*60)
        print("APPROACH PLATE PIPELINE - PREDICTION EVALUATION")
        if args.use_relative_actions:
            print("RELATIVE ACTION MODE ENABLED")
            print(f"  - Arm dimension: {args.arm_dim}")
            print(f"  - PD2.1: Actions relative to current position")
            print(f"  - PD2.2: Observations relative to current position")
        else:
            print("ABSOLUTE ACTION MODE")
        print("="*60)
        evaluator = PredictionEvaluator(
            args.checkpoint,
            args.device,
            use_relative_actions=args.use_relative_actions,
            arm_dim=args.arm_dim,
        )
        results = evaluator.evaluate(args.rosbag, args.num_samples)
        evaluator.print_results(results)
        
        # Generate plots if requested
        if args.plot:
            evaluator.plot_results(results, args.save_plot)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
