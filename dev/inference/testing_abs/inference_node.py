#!/usr/bin/env python3
"""
ROS2 Inference Node for LeRobot Deployment - Approach Plate

This node subscribes to sensor topics, runs inference, and publishes actions.
Includes image preprocessing matching the training pipeline:
- Top view (head camera): Crop (260, 135, 178, 224) → Rotate 90° CW → 224×178
- Left arm camera: Resize to 224×178
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from lerobot_inference import LeRobotInference


class InferenceNode(Node):
    """ROS2 node for LeRobot inference deployment with image preprocessing."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", inference_frequency: float = 20.0, mode: str = "continuous"):
        """
        Initialize the ROS inference node.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            inference_frequency: Continuous inference frequency (Hz)
            mode: Inference mode - 'continuous' or 'triggered'
        """
        super().__init__('lerobot_inference_node')
        
        # Initialize inference system
        self.inference = LeRobotInference(checkpoint_path, device)
        
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
        
        self.get_logger().info("LeRobot Inference Node ready")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='LeRobot ROS2 Inference Node - Approach Plate')
    parser.add_argument(
        '--checkpoint', 
        type=str,
        default='/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot',
        help='Path to model checkpoint directory'
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
            device=parsed_args.device,
            inference_frequency=parsed_args.frequency,
            mode=parsed_args.mode
        )
        
        print("LeRobot Inference Node started (Approach Plate)")
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
    finally:
        # Cleanup
        if 'node' in locals():
            node.stop_inference_loop()
            node.destroy_node()
        
        rclpy.shutdown()
        print("Node stopped")


if __name__ == '__main__':
    main()
