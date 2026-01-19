#!/usr/bin/env python3
"""
ROS Bag to LeRobot Dataset Converter

Converts ROS2 bag files to LeRobot dataset format with flexible input/output modes.
Supports parallel batch processing of multiple bags.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions

# LeRobot imports
sys.path.append(str(Path(__file__).parent))
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class ROSBag2Converter:
    """ROS2 bag to LeRobot dataset converter with flexible input/output modes."""
    
    def __init__(
        self,
        observation_topics: List[str],
        dataset_name: str,
        output_dir: str,
        action_topics: Optional[List[str]] = None,
        fps: int = 10,
        task_description: str = "Robot manipulation task",
        downsize_images: bool = True,
        tolerance_s: float = 1.0,
        trim_unmoving_end: bool = True,
        right_hand_joint_names: Optional[List[str]] = None,
        input_mode: str = "vision_pos_vel",
        output_mode: str = "pos_vel",
        force: bool = False
    ):
        """Initialize the converter."""
        self.observation_topics = observation_topics
        self.action_topics = action_topics or []
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.task_description = task_description
        self.downsize_images = downsize_images
        self.tolerance_s = tolerance_s
        self.trim_unmoving_end = trim_unmoving_end
        self.force = force
        
        # Input/output modes
        self.input_mode = input_mode
        self.output_mode = output_mode
        
        # Validate modes
        valid_input_modes = ["vision_only", "vision_pos", "vision_pos_vel"]
        valid_output_modes = ["pos_only", "pos_vel"]
        
        if input_mode not in valid_input_modes:
            raise ValueError(f"Invalid input_mode: {input_mode}. Must be one of {valid_input_modes}")
        if output_mode not in valid_output_modes:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be one of {valid_output_modes}")
        
        print(f"Converter modes - Input: {input_mode}, Output: {output_mode}")
        
        # Define right arm joint names with gripper at index 5
        self.right_hand_joint_names = right_hand_joint_names or [
            'ra_shoulder_pan_joint',     # 0
            'ra_shoulder_lift_joint',    # 1
            'ra_elbow_joint',            # 2
            'ra_wrist_1_joint',          # 3
            'ra_wrist_2_joint',          # 4
            'ra_robotiq_85_left_knuckle_joint', # 5 (Gripper)
            'ra_wrist_3_joint'           # 6
        ]
        
        # Topic mapping
        self.image_topics = [t for t in observation_topics if 'image' in t and 'compressed' in t]
        self.joint_state_topics = [t for t in observation_topics if 'joint_states' in t]
        self.action_command_topics = [t for t in self.action_topics if 'trajectory' in t or 'command' in t or 'cmd' in t]
        
        self.dataset = None
        self.episode_index = 0
        
        # Message type cache
        self._message_type_cache = {}
        
        # Episode mapping: episode_index -> bag_path
        self.episode_mapping = {}
        
    def get_message_type(self, topic_type: str):
        """Get message type class from string, with caching."""
        if topic_type not in self._message_type_cache:
            self._message_type_cache[topic_type] = get_message(topic_type)
        return self._message_type_cache[topic_type]
        
    def get_synchronized_messages(self, bag_path: str) -> List[List[Tuple]]:
        """
        Get pre-synchronized messages grouped by message index using rosbag2_py.
        Since all topics have the same count, we can align by index.
        """
        # Set up storage options
        storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        # Create reader
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic metadata
        topic_metadata = reader.get_all_topics_and_types()
        print(f"Available topics: {[t.name for t in topic_metadata]}")
        
        # Filter for relevant topics (both observation and action topics)
        all_topics = self.observation_topics + self.action_topics
        relevant_topics = {}
        for topic_info in topic_metadata:
            if topic_info.name in all_topics:
                relevant_topics[topic_info.name] = topic_info.type
        
        if not relevant_topics:
            print(f"No relevant topics found. Available: {[t.name for t in topic_metadata]}")
            return []
        
        print(f"Using topics: {list(relevant_topics.keys())}")
        
        # Set topic filter
        from rosbag2_py import StorageFilter
        storage_filter = StorageFilter(topics=list(relevant_topics.keys()))
        reader.set_filter(storage_filter)
        
        # Read all messages organized by topic
        topic_messages = {topic: [] for topic in relevant_topics.keys()}
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic in topic_messages:
                # Get message type and deserialize
                msg_type = self.get_message_type(relevant_topics[topic])
                # Deserialize the message - any failure will prevent this bag from being processed
                msg = deserialize_message(data, msg_type)
                topic_messages[topic].append((msg, timestamp))
        
        # Note: SequentialReader doesn't have a close() method
        
        # Verify all topics have the same message count
        message_counts = [len(msgs) for msgs in topic_messages.values()]
        unique_counts = set(message_counts)
        
        if len(unique_counts) != 1:
            min_count = min(message_counts)
            max_count = max(message_counts)
            count_diff = max_count - min_count
            
            print(f"Warning: Topics have different message counts: {dict(zip(topic_messages.keys(), message_counts))}")
            
            # If difference is small (up to 5 messages), truncate to shortest topic
            if count_diff <= 2:
                print(f"📏 Message count differs by {count_diff} - truncating all topics to {min_count} messages")
                
                # Truncate all topics to the minimum count
                for topic_name in topic_messages.keys():
                    if len(topic_messages[topic_name]) > min_count:
                        dropped_count = len(topic_messages[topic_name]) - min_count
                        topic_messages[topic_name] = topic_messages[topic_name][:min_count]
                        print(f"   Dropped {dropped_count} message(s) from {topic_name}")
                
                num_messages = min_count
                print(f"Synchronized all topics to {num_messages} messages")
            else:
                print(f"Message count difference ({count_diff}) too large - skipping episode")
                return []
        else:
            num_messages = message_counts[0]
            print(f"All topics have {num_messages} messages")
        
        # Group messages by index (0, 1, 2, ...)
        synchronized_frames = []
        for i in range(num_messages):
            frame_data = {}
            for topic_name, messages in topic_messages.items():
                msg, timestamp = messages[i]
                frame_data[topic_name] = (msg, timestamp)
            
            synchronized_frames.append(frame_data)
        
        return synchronized_frames
    
    def decode_compressed_image_from_msg(self, msg, downsize: bool = True) -> np.ndarray:
        """
        Decode compressed image from ROS CompressedImage message.
        Raises exception on any error - no fallback values.
        """
        # Extract image data from message
        image_data = bytes(msg.data)
        
        if not image_data:
            raise ValueError("Empty image data in message")
        
        # Decode image using OpenCV
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            raise ValueError("Failed to decode image data")
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # IMAGE DOWNSIZING: Reduce to half size for storage efficiency
        if downsize:
            height, width = image.shape[:2]
            new_height = height // 2
            new_width = width // 2
            # Use INTER_AREA for downsampling (best quality for shrinking)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_right_hand_state_from_msg(self, msg) -> Dict[str, np.ndarray]:
        """
        Extract right hand joint positions and velocities from ROS JointState message.
        Filters only right hand/arm joints and preserves original data types.
        Returns dict with 'position' and 'velocity' arrays.
        """
        # Get joint names, positions, and velocities
        if not hasattr(msg, 'name') or not msg.name:
            raise ValueError("Joint state message has no joint names")
        if not hasattr(msg, 'position') or not msg.position:
            raise ValueError("Joint state message has no position data")
        if not hasattr(msg, 'velocity') or not msg.velocity:
            # Gripper might not have velocity in some messages, handle gracefully
            velocities = np.zeros_like(msg.position)
        else:
            velocities = np.array(msg.velocity)
            
        joint_names = list(msg.name)
        positions = np.array(msg.position)
        
        # Find indices of right hand joints
        right_hand_indices = []
        for joint_name in self.right_hand_joint_names:
            if joint_name in joint_names:
                right_hand_indices.append(joint_names.index(joint_name))
        
        if not right_hand_indices:
            raise ValueError(f"No right hand joints found in message. Available: {joint_names}")
        
        # Extract right hand data
        right_positions = positions[right_hand_indices]
        right_velocities = velocities[right_hand_indices]
        
        # Validate data
        if not np.all(np.isfinite(right_positions)):
            raise ValueError("Right hand positions contain NaN or Inf values")
        if not np.all(np.isfinite(right_velocities)):
            # Warning only, as velocity might be less critical or expectedly NaN in some systems
            print("Warning: Right hand velocities contain NaN or Inf values")
            right_velocities = np.nan_to_num(right_velocities)
            
        # Preserve original data type from rosbag (usually float64)
        return {
            'position': right_positions.astype(positions.dtype),
            'velocity': right_velocities.astype(positions.dtype)
        }
    
    def extract_controller_command_from_msg(self, msg) -> np.ndarray:
        """
        Extract action commands from ROS JointTrajectory message based on output mode.
        Returns positions only for pos_only mode, or [positions, velocities] for pos_vel mode.
        Preserves original data types from rosbag.
        """
        if not hasattr(msg, 'points') or not msg.points:
            raise ValueError("JointTrajectory message has no trajectory points")
        
        # Use the first trajectory point (most recent command)
        first_point = msg.points[0]
        
        if not hasattr(first_point, 'positions') or not first_point.positions:
            raise ValueError("Trajectory point has no position commands")
            
        command_positions = np.array(first_point.positions)
        
        # Validate position data
        if not np.all(np.isfinite(command_positions)):
            raise ValueError("Controller position commands contain NaN or Inf values")
            
        # Return based on output mode
        if self.output_mode == "pos_only":
            # Return positions only
            return command_positions.astype(command_positions.dtype)
        
        elif self.output_mode == "pos_vel":
            # Extract velocities if available, otherwise use zeros
            if hasattr(first_point, 'velocities') and first_point.velocities:
                command_velocities = np.array(first_point.velocities)
            else:
                # If no velocities in the message, use zeros
                command_velocities = np.zeros_like(command_positions)
            
            # Validate velocity data
            if not np.all(np.isfinite(command_velocities)):
                raise ValueError("Controller velocity commands contain NaN or Inf values")
                
            # Combine positions and velocities
            combined_commands = np.concatenate([command_positions, command_velocities])
            return combined_commands.astype(command_positions.dtype)
        
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
            
    def extract_gripper_command_from_msg(self, msg) -> np.ndarray:
        """
        Extract gripper position command from ROS GripperCommand message.
        """
        if not hasattr(msg, 'command'):
             # Some versions might have msg.position directly, but control_msgs usually has .command
             if hasattr(msg, 'position'):
                 return np.array([msg.position])
             raise ValueError("GripperCommand message has no command or position attribute")
        
        return np.array([msg.command.position])
    
    def create_regular_timestamps(self, num_frames: int, start_time: float = None) -> np.ndarray:
        """
        Generate perfectly regular relative timestamps based on FPS.
        Uses relative timesteps starting from 0.0 for LeRobot datasets.
        This avoids LeRobot's strict timestamp validation errors.
        """
        dt = 1.0 / self.fps  # Time between frames
        # Use relative timestamps starting from 0.0 (ignore start_time parameter)
        return np.array([i * dt for i in range(num_frames)])
    
    def find_task_completion_point(self, joint_positions_sequence: List[np.ndarray]) -> int:
        """Find the point where the robot completed its task by detecting sustained low movement."""
        if not self.trim_unmoving_end or len(joint_positions_sequence) < 20:
            return len(joint_positions_sequence) - 1
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(joint_positions_sequence)):
            prev_pos = joint_positions_sequence[i-1]
            curr_pos = joint_positions_sequence[i]
            velocity = np.linalg.norm(curr_pos - prev_pos)
            velocities.append(velocity)
        
        if len(velocities) < 10:
            return len(joint_positions_sequence) - 1
        
        # Simple threshold-based detection
        threshold = 0.0001
        min_idle_frames = 10
        
        # Find sustained low movement from the end
        for i in range(len(velocities) - min_idle_frames, min_idle_frames, -1):
            if all(v <= threshold for v in velocities[i:i + min_idle_frames]):
                return min(i + min_idle_frames // 2, len(joint_positions_sequence) - 1)
        
        return len(joint_positions_sequence) - 1
    
    def create_dataset_features(self, sample_images: Dict[str, np.ndarray] = None) -> Dict:
        """
        Create feature definitions that match LeRobot's expected format.
        """
        features = {}
        
        # Add image features with actual dimensions
        for topic in self.image_topics:
            feature_key = self.topic_to_feature_key(topic, 'observation.images')
            
            if sample_images and topic in sample_images:
                height, width, channels = sample_images[topic].shape
                features[feature_key] = {
                    "dtype": "video",  # Use video format for efficiency
                    "shape": (channels, height, width),  # CHW format
                    "names": ["channel", "height", "width"]
                }
            else:
                # Default dimensions for downsized images
                features[feature_key] = {
                    "dtype": "video",
                    "shape": (3, 240, 424),  # Downsized from 480x848
                    "names": ["channel", "height", "width"]
                }
        
        # Add right hand joint state observation based on input mode
        if self.joint_state_topics and self.input_mode != "vision_only":
            num_right_joints = len(self.right_hand_joint_names)
            
            if self.input_mode == "vision_pos":
                # Position only
                features["observation.state"] = {
                    "dtype": "float32",  # Converted from rosbag float64
                    "shape": (num_right_joints,),  # position only for each joint
                    "names": [f"{joint}_pos" for joint in self.right_hand_joint_names]
                }
            elif self.input_mode == "vision_pos_vel":
                # Position and velocity
                features["observation.state"] = {
                    "dtype": "float32",  # Converted from rosbag float64
                    "shape": (num_right_joints * 2,),  # position + velocity for each joint
                    "names": ([f"{joint}_pos" for joint in self.right_hand_joint_names] + 
                             [f"{joint}_vel" for joint in self.right_hand_joint_names])
                }
        
        # Add action based on output mode
        num_right_joints = len(self.right_hand_joint_names)
        
        if self.output_mode == "pos_only":
            # Position only output
            features["action"] = {
                "dtype": "float32",  # Converted from rosbag float64
                "shape": (num_right_joints,),
                "names": [f"{joint}_pos_cmd" for joint in self.right_hand_joint_names]
            }
        elif self.output_mode == "pos_vel":
            # Position and velocity output
            features["action"] = {
                "dtype": "float32",  # Converted from rosbag float64
                "shape": (num_right_joints * 2,),
                "names": [f"{joint}_pos_cmd" for joint in self.right_hand_joint_names] + 
                        [f"{joint}_vel_cmd" for joint in self.right_hand_joint_names]
            }
        
        # Add required LeRobot features
        features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
        features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
        
        return features
    
    def topic_to_feature_key(self, topic: str, prefix: str) -> str:
        """Convert ROS topic name to clean LeRobot feature key."""
        # Clean up topic name for LeRobot format
        clean_name = topic.replace('/', '_').replace('__', '_').strip('_')
        clean_name = clean_name.replace('_sync', '')  # Remove sync suffix
        clean_name = clean_name.replace('emily01_', '')  # Remove robot name
        clean_name = clean_name.replace('_color_image_raw_compressed', '_cam')  # Shorten
        return f"{prefix}.{clean_name}"
    
    def save_episode_mapping(self, output_file: Optional[str] = None):
        """Save episode mapping to JSON file."""
        if not self.episode_mapping:
            print("No episode mapping to save")
            return
        
        if output_file is None:
            mapping_file = self.output_dir / "episode_mapping.json"
        else:
            mapping_file = Path(output_file)
        
        mapping_data = {
            "conversion_summary": {
                "total_episodes": len(self.episode_mapping),
                "dataset_name": self.dataset_name,
                "output_directory": str(self.output_dir),
                "conversion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "episode_mapping": self.episode_mapping
        }
        
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            print(f"Episode mapping saved to: {mapping_file}")
        except Exception as e:
            print(f"Failed to save episode mapping: {e}")
    
    def _extract_bag_worker(self, bag_path: str) -> Tuple:
        """Worker function for parallel bag extraction."""
        try:
            synchronized_frames = self.get_synchronized_messages(bag_path)
            if not synchronized_frames:
                print(f"❌ No synchronized frames found in {bag_path}")
                return None, None, False
            
            # Process frames
            processed_frames = []
            timestamps = self.create_regular_timestamps(len(synchronized_frames))
            
            for i, frame_data in enumerate(synchronized_frames):
                frame_dict = {}
                
                for topic_name, (msg, original_timestamp) in frame_data.items():
                    if topic_name in self.image_topics:
                        image = self.decode_compressed_image_from_msg(msg, downsize=self.downsize_images)
                        image = np.transpose(image, (2, 0, 1))
                        feature_key = self.topic_to_feature_key(topic_name, 'observation.images')
                        frame_dict[feature_key] = image
                    
                    elif topic_name in self.joint_state_topics and self.input_mode != "vision_only":
                        right_hand_state = self.extract_right_hand_state_from_msg(msg)
                        
                        if self.input_mode == "vision_pos":
                            frame_dict["observation.state"] = right_hand_state['position'].astype(np.float32)
                        elif self.input_mode == "vision_pos_vel":
                            combined_state = np.concatenate([
                                right_hand_state['position'],
                                right_hand_state['velocity']
                            ])
                            frame_dict["observation.state"] = combined_state.astype(np.float32)
                    
                    elif topic_name in self.action_command_topics:
                        if 'trajectory' in topic_name:
                            # Arm trajectory
                            arm_action = self.extract_controller_command_from_msg(msg)
                            # Store temporarily to combine with gripper later
                            frame_dict["_temp_arm_action"] = arm_action
                        elif 'gripper' in topic_name:
                            # Gripper command
                            gripper_action = self.extract_gripper_command_from_msg(msg)
                            # Store temporarily to combine with arm later
                            frame_dict["_temp_gripper_action"] = gripper_action
                
                # After iterating all topics for this frame, combine actions if both exist
                if "_temp_arm_action" in frame_dict:
                    arm_act = frame_dict.pop("_temp_arm_action")
                    if "_temp_gripper_action" in frame_dict:
                        gripper_act = frame_dict.pop("_temp_gripper_action")
                    else:
                        # Fallback if gripper missing: pad with 0
                        print(f"⚠️ Warning: Gripper action missing at frame {i}")
                        gripper_act = np.zeros(1, dtype=arm_act.dtype)
                    
                    # Interleave gripper at index 5 to match state
                    if self.output_mode == "pos_only":
                        # Arm has 6 pos, gripper has 1 pos
                        # Result: [arm_pos (5), gripper_pos (1), arm_pos (1)] -> Total 7
                        combined = np.concatenate([
                            arm_act[:5],
                            gripper_act,
                            arm_act[5:6]
                        ])
                    else:
                        # Arm has 12: [p0..p5, v0..v5]
                        # Gripper has 1: [gp] (assuming no vel command)
                        # Result: [p0..p4, gp, p5, v0..v4, gv, v5] -> Total 14
                        arm_pos = arm_act[:6]
                        arm_vel = arm_act[6:12]
                        gripper_pos = gripper_act[0:1]
                        gripper_vel = np.zeros_like(gripper_pos) # Gripper v-cmd unavailable
                        
                        combined = np.concatenate([
                            arm_pos[:5], gripper_pos, arm_pos[5:6],
                            arm_vel[:5], gripper_vel, arm_vel[5:6]
                        ])
                    
                    frame_dict["action"] = combined.astype(np.float32)
                elif "_temp_gripper_action" in frame_dict:
                    # Rare case: gripper but no arm
                    frame_dict.pop("_temp_gripper_action")
                    print(f"⚠️ Warning: Arm action missing but gripper present at frame {i}")
                
                processed_frames.append(frame_dict)
            
            return processed_frames, timestamps, True
        
        except Exception as e:
            print(f"❌ Error extracting {bag_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, False
    
    def convert_bags(self, bag_directories: List[str], num_workers: int = 4) -> None:
        """
        Convert multiple bag files to LeRobot dataset with parallel extraction.
        
        Args:
            bag_directories: List of paths to bag directories
            num_workers: Number of parallel workers for bag extraction (default: 4)
        """
        print(f"Converting {len(bag_directories)} bag files to LeRobot dataset (parallel mode, {num_workers} workers)")
        if not bag_directories:
            raise ValueError("No bag directories provided")
        
        # Initialize ROS2 (required for message deserialization)
        rclpy.init()
        
        try:
            # Analyze first bag for image dimensions
            sample_images = {}
            first_bag = bag_directories[0]
            print(f"📊 Analyzing first bag for features: {first_bag}")
            
            sample_frames = self.get_synchronized_messages(first_bag)
            
            # Get sample images from first few frames
            for frame_data in sample_frames[:5]:  # Check first 5 frames
                for topic_name, (msg, timestamp) in frame_data.items():
                    if topic_name in self.image_topics and topic_name not in sample_images:
                        try:
                            image = self.decode_compressed_image_from_msg(msg, downsize=self.downsize_images)
                            sample_images[topic_name] = image
                            print(f"📸 Sample image from {topic_name}: {image.shape}")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not decode sample image from {topic_name}: {e}")
                            continue
                
                if len(sample_images) >= len(self.image_topics):
                    break
            
            # Create dataset features
            features = self.create_dataset_features(sample_images)
            print(f"🏗️ Dataset features: {list(features.keys())}")
            
            # Create LeRobot dataset with relaxed tolerance
            if self.output_dir.exists():
                if not self.force:
                    print(f"❌ Error: Output directory {self.output_dir} already exists.")
                    print("To overwrite it, use the --force flag.")
                    return
                
                print(f"⚠️ Overwriting existing dataset at {self.output_dir}")
                import shutil
                shutil.rmtree(self.output_dir)
            
            self.dataset = LeRobotDataset.create(
                repo_id=self.dataset_name,
                fps=self.fps,
                features=features,
                root=self.output_dir,
                use_videos=True,
                image_writer_threads=4,
                tolerance_s=self.tolerance_s  # Relaxed tolerance
            )
            
            print(f"\n{'='*60}")
            print(f"📊 PHASE 1: Extracting frames from bags (parallel)")
            print(f"{'='*60}")
            
            # PHASE 1: Parallel extraction of frames from bags using threads
            extracted_frames = {}
            
            # Run parallel extraction with thread pool
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for bag_path in bag_directories:
                    future = executor.submit(self._extract_bag_worker, bag_path)
                    futures[future] = bag_path
                
                # Collect results as they complete
                successful_extractions = 0
                for future in as_completed(futures):
                    bag_path = futures[future]
                    try:
                        frames_data, timestamps, success = future.result()
                        if success:
                            extracted_frames[bag_path] = (frames_data, timestamps)
                            successful_extractions += 1
                            print(f"✅ Extracted: {Path(bag_path).name} ({len(frames_data)} frames)")
                        else:
                            print(f"❌ Failed to extract: {Path(bag_path).name}")
                    except Exception as e:
                        print(f"❌ Error extracting {Path(bag_path).name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\n✅ Extraction phase complete: {successful_extractions}/{len(bag_directories)} bags")
            
            # PHASE 2: Sequential writing to dataset (must be done in order)
            print(f"\n{'='*60}")
            print(f"💾 PHASE 2: Writing frames to dataset (sequential)")
            print(f"{'='*60}")
            
            successful_writes = 0
            total_episodes_to_write = len(extracted_frames)
            for i, bag_path in enumerate(bag_directories):
                if bag_path not in extracted_frames:
                    print(f"⏭️ Skipping {Path(bag_path).name} (extraction failed)")
                    continue
                
                frames_data, timestamps = extracted_frames[bag_path]
                
                print(f"\n--- 💾 Writing {successful_writes + 1}/{total_episodes_to_write} ({Path(bag_path).name}) ---")
                try:
                    if self._write_episode_worker(frames_data, timestamps):
                        successful_writes += 1
                        # Release memory by deleting processed frames
                        del extracted_frames[bag_path]
                except Exception as e:
                    print(f"❌ Error writing {bag_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n{'='*60}")
            print(f"🎉 Successfully converted {successful_writes}/{len(bag_directories)} bag files")
            print(f"{'='*60}")
            
            # Save episode mapping for reference
            self.save_episode_mapping()
            
        finally:
            # Cleanup
            try:
                if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer:
                    self.dataset.stop_image_writer()
            except Exception as e:
                print(f"⚠️ Warning during cleanup: {e}")
            
            # Shutdown ROS2
            rclpy.shutdown()
    
    def _write_episode_worker(self, frames_data: List[Dict], timestamps: List[float]) -> bool:
        """Write episode data to dataset (sequential, thread-safe)."""
        try:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.episode_index)
            
            for i, frame_dict in enumerate(frames_data):
                # Ensure required fields
                if "next.reward" not in frame_dict:
                    frame_dict["next.reward"] = np.array([0.0], dtype=np.float32)
                if "next.done" not in frame_dict:
                    frame_dict["next.done"] = np.array([False], dtype=bool)
                
                self.dataset.add_frame(frame_dict, self.task_description, timestamps[i])
            
            # Mark last frame as episode end
            if self.dataset.episode_buffer is not None and len(self.dataset.episode_buffer["next.done"]) > 0:
                self.dataset.episode_buffer["next.done"][-1] = np.array([True], dtype=bool)
            
            self.dataset.save_episode()
            
            print(f"✅ Saved episode {self.episode_index} with {len(frames_data)} frames")
            self.episode_index += 1
            return True
        
        except Exception as e:
            print(f"❌ Error writing frames: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self.dataset, 'episode_buffer') and self.dataset.episode_buffer:
                self.dataset.episode_buffer = None
            return False


def main():
    parser = argparse.ArgumentParser(description="Convert pre-synchronized ROS2 bags to LeRobot dataset using rosbag2_py")
    parser.add_argument(
        "bag_dirs",
        nargs="+",
        help="One or more directories containing bag files"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--dataset-name", "-n", default="rosbag_dataset", help="Name for the dataset")
    parser.add_argument("--fps", type=int, default=10, help="Dataset FPS (default: 10)")
    parser.add_argument("--task", default="Robot manipulation task", help="Task description")
    parser.add_argument("--no-downsize", action="store_true", help="Don't downsize images")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Timestamp tolerance in seconds")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to convert")
    
    # Trimming options
    parser.add_argument("--no-trim-unmoving", action="store_true", help="Don't trim post-task idle period")
    
    # Topic specification
    parser.add_argument("--observation-topics", nargs="+", 
                       default=[
                           "/sync/emily01/front/color/image_raw/compressed",
                           "/sync/emily01/head/color/image_raw/compressed", 
                           "/sync/joint_states"
                       ],
                       help="ROS topics for observations (two cameras + joint states)")
    
    parser.add_argument("--action-topics", nargs="+",
                       default=[
                           "/sync/ra_trajectory_controller/joint_trajectory",
                           "/sync/sns_right_gripper_cmd"
                       ],
                       help="ROS topics for action commands (arm trajectory + gripper command)")
    
    parser.add_argument("--right-hand-joints", nargs="+",
                       default=None,  # Will use default from converter: ra_shoulder_pan_joint, ...
                       help="Names of right arm joints to extract from joint states")
    
    # Flexible input/output mode arguments
    parser.add_argument("--input-mode",
                       choices=["vision_only", "vision_pos", "vision_pos_vel"],
                       default="vision_pos_vel",
                       help="Input mode: vision_only (no proprioception), vision_pos (vision + position), vision_pos_vel (vision + position + velocity)")
    
    parser.add_argument("--output-mode", 
                       choices=["pos_only", "pos_vel"],
                       default="pos_vel",
                       help="Output mode: pos_only (position commands), pos_vel (position + velocity commands)")
    
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers for bag extraction (default: 1 for sequential). "
                            "Parallel processing (2-8 workers) speeds up conversion but requires more RAM. "
                            "Each worker loads a full episode into memory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset if it exists")
    
    args = parser.parse_args()
    
    # Find bag directories from all provided parent dirs
    bag_directories = []
    for bag_base_dir_str in args.bag_dirs:
        bag_base_dir = Path(bag_base_dir_str)
        if not bag_base_dir.exists():
            print(f"❌ Error: Directory {bag_base_dir} does not exist")
            sys.exit(1)
        for item in bag_base_dir.iterdir():
            if item.is_dir():
                # Check for bag files (db3, bag, or mcap format)
                db3_files = list(item.glob("*.db3"))
                bag_files = list(item.glob("*.bag"))
                mcap_files = list(item.glob("*.mcap"))
                if db3_files or bag_files or mcap_files:
                    bag_directories.append(str(item))
    bag_directories.sort()
    # Limit episodes if requested
    if args.max_episodes:
        bag_directories = bag_directories[:args.max_episodes]
    print(f"Found {len(bag_directories)} bag directories")
    if not bag_directories:
        print("No bag directories found")
        sys.exit(1)
    
    # Create converter
    converter = ROSBag2Converter(
        observation_topics=args.observation_topics,
        action_topics=args.action_topics,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        fps=args.fps,
        task_description=args.task,
        downsize_images=not args.no_downsize,
        tolerance_s=args.tolerance,
        trim_unmoving_end=not args.no_trim_unmoving,
        right_hand_joint_names=args.right_hand_joints,
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        force=args.force
    )
    
    start_time = time.time()
    converter.convert_bags(bag_directories, num_workers=args.num_workers)
    end_time = time.time()
    
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
