#!/usr/bin/env python3
"""
ROS Bag to LeRobot Dataset Converter with Cropping and Trimming

Converts ROS2 bag files to LeRobot dataset format with:
- Top view image cropping (bounding box)
- Episode-specific frame trimming
- Flexible input/output modes
- Parallel batch processing of multiple bags
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


class ROSBag2ConverterCroppedTrimmed:
    """ROS2 bag to LeRobot dataset converter with cropping and trimming."""
    
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
        left_hand_joint_names: Optional[List[str]] = None,
        input_mode: str = "vision_pos_vel",
        output_mode: str = "pos_vel",
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        episode_trim_frames: Optional[Dict[int, Tuple[int, int]]] = None,
        episode_mapping_file: Optional[str] = None,
        skip_episodes: Optional[List[int]] = None
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
        
        # Input/output modes
        self.input_mode = input_mode
        self.output_mode = output_mode
        
        # Cropping and trimming
        self.crop_box = crop_box  # (x, y, w, h) for top view
        self.episode_trim_frames = episode_trim_frames or {}
        self.skip_episodes = set(skip_episodes or [])
        
        # Episode mapping
        self.episode_mapping_file = episode_mapping_file
        self.loaded_episode_mapping = None
        
        # Validate modes
        valid_input_modes = ["vision_only", "vision_pos", "vision_pos_vel"]
        valid_output_modes = ["pos_only", "pos_vel"]
        
        if input_mode not in valid_input_modes:
            raise ValueError(f"Invalid input_mode: {input_mode}. Must be one of {valid_input_modes}")
        if output_mode not in valid_output_modes:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be one of {valid_output_modes}")
        
        print(f"Converter modes - Input: {input_mode}, Output: {output_mode}")
        
        if self.crop_box:
            x, y, w, h = self.crop_box
            print(f"Top view crop box: x={x}, y={y}, w={w}, h={h}")
        
        if self.skip_episodes:
            print(f"Skipping episodes: {sorted(self.skip_episodes)}")
        
        # Define left arm joint names
        self.left_hand_joint_names = left_hand_joint_names or [
            'la_shoulder_pan_joint',
            'la_shoulder_lift_joint',
            'la_elbow_joint',
            'la_wrist_1_joint',
            'la_wrist_2_joint',
            'la_wrist_3_joint'
        ]
        
        # Topic mapping
        self.image_topics = [t for t in observation_topics if 'image' in t and 'compressed' in t]
        self.joint_state_topics = [t for t in observation_topics if 'joint_states' in t]
        self.action_command_topics = [t for t in self.action_topics if 'trajectory' in t or 'command' in t]
        
        # Identify top view topic (head camera)
        self.top_view_topic = None
        for topic in self.image_topics:
            if 'head' in topic:
                self.top_view_topic = topic
                break
        
        if self.top_view_topic:
            print(f"Top view topic (will be cropped): {self.top_view_topic}")
        
        self.dataset = None
        self.episode_index = 0
        
        # Message type cache
        self._message_type_cache = {}
        
        # Episode mapping: episode_index -> bag_path
        self.episode_mapping = {}
        
    def load_episode_mapping_from_file(self):
        """Load episode mapping from JSON file."""
        if not self.episode_mapping_file:
            return None
        
        mapping_path = Path(self.episode_mapping_file)
        if not mapping_path.exists():
            print(f"⚠️ Warning: Episode mapping file not found: {self.episode_mapping_file}")
            return None
        
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        
        print(f"📄 Loaded episode mapping from: {self.episode_mapping_file}")
        return data
        
    def get_message_type(self, topic_type: str):
        """Get message type class from string, with caching."""
        if topic_type not in self._message_type_cache:
            self._message_type_cache[topic_type] = get_message(topic_type)
        return self._message_type_cache[topic_type]
        
    def get_synchronized_messages(self, bag_path: str) -> List[Dict]:
        """
        Get pre-synchronized messages grouped by message index.
        Aligns all topics to have the same message count by truncating from the end.
        """
        storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic metadata
        topic_metadata = reader.get_all_topics_and_types()
        
        # Filter for relevant topics
        all_topics = self.observation_topics + self.action_topics
        relevant_topics = {}
        for topic_info in topic_metadata:
            if topic_info.name in all_topics:
                relevant_topics[topic_info.name] = topic_info.type
        
        if not relevant_topics:
            print(f"❌ No relevant topics found")
            return []
        
        print(f"📊 Using topics: {list(relevant_topics.keys())}")
        
        # Set topic filter
        from rosbag2_py import StorageFilter
        storage_filter = StorageFilter(topics=list(relevant_topics.keys()))
        reader.set_filter(storage_filter)
        
        # Read all messages organized by topic
        topic_messages = {topic: [] for topic in relevant_topics.keys()}
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic in topic_messages:
                msg_type = self.get_message_type(relevant_topics[topic])
                msg = deserialize_message(data, msg_type)
                topic_messages[topic].append((msg, timestamp))
        
        # Check message counts
        message_counts = {topic: len(msgs) for topic, msgs in topic_messages.items()}
        print(f"📊 Message counts: {message_counts}")
        
        # Find minimum message count
        min_count = min(message_counts.values())
        max_count = max(message_counts.values())
        
        if min_count != max_count:
            count_diff = max_count - min_count
            if count_diff > 2:
                print(f"❌ Topic message count difference too large ({count_diff} > 2), skipping bag")
                return []
            print(f"⚠️ Topics have different message counts (min={min_count}, max={max_count})")
            print(f"✂️ Truncating all topics to {min_count} messages (cutting from end)")
            
            # Truncate all topics to minimum count (from the end)
            for topic_name in topic_messages.keys():
                if len(topic_messages[topic_name]) > min_count:
                    topic_messages[topic_name] = topic_messages[topic_name][:min_count]
        
        num_messages = min_count
        print(f"✅ Aligned all topics to {num_messages} messages")
        
        # Group messages by index
        synchronized_frames = []
        for i in range(num_messages):
            frame_data = {}
            for topic_name, messages in topic_messages.items():
                msg, timestamp = messages[i]
                frame_data[topic_name] = (msg, timestamp)
            synchronized_frames.append(frame_data)
        
        return synchronized_frames
    
    def decode_compressed_image_from_msg(self, msg, topic_name: str, downsize: bool = True) -> np.ndarray:
        """
        Decode compressed image from ROS CompressedImage message.
        Applies cropping and rotation for top view, resizing for left arm.
        Both cameras output 224×224 resolution.
        """
        image_data = bytes(msg.data)
        
        if not image_data:
            raise ValueError("Empty image data in message")
        
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            raise ValueError("Failed to decode image data")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply cropping FIRST (only for top view)
        if self.crop_box and topic_name == self.top_view_topic:
            x, y, w, h = self.crop_box
            image = image[y:y+h, x:x+w]  # Crop to 193×237 (height=237, width=193)
        
        # Then apply camera-specific transformations
        if topic_name == self.top_view_topic:
            # Top view (head camera): Rotate 90° clockwise → 237×193 (height=193, width=237)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            # Left arm camera: Resize to 224×224 (height=224, width=224)
            # cv2.resize takes (width, height), so (224, 224) for width=224, height=224
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_left_hand_state_from_msg(self, msg) -> Dict[str, np.ndarray]:
        """Extract left hand joint positions and velocities from ROS JointState message."""
        if not hasattr(msg, 'name') or not msg.name:
            raise ValueError("Joint state message has no joint names")
        if not hasattr(msg, 'position') or not msg.position:
            raise ValueError("Joint state message has no position data")
        if not hasattr(msg, 'velocity') or not msg.velocity:
            raise ValueError("Joint state message has no velocity data")
            
        joint_names = list(msg.name)
        positions = np.array(msg.position)
        velocities = np.array(msg.velocity)
        
        # Find indices of left hand joints
        left_hand_indices = []
        for joint_name in self.left_hand_joint_names:
            if joint_name in joint_names:
                left_hand_indices.append(joint_names.index(joint_name))
        
        if not left_hand_indices:
            raise ValueError(f"No left hand joints found in message")
        
        # Extract left hand data
        left_positions = positions[left_hand_indices]
        left_velocities = velocities[left_hand_indices]
        
        # Validate data
        if not np.all(np.isfinite(left_positions)):
            raise ValueError("Left hand positions contain NaN or Inf values")
        if not np.all(np.isfinite(left_velocities)):
            raise ValueError("Left hand velocities contain NaN or Inf values")
            
        return {
            'position': left_positions.astype(positions.dtype),
            'velocity': left_velocities.astype(velocities.dtype)
        }
    
    def extract_controller_command_from_msg(self, msg) -> np.ndarray:
        """Extract action commands from ROS JointTrajectory message based on output mode."""
        if not hasattr(msg, 'points') or not msg.points:
            raise ValueError("JointTrajectory message has no trajectory points")
        
        first_point = msg.points[0]
        
        if not hasattr(first_point, 'positions') or not first_point.positions:
            raise ValueError("Trajectory point has no position commands")
            
        command_positions = np.array(first_point.positions)
        
        if not np.all(np.isfinite(command_positions)):
            raise ValueError("Controller position commands contain NaN or Inf values")
            
        if self.output_mode == "pos_only":
            return command_positions.astype(command_positions.dtype)
        
        elif self.output_mode == "pos_vel":
            if hasattr(first_point, 'velocities') and first_point.velocities:
                command_velocities = np.array(first_point.velocities)
            else:
                command_velocities = np.zeros_like(command_positions)
            
            if not np.all(np.isfinite(command_velocities)):
                raise ValueError("Controller velocity commands contain NaN or Inf values")
                
            combined_commands = np.concatenate([command_positions, command_velocities])
            return combined_commands.astype(command_positions.dtype)
        
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
    
    def create_regular_timestamps(self, num_frames: int, start_time: float = None) -> np.ndarray:
        """Generate perfectly regular relative timestamps based on FPS."""
        dt = 1.0 / self.fps
        return np.array([i * dt for i in range(num_frames)])
    
    def apply_episode_trimming(self, synchronized_frames: List[Dict], episode_index: int) -> List[Dict]:
        """
        Apply episode-specific frame trimming.
        Trims frames from the beginning and end based on episode_trim_frames mapping.
        episode_trim_frames[episode_index] = (start_frame, end_frame) where frames are 1-indexed.
        """
        if episode_index not in self.episode_trim_frames:
            print(f"ℹ️ No trimming configured for episode {episode_index}")
            return synchronized_frames
        
        # Get start and end frames (1-indexed in specification)
        trim_info = self.episode_trim_frames[episode_index]
        if isinstance(trim_info, tuple):
            start_frame_1indexed, end_frame_1indexed = trim_info
        else:
            # Backward compatibility: if just an int, treat as start_frame only
            start_frame_1indexed = trim_info
            end_frame_1indexed = len(synchronized_frames)
        
        # Convert to 0-indexed
        start_frame_0indexed = start_frame_1indexed - 1 if start_frame_1indexed > 0 else 0
        end_frame_0indexed = end_frame_1indexed  # end_frame is inclusive in 1-indexed, exclusive in 0-indexed slicing
        
        # Validate bounds
        if start_frame_0indexed >= len(synchronized_frames):
            print(f"⚠️ Warning: Start frame {start_frame_1indexed} exceeds total frames {len(synchronized_frames)}")
            return synchronized_frames
        
        if end_frame_0indexed > len(synchronized_frames):
            print(f"⚠️ Warning: End frame {end_frame_1indexed} exceeds total frames {len(synchronized_frames)}, using all available frames")
            end_frame_0indexed = len(synchronized_frames)
        
        if start_frame_0indexed >= end_frame_0indexed:
            print(f"⚠️ Warning: Start frame {start_frame_1indexed} >= end frame {end_frame_1indexed}")
            return synchronized_frames
        
        # Trim frames
        trimmed_frames = synchronized_frames[start_frame_0indexed:end_frame_0indexed]
        frames_dropped_start = start_frame_0indexed
        frames_dropped_end = len(synchronized_frames) - end_frame_0indexed
        
        print(f"✂️ Episode {episode_index}: Trimming frames {start_frame_1indexed}-{end_frame_1indexed} "
              f"(dropped {frames_dropped_start} from start, {frames_dropped_end} from end), "
              f"keeping {len(trimmed_frames)}/{len(synchronized_frames)}")
        
        return trimmed_frames
    
    def create_dataset_features(self, sample_images: Dict[str, np.ndarray] = None) -> Dict:
        """Create feature definitions that match LeRobot's expected format."""
        features = {}
        
        # Add image features with actual dimensions
        for topic in self.image_topics:
            feature_key = self.topic_to_feature_key(topic, 'observation.images')
            
            if sample_images and topic in sample_images:
                height, width, channels = sample_images[topic].shape
                features[feature_key] = {
                    "dtype": "video",
                    "shape": (channels, height, width),  # CHW format
                    "names": ["channel", "height", "width"]
                }
                print(f"📸 {feature_key}: {channels}x{height}x{width}")
            else:
                # Default dimensions
                features[feature_key] = {
                    "dtype": "video",
                    "shape": (3, 240, 424),
                    "names": ["channel", "height", "width"]
                }
        
        # Add joint state observation based on input mode
        if self.joint_state_topics and self.input_mode != "vision_only":
            num_left_joints = len(self.left_hand_joint_names)
            
            if self.input_mode == "vision_pos":
                features["observation.state"] = {
                    "dtype": "float64",
                    "shape": (num_left_joints,),
                    "names": [f"{joint}_pos" for joint in self.left_hand_joint_names]
                }
            elif self.input_mode == "vision_pos_vel":
                features["observation.state"] = {
                    "dtype": "float64",
                    "shape": (num_left_joints * 2,),
                    "names": ([f"{joint}_pos" for joint in self.left_hand_joint_names] + 
                             [f"{joint}_vel" for joint in self.left_hand_joint_names])
                }
        
        # Add action based on output mode
        num_left_joints = len(self.left_hand_joint_names)
        
        if self.output_mode == "pos_only":
            features["action"] = {
                "dtype": "float64",
                "shape": (num_left_joints,),
                "names": [f"{joint}_pos_cmd" for joint in self.left_hand_joint_names]
            }
        elif self.output_mode == "pos_vel":
            features["action"] = {
                "dtype": "float64",
                "shape": (num_left_joints * 2,),
                "names": [f"{joint}_pos_cmd" for joint in self.left_hand_joint_names] + 
                        [f"{joint}_vel_cmd" for joint in self.left_hand_joint_names]
            }
        
        # Add required LeRobot features
        features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
        features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
        
        return features
    
    def topic_to_feature_key(self, topic: str, prefix: str) -> str:
        """Convert ROS topic name to clean LeRobot feature key."""
        clean_name = topic.replace('/', '_').replace('__', '_').strip('_')
        clean_name = clean_name.replace('_sync', '')
        clean_name = clean_name.replace('emily01_', '')
        clean_name = clean_name.replace('_color_image_raw_compressed', '_cam')
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
                "conversion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "skipped_episodes": sorted(list(self.skip_episodes))
            },
            "episode_mapping": self.episode_mapping
        }
        
        try:
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            print(f"📄 Episode mapping saved to: {mapping_file}")
        except Exception as e:
            print(f"❌ Failed to save episode mapping: {e}")
    
    def process_bag_file(self, bag_path: str, original_episode_index: int) -> bool:
        """
        Process a single bag file.
        
        Args:
            bag_path: Path to the ROS bag directory
            original_episode_index: Original episode index from mapping file
        """
        print(f"\n{'='*60}")
        print(f"📁 Processing bag: {bag_path}")
        print(f"📊 Original episode index: {original_episode_index}")
        print(f"{'='*60}")
        
        try:
            # Get synchronized messages
            synchronized_frames = self.get_synchronized_messages(bag_path)
            if not synchronized_frames:
                print(f"❌ No synchronized frames found")
                return False
            
            print(f"📊 Extracted {len(synchronized_frames)} synchronized frames")
            
            # Apply episode-specific trimming
            trimmed_frames = self.apply_episode_trimming(synchronized_frames, original_episode_index)
            
            if len(trimmed_frames) < 1:
                print(f"❌ No frames remaining after trimming")
                return False
            
            # Create episode buffer
            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.episode_index)
            
            # Generate regular timestamps
            regular_timestamps = self.create_regular_timestamps(len(trimmed_frames))
            
            # Process each frame
            for i, frame_data in enumerate(trimmed_frames):
                frame_dict = {}
                
                # Process each topic in the frame
                for topic_name, (msg, original_timestamp) in frame_data.items():
                    if topic_name in self.image_topics:
                        # Decode image (with cropping for top view)
                        image = self.decode_compressed_image_from_msg(msg, topic_name, downsize=self.downsize_images)
                        # Convert to CHW format
                        image = np.transpose(image, (2, 0, 1))
                        feature_key = self.topic_to_feature_key(topic_name, 'observation.images')
                        frame_dict[feature_key] = image
                            
                    elif topic_name in self.joint_state_topics and self.input_mode != "vision_only":
                        # Extract left hand joint data
                        left_hand_state = self.extract_left_hand_state_from_msg(msg)
                        
                        if self.input_mode == "vision_pos":
                            frame_dict["observation.state"] = left_hand_state['position']
                        elif self.input_mode == "vision_pos_vel":
                            combined_state = np.concatenate([
                                left_hand_state['position'], 
                                left_hand_state['velocity']
                            ])
                            frame_dict["observation.state"] = combined_state
                    
                    elif topic_name in self.action_command_topics:
                        # Extract controller commands
                        controller_commands = self.extract_controller_command_from_msg(msg)
                        frame_dict["action"] = controller_commands
                
                # Validate frame
                if not frame_dict:
                    raise ValueError(f"Frame {i} has no valid data")
                
                # Ensure required data
                if self.input_mode != "vision_only" and "observation.state" not in frame_dict:
                    raise ValueError(f"Frame {i} missing joint state data")
                
                # Add action if not set
                if "action" not in frame_dict:
                    if self.input_mode == "vision_only":
                        raise ValueError(f"Frame {i} missing action data")
                    
                    if self.output_mode == "pos_only":
                        if self.input_mode == "vision_pos":
                            frame_dict["action"] = frame_dict["observation.state"].copy()
                        elif self.input_mode == "vision_pos_vel":
                            frame_dict["action"] = frame_dict["observation.state"][:6].copy()
                    elif self.output_mode == "pos_vel":
                        if self.input_mode == "vision_pos":
                            positions = frame_dict["observation.state"]
                            velocities = np.zeros_like(positions)
                            frame_dict["action"] = np.concatenate([positions, velocities])
                        elif self.input_mode == "vision_pos_vel":
                            frame_dict["action"] = frame_dict["observation.state"].copy()
                
                # Add required LeRobot fields
                frame_dict["next.reward"] = np.array([0.0], dtype=np.float32)
                frame_dict["next.done"] = np.array([False], dtype=bool)
                
                # Add frame
                self.dataset.add_frame(frame_dict, self.task_description, regular_timestamps[i])
            
            # Mark last frame as episode end
            self.dataset.episode_buffer["next.done"][-1] = np.array([True], dtype=bool)
            
            # Save episode
            self.dataset.save_episode()
            
            # Record mapping
            bag_name = Path(bag_path).name
            self.episode_mapping[self.episode_index] = {
                "bag_path": bag_path,
                "bag_name": bag_name,
                "original_episode_index": original_episode_index,
                "original_frames": len(synchronized_frames),
                "final_frames": len(trimmed_frames)
            }
            
            print(f"✅ Saved episode {self.episode_index} with {len(trimmed_frames)} frames")
            self.episode_index += 1
            return True
            
        except Exception as e:
            print(f"❌ Error processing {bag_path}: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self.dataset, 'episode_buffer') and self.dataset.episode_buffer:
                self.dataset.episode_buffer = None
            return False
    
    def extract_frames_from_bag(self, bag_path: str, original_episode_index: int) -> Tuple[List[Dict], List[float], bool]:
        """
        Extract and prepare frames from a bag without writing to dataset.
        This is used for parallel processing.
        
        Returns:
            Tuple of (frame_data_list, timestamps, success)
        """
        try:
            # Get synchronized messages
            synchronized_frames = self.get_synchronized_messages(bag_path)
            if not synchronized_frames:
                print(f"❌ No synchronized frames found in {bag_path}")
                return None, None, False
            
            print(f"📊 Extracted {len(synchronized_frames)} synchronized frames from {Path(bag_path).name}")
            
            # Apply episode-specific trimming
            trimmed_frames = self.apply_episode_trimming(synchronized_frames, original_episode_index)
            
            if len(trimmed_frames) < 1:
                print(f"❌ No frames remaining after trimming")
                return None, None, False
            
            # Generate regular timestamps
            regular_timestamps = self.create_regular_timestamps(len(trimmed_frames))
            
            # Process each frame
            processed_frames = []
            for i, frame_data in enumerate(trimmed_frames):
                frame_dict = {}
                
                # Process each topic in the frame
                for topic_name, (msg, original_timestamp) in frame_data.items():
                    if topic_name in self.image_topics:
                        # Decode image (with cropping for top view)
                        image = self.decode_compressed_image_from_msg(msg, topic_name, downsize=self.downsize_images)
                        # Convert to CHW format
                        image = np.transpose(image, (2, 0, 1))
                        feature_key = self.topic_to_feature_key(topic_name, 'observation.images')
                        frame_dict[feature_key] = image
                            
                    elif topic_name in self.joint_state_topics and self.input_mode != "vision_only":
                        # Extract left hand joint data
                        left_hand_state = self.extract_left_hand_state_from_msg(msg)
                        
                        if self.input_mode == "vision_pos":
                            frame_dict["observation.state"] = left_hand_state['position']
                        elif self.input_mode == "vision_pos_vel":
                            combined_state = np.concatenate([
                                left_hand_state['position'],
                                left_hand_state['velocity']
                            ])
                            frame_dict["observation.state"] = combined_state
                    
                    elif topic_name in self.action_command_topics:
                        # Extract action command
                        action = self.extract_controller_command_from_msg(msg)
                        frame_dict["action"] = action
                
                processed_frames.append(frame_dict)
            
            print(f"✅ Extracted {len(processed_frames)} processed frames from {Path(bag_path).name}")
            return processed_frames, regular_timestamps, True
            
        except Exception as e:
            print(f"❌ Error extracting frames from {bag_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, False
    
    def write_frames_to_dataset(self, frames_data: List[Dict], timestamps: List[float]) -> bool:
        """
        Write pre-extracted frames to the dataset.
        This is thread-safe when called sequentially.
        
        Args:
            frames_data: List of processed frame dictionaries
            timestamps: List of timestamps for frames
            
        Returns:
            Success boolean
        """
        try:
            # Create episode buffer
            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.episode_index)
            
            # Write each frame
            for i, frame_dict in enumerate(frames_data):
                timestamp = timestamps[i]
                
                # Add frame to dataset
                self.dataset.add_frame(frame_dict)
            
            # Save episode
            self.dataset.save_episode()
            
            print(f"✅ Saved episode {self.episode_index} with {len(frames_data)} frames")
            self.episode_index += 1
            return True
            
        except Exception as e:
            print(f"❌ Error writing frames to dataset: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self.dataset, 'episode_buffer') and self.dataset.episode_buffer:
                self.dataset.episode_buffer = None
            return False
    
    def convert_bags(self, bag_directories: List[str], num_workers: int = 4) -> None:
        """
        Convert multiple bag files to LeRobot dataset with parallel extraction.
        
        Args:
            bag_directories: List of paths to bag directories
            num_workers: Number of parallel workers for bag extraction (default: 4)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting ROS Bag to LeRobot Conversion (Parallel Mode)")
        print(f"{'='*60}")
        print(f"Total bag files: {len(bag_directories)}")
        print(f"Parallel workers: {num_workers}")
        
        if not bag_directories:
            raise ValueError("No bag directories provided")
        
        # Load episode mapping if provided
        if self.episode_mapping_file:
            self.loaded_episode_mapping = self.load_episode_mapping_from_file()
            
            # Build a mapping from bag_path to episode_index for quick lookup
            if self.loaded_episode_mapping:
                self.bag_path_to_episode_index = {}
                for ep_key, ep_info in self.loaded_episode_mapping.items():
                    if isinstance(ep_info, dict) and 'bag_path' in ep_info and 'episode_index' in ep_info:
                        self.bag_path_to_episode_index[ep_info['bag_path']] = ep_info['episode_index']
                
                print(f"📄 Loaded mapping for {len(self.bag_path_to_episode_index)} episodes")
                if self.skip_episodes:
                    print(f"🚫 Will skip episodes with indices: {sorted(self.skip_episodes)}")
        
        # Initialize ROS2
        rclpy.init()
        
        try:
            # Analyze first bag for features
            sample_images = {}
            first_bag = bag_directories[0]
            print(f"\n📊 Analyzing first bag for features: {first_bag}")
            
            sample_frames = self.get_synchronized_messages(first_bag)
            
            # Get sample images
            for frame_data in sample_frames[:5]:
                for topic_name, (msg, timestamp) in frame_data.items():
                    if topic_name in self.image_topics and topic_name not in sample_images:
                        try:
                            image = self.decode_compressed_image_from_msg(msg, topic_name, downsize=self.downsize_images)
                            sample_images[topic_name] = image
                            print(f"📸 Sample image from {topic_name}: {image.shape}")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not decode sample image: {e}")
                            continue
                
                if len(sample_images) >= len(self.image_topics):
                    break
            
            # Create dataset features
            features = self.create_dataset_features(sample_images)
            print(f"\n🏗️ Dataset features: {list(features.keys())}")
            
            # Create LeRobot dataset
            if self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
            
            self.dataset = LeRobotDataset.create(
                repo_id=self.dataset_name,
                fps=self.fps,
                features=features,
                root=self.output_dir,
                use_videos=True,
                image_writer_threads=4,
                tolerance_s=self.tolerance_s
            )
            
            # Prepare bags for processing (filter skipped episodes)
            bags_to_process = []
            for i, bag_path in enumerate(bag_directories):
                # Determine original episode index from mapping file
                if hasattr(self, 'bag_path_to_episode_index') and bag_path in self.bag_path_to_episode_index:
                    original_episode_index = self.bag_path_to_episode_index[bag_path]
                else:
                    # Fallback: use sequential index
                    original_episode_index = i
                    if self.loaded_episode_mapping:
                        print(f"⚠️ Warning: Could not find episode index for {bag_path}, using {i}")
                
                # Skip if in skip list
                if original_episode_index in self.skip_episodes:
                    print(f"\n⏭️ Skipping episode {original_episode_index} (bag: {Path(bag_path).name})")
                    continue
                
                bags_to_process.append((i, bag_path, original_episode_index))
            
            print(f"\n📊 Will process {len(bags_to_process)} bags with {num_workers} workers")
            
            # PHASE 1: Parallel extraction of frames from bags
            print(f"\n{'='*60}")
            print(f"📊 PHASE 1: Extracting frames from bags (parallel)")
            print(f"{'='*60}")
            
            extracted_frames = {}
            
            # Create extraction tasks
            extraction_tasks = [
                (bag_path, original_episode_index) 
                for _, bag_path, original_episode_index in bags_to_process
            ]
            
            # Run parallel extraction using thread pool
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for bag_path, original_episode_index in extraction_tasks:
                    future = executor.submit(self._extract_bag_worker, bag_path, original_episode_index)
                    futures[future] = (bag_path, original_episode_index)
                
                # Collect results as they complete
                successful_extractions = 0
                for future in as_completed(futures):
                    bag_path, original_episode_index = futures[future]
                    try:
                        frames_data, timestamps, success = future.result()
                        if success:
                            extracted_frames[bag_path] = (frames_data, timestamps, original_episode_index)
                            successful_extractions += 1
                            print(f"✅ Extracted: {Path(bag_path).name} ({len(frames_data)} frames)")
                        else:
                            print(f"❌ Failed to extract: {Path(bag_path).name}")
                    except Exception as e:
                        print(f"❌ Error extracting {Path(bag_path).name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\n✅ Extraction phase complete: {successful_extractions}/{len(extraction_tasks)} bags")
            
            # PHASE 2: Sequential writing to dataset (must be done in order)
            print(f"\n{'='*60}")
            print(f"💾 PHASE 2: Writing frames to dataset (sequential)")
            print(f"{'='*60}")
            
            successful_writes = 0
            total_episodes_to_write = len(extracted_frames)
            for i, (bag_path, original_episode_index) in enumerate(extraction_tasks):
                if bag_path not in extracted_frames:
                    print(f"⏭️ Skipping {Path(bag_path).name} (extraction failed)")
                    continue
                
                frames_data, timestamps, _ = extracted_frames[bag_path]
                
                print(f"\n--- 💾 Writing {successful_writes + 1}/{total_episodes_to_write} ({Path(bag_path).name}) ---")
                
                try:
                    if self._write_episode_worker(frames_data, timestamps, original_episode_index, bag_path):
                        successful_writes += 1
                        # Release memory by deleting processed frames
                        del extracted_frames[bag_path]
                except Exception as e:
                    print(f"❌ Error writing {bag_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n{'='*60}")
            print(f"🎉 Conversion Complete!")
            print(f"{'='*60}")
            print(f"Successfully converted: {successful_writes}/{len(extraction_tasks)} bag files")
            print(f"Skipped episodes: {sorted(list(self.skip_episodes))}")
            
            # Save episode mapping
            self.save_episode_mapping()
            
        finally:
            # Cleanup
            try:
                if hasattr(self.dataset, 'image_writer') and self.dataset.image_writer:
                    self.dataset.stop_image_writer()
            except Exception as e:
                print(f"⚠️ Warning during cleanup: {e}")
            
            rclpy.shutdown()
    
    def _extract_bag_worker(self, bag_path: str, original_episode_index: int) -> Tuple:
        """
        Worker function for parallel bag extraction.
        Called from multiprocessing pool.
        """
        return self.extract_frames_from_bag(bag_path, original_episode_index)
    
    def _write_episode_worker(self, frames_data: List[Dict], timestamps: List[float], original_episode_index: int, bag_path: str) -> bool:
        """
        Write episode data to dataset.
        Must be called sequentially (not in parallel).
        """
        try:
            # Create episode buffer
            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.episode_index)
            
            # Write each frame
            for i, frame_dict in enumerate(frames_data):
                # Ensure required fields
                if "next.reward" not in frame_dict:
                    frame_dict["next.reward"] = np.array([0.0], dtype=np.float32)
                if "next.done" not in frame_dict:
                    frame_dict["next.done"] = np.array([False], dtype=bool)
                
                # Add frame
                self.dataset.add_frame(frame_dict, self.task_description, timestamps[i])
            
            # Mark last frame as episode end
            if self.dataset.episode_buffer is not None and len(self.dataset.episode_buffer["next.done"]) > 0:
                self.dataset.episode_buffer["next.done"][-1] = np.array([True], dtype=bool)
            
            # Save episode
            self.dataset.save_episode()
            
            # Record mapping
            bag_name = Path(bag_path).name
            self.episode_mapping[self.episode_index] = {
                "bag_path": bag_path,
                "bag_name": bag_name,
                "original_episode_index": original_episode_index,
                "final_frames": len(frames_data)
            }
            
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
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bags to LeRobot dataset with cropping and trimming",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bag_dir", help="Directory containing bag files")
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
                           "/sync/emily01/left_arm/color/image_raw/compressed",
                           "/sync/emily01/head/color/image_raw/compressed", 
                           "/sync/joint_states"
                       ],
                       help="ROS topics for observations")
    
    parser.add_argument("--action-topics", nargs="+",
                       default=["/sync/la_trajectory_controller/joint_trajectory"],
                       help="ROS topics for action commands")
    
    parser.add_argument("--left-hand-joints", nargs="+", default=None,
                       help="Names of left arm joints")
    
    # Input/output modes
    parser.add_argument("--input-mode",
                       choices=["vision_only", "vision_pos", "vision_pos_vel"],
                       default="vision_pos_vel",
                       help="Input mode")
    
    parser.add_argument("--output-mode", 
                       choices=["pos_only", "pos_vel"],
                       default="pos_vel",
                       help="Output mode")
    
    # Cropping and trimming
    parser.add_argument("--crop", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                       help="Crop bounding box for top view: x y width height")
    
    parser.add_argument("--episode-mapping", help="Path to episode_mapping.json file")
    
    parser.add_argument("--skip-episodes", nargs="+", type=int, default=[],
                       help="Episode indices to skip (e.g., 1 2)")
    
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers for bag extraction (default: 1 for sequential). "
                            "Parallel processing (2-8 workers) speeds up conversion but requires more RAM. "
                            "Each worker loads a full episode into memory.")
    
    args = parser.parse_args()
    
    # Find bag directories
    bag_base_dir = Path(args.bag_dir)
    if not bag_base_dir.exists():
        print(f"❌ Error: Directory {bag_base_dir} does not exist")
        sys.exit(1)
    
    bag_directories = []
    for item in bag_base_dir.iterdir():
        if item.is_dir():
            db3_files = list(item.glob("*.db3"))
            bag_files = list(item.glob("*.bag")) + list(item.glob("*.mcap"))
            if db3_files or bag_files:
                bag_directories.append(str(item))
    
    bag_directories.sort()
    
    # Limit episodes if requested
    if args.max_episodes:
        bag_directories = bag_directories[:args.max_episodes]
    
    print(f"Found {len(bag_directories)} bag directories")
    
    if not bag_directories:
        print("No bag directories found")
        sys.exit(1)
    
    # Parse crop box
    crop_box = None
    if args.crop:
        crop_box = tuple(args.crop)
    
    # Load episode trim frames from mapping file if provided
    episode_trim_frames = {}
    if args.episode_mapping:
        try:
            with open(args.episode_mapping, 'r') as f:
                mapping_data = json.load(f)
            
            # Extract episode indices and their start/end frames
            for ep_key, ep_info in mapping_data.items():
                if ep_key.startswith('episode_'):
                    ep_idx = int(ep_key.split('_')[1])
                    # Extract start_frame and end_frame from mapping file
                    start_frame = ep_info.get('start_frame')
                    end_frame = ep_info.get('end_frame')
                    
                    if start_frame is not None and end_frame is not None:
                        episode_trim_frames[ep_idx] = (start_frame, end_frame)
                    elif start_frame is not None:
                        # If only start_frame is provided, use it (no end trimming)
                        episode_trim_frames[ep_idx] = start_frame
            
            print(f"✅ Loaded frame trimming for {len(episode_trim_frames)} episodes from mapping file")
        except Exception as e:
            print(f"⚠️ Error: Could not load episode mapping: {e}")
            print("Please provide a valid episode mapping file with 'start_frame' and 'end_frame' information.")
            sys.exit(1)
    
    # Create converter
    converter = ROSBag2ConverterCroppedTrimmed(
        observation_topics=args.observation_topics,
        action_topics=args.action_topics,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        fps=args.fps,
        task_description=args.task,
        downsize_images=not args.no_downsize,
        tolerance_s=args.tolerance,
        trim_unmoving_end=not args.no_trim_unmoving,
        left_hand_joint_names=args.left_hand_joints,
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        crop_box=crop_box,
        episode_trim_frames=episode_trim_frames,
        episode_mapping_file=args.episode_mapping,
        skip_episodes=args.skip_episodes
    )
    
    start_time = time.time()
    converter.convert_bags(bag_directories, num_workers=args.num_workers)
    end_time = time.time()
    
    print(f"\n⏱️ Total conversion time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
