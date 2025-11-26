#!/usr/bin/env python3
"""
Left Arm Video Extractor

This script extracts video frames from ROS bag files and creates clean videos
without optical flow analysis, focusing on the left arm camera feed.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions


class LeftArmVideoExtractor:
    """Extract clean video frames from ROS bag files without optical flow analysis."""
    
    def __init__(self, output_dir: str, fps: int = 30, downsize: bool = True):
        """
        Initialize the video extractor.
        
        Args:
            output_dir: Directory to save output videos
            fps: Video frame rate
            downsize: Whether to use 2/3 resolution for smaller file sizes
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.downsize = downsize
        self.output_dir.mkdir(exist_ok=True)
        
        # Message type cache
        self._message_type_cache = {}
        
        print(f"📹 Left arm video extractor initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   FPS: {self.fps}")
        print(f"   2/3 resolution: {self.downsize}")
        
    def get_message_type(self, topic_type: str):
        """Get message type class from string, with caching."""
        if topic_type not in self._message_type_cache:
            self._message_type_cache[topic_type] = get_message(topic_type)
        return self._message_type_cache[topic_type]
    
    def decode_compressed_image_from_msg(self, msg) -> Optional[np.ndarray]:
        """Decode compressed image from ROS CompressedImage message."""
        try:
            image_data = bytes(msg.data)
            if not image_data:
                return None
            
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None or image.size == 0:
                return None
            
            # Downsize to 2/3 resolution if requested
            if self.downsize:
                height, width = image.shape[:2]
                new_height = int(height * 2 / 3)
                new_width = int(width * 2 / 3)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Keep in BGR for OpenCV video writing
            return image
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to decode image: {e}")
            return None
    
    def extract_frames_from_bag(self, bag_path: str, topic: str) -> List[np.ndarray]:
        """Extract all frames from a ROS bag file."""
        print(f"📂 Extracting frames from: {bag_path}")
        
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic metadata
        topic_metadata = reader.get_all_topics_and_types()
        topic_type = None
        for topic_info in topic_metadata:
            if topic_info.name == topic:
                topic_type = topic_info.type
                break
                
        if not topic_type:
            print(f"❌ Topic {topic} not found")
            return []
        
        # Set topic filter
        from rosbag2_py import StorageFilter
        storage_filter = StorageFilter(topics=[topic])
        reader.set_filter(storage_filter)
        
        frames = []
        
        while reader.has_next():
            (topic_name, data, timestamp) = reader.read_next()
            if topic_name == topic:
                try:
                    msg_type = self.get_message_type(topic_type)
                    msg = deserialize_message(data, msg_type)
                    image = self.decode_compressed_image_from_msg(msg)
                    
                    if image is not None:
                        frames.append(image)
                        
                except Exception as e:
                    print(f"⚠️ Warning: Failed to decode frame: {e}")
                    continue
        
        print(f"📊 Extracted {len(frames)} frames")
        return frames
    
    def add_frame_info(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Add simple frame information overlay to the video."""
        overlay = frame.copy()
        
        # Add frame counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Frame counter with background
        text = f"Frame: {frame_idx + 1}/{total_frames}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Add semi-transparent background
        cv2.rectangle(overlay, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
        cv2.rectangle(overlay, (5, 5), (text_size[0] + 15, text_size[1] + 15), (255, 255, 255), 1)
        
        # Add text
        cv2.putText(overlay, text, (10, text_size[1] + 10), font, font_scale, (255, 255, 255), thickness)
        
        return overlay
    
    def create_episode_mapping(self, bag_paths: List[str]) -> Dict:
        """Create episode mapping from bag paths."""
        episode_mapping = {}
        
        for idx, bag_path in enumerate(sorted(bag_paths)):
            bag_name = Path(bag_path).name
            episode_id = f"episode_{idx:03d}"
            
            episode_mapping[episode_id] = {
                "bag_path": str(bag_path),
                "bag_name": bag_name,
                "episode_index": idx
            }
        
        return episode_mapping
    
    def save_episode_mapping(self, episode_mapping: Dict, filename: str = "episode_mapping.json"):
        """Save episode mapping to JSON file."""
        mapping_path = self.output_dir / filename
        
        with open(mapping_path, 'w') as f:
            json.dump(episode_mapping, f, indent=2)
        
        print(f"📄 Episode mapping saved: {mapping_path}")
        return mapping_path
    
    def create_left_arm_video(self, bag_path: str, episode_name: str = None):
        """Create a clean video from left arm camera frames."""
        if episode_name is None:
            episode_name = Path(bag_path).stem
        
        print(f"🎬 Creating left arm video for: {episode_name}")
        
        # Extract frames from left arm camera topic
        topic = "/sync/emily01/left_arm/color/image_raw/compressed"
        frames = self.extract_frames_from_bag(bag_path, topic)
        
        if len(frames) < 1:
            print(f"❌ No frames found for {episode_name}")
            return
        
        # Set up video writer
        output_path = self.output_dir / f"{episode_name}_left_arm.mp4"
        height, width = frames[0].shape[:2]
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        print(f"📹 Creating video with {len(frames)} frames...")
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Add simple frame information
            processed_frame = self.add_frame_info(frame, i, len(frames))
            
            # Write frame to video
            video_writer.write(processed_frame)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(frames) - 1:
                print(f"   Processed {i + 1}/{len(frames)} frames ({(i+1)/len(frames)*100:.1f}%)")
        
        # Clean up
        video_writer.release()
        
        print(f"✅ Left arm video saved: {output_path}")
        print(f"   Duration: {len(frames)/self.fps:.1f} seconds")
        print(f"   Resolution: {width}x{height}")
        print(f"   Total frames: {len(frames)}")
        
        return output_path
    
    def process_multiple_bags(self, bag_directory: str, max_episodes: int = None):
        """Process multiple bag files in a directory with episode mapping."""
        bag_dir = Path(bag_directory)
        
        # Find all bag directories
        bag_paths = []
        for item in bag_dir.iterdir():
            if item.is_dir():
                # Check if it contains ROS bag files
                db_files = list(item.glob("*.db3"))
                if db_files:
                    bag_paths.append(str(item))
        
        if not bag_paths:
            print(f"❌ No bag files found in: {bag_directory}")
            return
        
        # Limit episodes if specified
        if max_episodes:
            bag_paths = sorted(bag_paths)[:max_episodes]
        
        print(f"📚 Found {len(bag_paths)} bag files to process")
        
        # Create episode mapping
        episode_mapping = self.create_episode_mapping(bag_paths)
        self.save_episode_mapping(episode_mapping)
        
        # Process each bag file
        processed_episodes = []
        for episode_id, episode_info in episode_mapping.items():
            bag_path = episode_info["bag_path"]
            episode_name = f"{episode_id}_{episode_info['bag_name']}"
            
            print(f"\n🎬 Processing {episode_id}: {episode_info['bag_name']}")
            
            try:
                output_path = self.create_left_arm_video(bag_path, episode_name)
                processed_episodes.append({
                    "episode_id": episode_id,
                    "bag_path": bag_path,
                    "video_path": str(output_path),
                    "status": "success"
                })
            except Exception as e:
                print(f"❌ Failed to process {episode_id}: {e}")
                processed_episodes.append({
                    "episode_id": episode_id,
                    "bag_path": bag_path,
                    "video_path": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save processing summary
        summary_path = self.output_dir / "video_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(processed_episodes, f, indent=2)
        
        print(f"\n✅ Processing complete! Summary saved: {summary_path}")
        return processed_episodes


def main():
    parser = argparse.ArgumentParser(description="Extract clean left arm videos from ROS bag files")
    parser.add_argument("bag_path", help="Path to bag file or directory")
    parser.add_argument("--output-dir", "-o", default="./left_arm_videos", help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--episode-name", help="Custom episode name")
    parser.add_argument("--no-downsize", action="store_true", help="Use full resolution (larger file sizes)")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to process")
    parser.add_argument("--batch-mode", action="store_true", help="Process multiple bags with episode mapping")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        extractor = LeftArmVideoExtractor(args.output_dir, args.fps, downsize=not args.no_downsize)
        
        bag_path = Path(args.bag_path)
        
        if bag_path.is_file():
            # Single bag file
            extractor.create_left_arm_video(str(bag_path), args.episode_name)
        elif bag_path.is_dir():
            # Check if it's a bag directory or contains bag directories
            db3_files = list(bag_path.glob("*.db3"))
            if db3_files:
                # It's a bag directory
                extractor.create_left_arm_video(str(bag_path), args.episode_name)
            else:
                # Look for subdirectories containing bags
                bag_dirs = []
                for item in bag_path.iterdir():
                    if item.is_dir() and list(item.glob("*.db3")):
                        bag_dirs.append(item)
                
                if bag_dirs:
                    if args.batch_mode:
                        print(f"📚 Batch processing {len(bag_dirs)} bag directories with episode mapping")
                        extractor.process_multiple_bags(str(bag_path), args.max_episodes)
                    else:
                        print(f"📁 Found {len(bag_dirs)} bag directories - processing individually")
                        # Limit episodes if specified
                        if args.max_episodes:
                            bag_dirs = sorted(bag_dirs)[:args.max_episodes]
                        
                        for bag_dir in sorted(bag_dirs):
                            episode_name = args.episode_name or bag_dir.name
                            extractor.create_left_arm_video(str(bag_dir), episode_name)
                else:
                    print(f"❌ No bag files found in {bag_path}")
        else:
            print(f"❌ Path not found: {bag_path}")
            
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == "__main__":
    main()
