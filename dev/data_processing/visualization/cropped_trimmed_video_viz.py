#!/usr/bin/env python3
"""
Cropped and Trimmed Video Visualization

This script creates videos from ROS bag files with:
1. Top view image cropping based on specified bounding box
2. Frame trimming based on episode-specific starting frames
3. Episode mapping from JSON file

Based on optical_flow_video_viz.py with modifications for cropping and trimming.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions


class CroppedTrimmedVideoVisualizer:
    """Create videos with cropping and frame trimming from ROS bag files."""
    
    def __init__(self, output_dir: str, fps: int = 30, crop_box: Tuple[int, int, int, int] = None):
        """
        Initialize the video visualizer.
        
        Args:
            output_dir: Directory to save output videos
            fps: Video frame rate
            crop_box: Tuple of (x, y, w, h) for cropping. If None, no cropping is applied.
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.crop_box = crop_box  # (x, y, w, h)
        self.output_dir.mkdir(exist_ok=True)
        
        # Message type cache
        self._message_type_cache = {}
        
        # Episode-specific trimming indices (1-indexed, as specified)
        # Frame index starts at 1 for each video
        self.episode_trim_frames = {
            0: 20,
            1: 20,
            2: 15,
            3: 10,
            4: 5,
            5: 10,
            6: 10,
            7: 10,
            8: 5,
            9: 10,
            10: 10,
            11: 10,
            12: 10,
            13: 0,
            14: 10,
            15: 10,
            16: 10,
            17: 5,
            18: 5,
            19: 5,
            20: 5,
            21: 5,
            22: 5,
            23: 5,
            24: 5,
            25: 5,
            26: 10,
            27: 5,
        }
        
        print(f"📹 Cropped & Trimmed Video Visualizer initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   FPS: {self.fps}")
        if self.crop_box:
            x, y, w, h = self.crop_box
            print(f"   Crop box (x, y, w, h): ({x}, {y}, {w}, {h})")
        else:
            print(f"   Crop box: None (no cropping)")
        
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
            
            # Apply cropping if specified
            if self.crop_box:
                x, y, w, h = self.crop_box
                image = image[y:y+h, x:x+w]
            
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
    
    def apply_frame_trimming(self, frames: List[np.ndarray], episode_index: int) -> List[np.ndarray]:
        """
        Apply frame trimming based on episode-specific starting frame.
        
        Args:
            frames: List of frames
            episode_index: Episode index (0-based)
            
        Returns:
            Trimmed list of frames
        """
        if episode_index not in self.episode_trim_frames:
            print(f"⚠️ Warning: No trimming info for episode {episode_index}, using all frames")
            return frames
        
        # Get the starting frame (1-indexed in specification, convert to 0-indexed)
        start_frame_1indexed = self.episode_trim_frames[episode_index]
        start_frame_0indexed = start_frame_1indexed - 1 if start_frame_1indexed > 0 else 0
        
        # Ensure we don't trim beyond available frames
        if start_frame_0indexed >= len(frames):
            print(f"⚠️ Warning: Start frame {start_frame_1indexed} exceeds total frames {len(frames)}")
            return frames
        
        trimmed_frames = frames[start_frame_0indexed:]
        print(f"✂️ Trimming: Dropping first {start_frame_1indexed} frames (1-indexed), keeping {len(trimmed_frames)}/{len(frames)} frames")
        
        return trimmed_frames
    
    def create_video_from_frames(self, frames: List[np.ndarray], output_path: Path, episode_name: str):
        """Create a video from a list of frames."""
        if len(frames) == 0:
            print(f"❌ No frames to create video for {episode_name}")
            return None
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        print(f"📹 Creating video with {len(frames)} frames...")
        
        # Write all frames
        for i, frame in enumerate(frames):
            video_writer.write(frame)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(frames) - 1:
                print(f"   Processed {i + 1}/{len(frames)} frames ({(i+1)/len(frames)*100:.1f}%)")
        
        # Clean up
        video_writer.release()
        
        print(f"✅ Video saved: {output_path}")
        print(f"   Duration: {len(frames)/self.fps:.1f} seconds")
        print(f"   Resolution: {width}x{height}")
        
        return output_path
    
    def create_video_from_bag(self, bag_path: str, episode_index: int, episode_name: str = None):
        """
        Create a cropped and trimmed video from a ROS bag file.
        
        Args:
            bag_path: Path to the ROS bag directory
            episode_index: Episode index for trimming lookup
            episode_name: Custom episode name for output file
        """
        if episode_name is None:
            episode_name = Path(bag_path).stem
        
        print(f"\n🎬 Processing Episode {episode_index}: {episode_name}")
        
        # Extract frames from top view topic
        topic = "/sync/emily01/head/color/image_raw/compressed"
        frames = self.extract_frames_from_bag(bag_path, topic)
        
        if len(frames) < 1:
            print(f"❌ Not enough frames for {episode_name}")
            return None
        
        # Apply frame trimming
        trimmed_frames = self.apply_frame_trimming(frames, episode_index)
        
        if len(trimmed_frames) < 1:
            print(f"❌ No frames remaining after trimming for {episode_name}")
            return None
        
        # Create output video
        output_path = self.output_dir / f"episode_{episode_index:03d}_{episode_name}_cropped_trimmed.mp4"
        return self.create_video_from_frames(trimmed_frames, output_path, episode_name)
    
    def load_episode_mapping(self, mapping_file: str) -> Dict:
        """Load episode mapping from JSON file."""
        mapping_path = Path(mapping_file)
        
        if not mapping_path.exists():
            print(f"❌ Episode mapping file not found: {mapping_file}")
            return {}
        
        with open(mapping_path, 'r') as f:
            episode_mapping = json.load(f)
        
        print(f"📄 Loaded episode mapping from: {mapping_file}")
        print(f"   Total episodes: {len(episode_mapping)}")
        
        return episode_mapping
    
    def process_from_episode_mapping(self, mapping_file: str, max_episodes: int = None):
        """
        Process multiple bag files based on episode mapping JSON.
        
        Args:
            mapping_file: Path to episode_mapping.json file
            max_episodes: Maximum number of episodes to process (optional)
        """
        episode_mapping = self.load_episode_mapping(mapping_file)
        
        if not episode_mapping:
            print("❌ No episodes to process")
            return []
        
        # Limit episodes if specified
        episode_ids = sorted(episode_mapping.keys())
        if max_episodes:
            episode_ids = episode_ids[:max_episodes]
        
        print(f"📚 Processing {len(episode_ids)} episodes")
        
        # Process each episode
        processed_episodes = []
        for episode_id in episode_ids:
            episode_info = episode_mapping[episode_id]
            bag_path = episode_info["bag_path"]
            bag_name = episode_info["bag_name"]
            episode_index = episode_info["episode_index"]
            
            try:
                output_path = self.create_video_from_bag(bag_path, episode_index, bag_name)
                processed_episodes.append({
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "bag_path": bag_path,
                    "bag_name": bag_name,
                    "video_path": str(output_path) if output_path else None,
                    "status": "success" if output_path else "failed"
                })
            except Exception as e:
                print(f"❌ Failed to process {episode_id}: {e}")
                import traceback
                traceback.print_exc()
                processed_episodes.append({
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "bag_path": bag_path,
                    "bag_name": bag_name,
                    "video_path": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save processing summary
        summary_path = self.output_dir / "cropped_trimmed_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(processed_episodes, f, indent=2)
        
        print(f"\n✅ Processing complete! Summary saved: {summary_path}")
        
        # Print summary statistics
        successful = sum(1 for ep in processed_episodes if ep["status"] == "success")
        failed = len(processed_episodes) - successful
        print(f"\n📊 Summary:")
        print(f"   Total episodes: {len(processed_episodes)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        return processed_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Create cropped and trimmed videos from ROS bag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all episodes from mapping file with cropping
  python cropped_trimmed_video_viz.py --mapping episode_mapping.json --crop 247 122 193 237
  
  # Process first 5 episodes only
  python cropped_trimmed_video_viz.py --mapping episode_mapping.json --crop 247 122 193 237 --max-episodes 5
  
  # Process without cropping
  python cropped_trimmed_video_viz.py --mapping episode_mapping.json
        """
    )
    parser.add_argument("--mapping", "-m", required=True, help="Path to episode_mapping.json file")
    parser.add_argument("--output-dir", "-o", default="./cropped_trimmed_videos", help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS (default: 30)")
    parser.add_argument("--crop", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                        help="Crop bounding box: x y width height (e.g., 247 122 193 237)")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to process")
    
    args = parser.parse_args()
    
    # Parse crop box
    crop_box = None
    if args.crop:
        crop_box = tuple(args.crop)
        print(f"🔲 Crop box specified: x={crop_box[0]}, y={crop_box[1]}, w={crop_box[2]}, h={crop_box[3]}")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        visualizer = CroppedTrimmedVideoVisualizer(args.output_dir, args.fps, crop_box)
        visualizer.process_from_episode_mapping(args.mapping, args.max_episodes)
            
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == "__main__":
    main()
