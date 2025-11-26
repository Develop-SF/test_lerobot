#!/usr/bin/env python3
"""
Extract frames from ROS bag using optical_flow_video_viz.py method
and create frame sets with equal time differences.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions


class FrameExtractor:
    """Extract frames from ROS bag with timestamp information."""
    
    def __init__(self, output_dir: str = "./extracted_frames"):
        """Initialize the frame extractor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Message type cache
        self._message_type_cache = {}
        
        print(f"📁 Frame extractor initialized")
        print(f"   Output directory: {self.output_dir}")
    
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
            
            return image
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to decode image: {e}")
            return None
    
    def extract_frames_with_timestamps(self, bag_path: str, topic: str) -> List[Tuple[np.ndarray, int, float]]:
        """Extract all frames from a ROS bag file with timestamps."""
        print(f"📂 Extracting frames from: {bag_path}")
        print(f"📡 Topic: {topic}")
        
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
        
        print(f"🔍 Found topic type: {topic_type}")
        
        # Set topic filter
        from rosbag2_py import StorageFilter
        storage_filter = StorageFilter(topics=[topic])
        reader.set_filter(storage_filter)
        
        frames_with_timestamps = []
        first_timestamp = None
        
        while reader.has_next():
            (topic_name, data, timestamp) = reader.read_next()
            if topic_name == topic:
                try:
                    msg_type = self.get_message_type(topic_type)
                    msg = deserialize_message(data, msg_type)
                    image = self.decode_compressed_image_from_msg(msg)
                    
                    if image is not None:
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        
                        # Convert timestamp to relative time in seconds
                        relative_time = (timestamp - first_timestamp) / 1e9
                        
                        frames_with_timestamps.append((image, timestamp, relative_time))
                        
                except Exception as e:
                    print(f"⚠️ Warning: Failed to decode frame: {e}")
                    continue
        
        print(f"📊 Extracted {len(frames_with_timestamps)} frames")
        if frames_with_timestamps:
            total_duration = frames_with_timestamps[-1][2]
            print(f"⏱️ Total duration: {total_duration:.2f} seconds")
        
        return frames_with_timestamps
    
    def select_frames_equal_time_diff(self, frames_with_timestamps: List[Tuple[np.ndarray, int, float]], 
                                    num_frames: int) -> List[Tuple[np.ndarray, int, float, int]]:
        """Select frames with equal time differences."""
        if len(frames_with_timestamps) < num_frames:
            print(f"❌ Not enough frames. Need {num_frames}, got {len(frames_with_timestamps)}")
            return []
        
        total_duration = frames_with_timestamps[-1][2] - frames_with_timestamps[0][2]
        
        if num_frames == 1:
            # Return middle frame
            mid_idx = len(frames_with_timestamps) // 2
            frame, timestamp, rel_time = frames_with_timestamps[mid_idx]
            return [(frame, timestamp, rel_time, mid_idx)]
        
        # Calculate time intervals
        time_interval = total_duration / (num_frames - 1)
        selected_frames = []
        
        print(f"🔍 Selecting {num_frames} frames with {time_interval:.3f}s intervals")
        
        for i in range(num_frames):
            target_time = frames_with_timestamps[0][2] + i * time_interval
            
            # Find closest frame to target time
            best_idx = 0
            best_diff = abs(frames_with_timestamps[0][2] - target_time)
            
            for idx, (_, _, rel_time) in enumerate(frames_with_timestamps):
                time_diff = abs(rel_time - target_time)
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_idx = idx
            
            frame, timestamp, rel_time = frames_with_timestamps[best_idx]
            selected_frames.append((frame, timestamp, rel_time, best_idx))
            
            print(f"   Frame {i+1}: t={rel_time:.3f}s (frame {best_idx+1}/{len(frames_with_timestamps)})")
        
        return selected_frames
    
    def save_frame_set(self, frames: List[Tuple[np.ndarray, int, float, int]], 
                      set_name: str, bag_name: str):
        """Save a set of frames to disk."""
        set_dir = self.output_dir / f"{bag_name}_{set_name}_frames"
        set_dir.mkdir(exist_ok=True)
        
        print(f"💾 Saving {set_name} frame set...")
        
        # Save frame info
        frame_info = []
        
        for i, (frame, timestamp, rel_time, frame_idx) in enumerate(frames):
            # Save image
            filename = f"frame_{i+1:02d}_t{rel_time:.3f}s_idx{frame_idx+1:04d}.jpg"
            filepath = set_dir / filename
            
            success = cv2.imwrite(str(filepath), frame)
            if success:
                print(f"   ✅ Saved: {filename}")
                frame_info.append({
                    "frame_number": i + 1,
                    "original_frame_index": frame_idx + 1,
                    "timestamp_ns": int(timestamp),
                    "relative_time_s": float(rel_time),
                    "filename": filename
                })
            else:
                print(f"   ❌ Failed to save: {filename}")
        
        # Save metadata
        metadata_file = set_dir / "frame_info.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump({
                "set_name": set_name,
                "bag_name": bag_name,
                "num_frames": len(frames),
                "frames": frame_info
            }, f, indent=2)
        
        print(f"   📄 Metadata saved: {metadata_file}")
        return set_dir
    
    def extract_and_create_sets(self, bag_path: str, topic: str, 
                               num_frames_sets: List[int] = [4, 5]):
        """Extract frames and create multiple sets with equal time differences."""
        bag_name = Path(bag_path).name
        
        # Extract all frames with timestamps
        frames_with_timestamps = self.extract_frames_with_timestamps(bag_path, topic)
        
        if not frames_with_timestamps:
            print("❌ No frames extracted")
            return
        
        # Create frame sets
        results = {}
        for num_frames in num_frames_sets:
            print(f"\n🎯 Creating {num_frames}-frame set...")
            selected_frames = self.select_frames_equal_time_diff(frames_with_timestamps, num_frames)
            
            if selected_frames:
                set_name = f"{num_frames}_frame_set"
                set_dir = self.save_frame_set(selected_frames, set_name, bag_name)
                results[set_name] = {
                    "directory": str(set_dir),
                    "num_frames": len(selected_frames),
                    "frames": selected_frames
                }
        
        return results


def main():
    """Main function to extract frames."""
    # Configuration
    bag_path = "/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_clean/sync/approach_clean_2025_08_07-14_51_16"
    topic = "/emily01/head/color/image_raw/compressed/sync"
    output_dir = "./extracted_frames"
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create extractor
        extractor = FrameExtractor(output_dir)
        
        # Extract frames and create sets
        print(f"🚀 Starting frame extraction...")
        print(f"   Bag path: {bag_path}")
        print(f"   Topic: {topic}")
        print(f"   Output: {output_dir}")
        
        results = extractor.extract_and_create_sets(bag_path, topic, [4, 5])
        
        if results:
            print(f"\n✅ Frame extraction complete!")
            for set_name, info in results.items():
                print(f"   {set_name}: {info['num_frames']} frames in {info['directory']}")
        else:
            print("❌ Frame extraction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == "__main__":
    main()
