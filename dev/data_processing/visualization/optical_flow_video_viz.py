#!/usr/bin/env python3
"""
Optical Flow Video Visualization

This script creates videos with optical flow analysis overlaid on original frames,
showing motion magnitude, flow vectors, trimming decisions, and real-time analysis.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ROS2 imports
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions


class OpticalFlowVideoVisualizer:
    """Create videos with optical flow analysis overlaid on original frames."""
    
    def __init__(self, output_dir: str, fps: int = 30, downsize: bool = True):
        """
        Initialize the video visualizer.
        
        Args:
            output_dir: Directory to save output videos
            fps: Video frame rate
            downsize: Whether to use half resolution for faster processing
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.downsize = downsize
        self.output_dir.mkdir(exist_ok=True)
        
        # Message type cache
        self._message_type_cache = {}
        
        # Motion analysis parameters (matching optical flow trimmer)
        self.motion_threshold = 2.0
        self.start_analysis_frames = 100
        self.end_analysis_frames = 60
        self.min_motion_frames = 8
        
        # Joint analysis settings
        self.joint_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.aggregate_color = 'red'
        
        print(f"📹 Video visualizer initialized")
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
    
    def load_left_arm_joint_data(self, bag_path: str) -> Dict:
        """Load left arm joint data from ROS bag file."""
        print(f"🦾 Loading left arm joint data...")
        
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
        joint_topic = "/sync/joint_states"
        
        for topic_info in topic_metadata:
            if topic_info.name == joint_topic:
                topic_type = topic_info.type
                break
                
        if not topic_type:
            print(f"❌ Joint states topic {joint_topic} not found")
            return {}
        
        # Set topic filter
        from rosbag2_py import StorageFilter
        storage_filter = StorageFilter(topics=[joint_topic])
        reader.set_filter(storage_filter)
        
        # Extract joint data
        timestamps = []
        all_positions = []
        all_velocities = []
        joint_names = []
        
        first_message = True
        
        while reader.has_next():
            (topic_name, data, timestamp) = reader.read_next()
            if topic_name == joint_topic:
                try:
                    msg_type = self.get_message_type(topic_type)
                    msg = deserialize_message(data, msg_type)
                    
                    if hasattr(msg, 'position') and len(msg.position) > 0:
                        timestamps.append(timestamp)
                        all_positions.append(list(msg.position))
                        
                        # Handle optional fields
                        if hasattr(msg, 'velocity') and len(msg.velocity) > 0:
                            all_velocities.append(list(msg.velocity))
                        else:
                            all_velocities.append([0.0] * len(msg.position))
                        
                        # Get joint names from first message
                        if first_message and hasattr(msg, 'name') and len(msg.name) > 0:
                            joint_names = list(msg.name)
                            first_message = False
                            
                except Exception as e:
                    print(f"⚠️ Warning: Failed to decode joint state message: {e}")
                    continue
        
        if not timestamps:
            print("❌ No joint state data found")
            return {}
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)
        
        # Filter for left arm joints only
        left_arm_indices = [i for i, name in enumerate(joint_names) if name.startswith('la_')]
        
        if not left_arm_indices:
            print("❌ No left arm joints found")
            return {}
        
        # Extract left arm data
        left_arm_names = [joint_names[i] for i in left_arm_indices]
        left_arm_positions = all_positions[:, left_arm_indices]
        left_arm_velocities = all_velocities[:, left_arm_indices]
        
        # Convert timestamps to relative time in seconds
        time_seconds = (timestamps - timestamps[0]) / 1e9
        
        # Calculate individual joint velocity magnitudes (absolute values)
        individual_velocity_magnitudes = np.abs(left_arm_velocities)
        
        # Calculate aggregated velocity magnitude (L2 norm across all left arm joints)
        aggregated_velocity_magnitude = np.linalg.norm(left_arm_velocities, axis=1)
        
        print(f"   Left arm joints: {len(left_arm_names)}")
        print(f"   Joint samples: {len(timestamps)}")
        
        return {
            'timestamps': timestamps,
            'time_seconds': time_seconds,
            'joint_names': left_arm_names,
            'positions': left_arm_positions,
            'velocities': left_arm_velocities,
            'individual_velocity_magnitudes': individual_velocity_magnitudes,
            'aggregated_velocity_magnitude': aggregated_velocity_magnitude,
            'num_joints': len(left_arm_names),
            'num_samples': len(timestamps)
        }
    
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
    
    def calculate_optical_flow_detailed(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate optical flow and return detailed information."""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate motion magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitude = np.mean(magnitude)
            
            return float(motion_magnitude), magnitude, angle, flow
            
        except Exception as e:
            print(f"⚠️ Warning: Optical flow calculation failed: {e}")
            h, w = prev_frame.shape[:2]
            return 0.0, np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w, 2))
    
    def create_flow_overlay(self, frame: np.ndarray, magnitude: np.ndarray, angle: np.ndarray, 
                           flow: np.ndarray, motion_value: float, frame_idx: int, 
                           total_frames: int, trimming_info: Dict, all_motions: List[float] = None,
                           joint_data: Dict = None) -> np.ndarray:
        """Create a comprehensive flow visualization overlay."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Create motion magnitude heatmap overlay
        # Normalize magnitude for visualization
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap (hot colormap for motion intensity)
        colormap = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_HOT)
        
        # Blend with original frame
        alpha = 0.3  # Transparency
        overlay = cv2.addWeighted(overlay, 1-alpha, colormap, alpha, 0)
        
        # 2. Draw flow vectors (sparse sampling for cleaner visualization)
        step = 20  # Sample every 20 pixels
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # Get flow vectors at sample points
        fx, fy = flow[y, x].T
        
        # Filter significant motion
        mag_sample = magnitude[y, x]
        motion_thresh = 1.0
        mask = mag_sample > motion_thresh
        
        if np.any(mask):
            # Draw flow vectors
            scale = 3  # Scale factor for vector visibility
            for i in np.where(mask)[0]:
                pt1 = (int(x[i]), int(y[i]))
                pt2 = (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale))
                
                # Color based on magnitude
                mag_color = min(255, int(mag_sample[i] * 50))
                color = (0, mag_color, 255 - mag_color)  # Blue to red based on intensity
                
                cv2.arrowedLine(overlay, pt1, pt2, color, 2, tipLength=0.3)
        
        # 3. Add simple text overlay directly on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Frame counter
        text = f"Frame: {frame_idx + 1}/{total_frames}"
        cv2.putText(overlay, text, (10, 30), font, font_scale, (255, 255, 255), thickness)
        
        # Motion magnitude
        text = f"Motion: {motion_value:.4f}"
        color = (0, 255, 0) if motion_value < self.motion_threshold else (0, 165, 255)
        cv2.putText(overlay, text, (10, 60), font, font_scale, color, thickness)
        
        # 4. Create motion timeline
        timeline_height = 80
        timeline = self.create_motion_timeline(all_motions or [], frame_idx, w, timeline_height)
        
        # 5. Create velocity plots 
        velocity_plot_height = 200
        velocity_plots = self.create_velocity_plots(joint_data or {}, frame_idx, w, velocity_plot_height)
        
        # 6. Combine overlay with timeline and velocity plots
        final_frame = np.vstack([overlay, timeline, velocity_plots])
        
        return final_frame
    
    def create_motion_timeline(self, all_motions: List[float], current_frame: int, 
                             width: int, height: int = 80) -> np.ndarray:
        """Create a motion magnitude timeline plot with current position indicator."""
        timeline = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not all_motions or len(all_motions) < 2:
            return timeline
        
        # Background
        cv2.rectangle(timeline, (0, 0), (width, height), (30, 30, 30), -1)
        
        # Calculate plot dimensions
        plot_margin = 10
        plot_width = width - 2 * plot_margin
        plot_height = height - 2 * plot_margin
        plot_x = plot_margin
        plot_y = plot_margin
        
        # Normalize motion values for plotting
        max_motion = max(all_motions) if max(all_motions) > 0 else 1.0
        
        # Draw motion magnitude line
        for i in range(1, len(all_motions)):
            x1 = plot_x + int((i - 1) * plot_width / (len(all_motions) - 1))
            y1 = plot_y + plot_height - int(all_motions[i - 1] / max_motion * plot_height)
            x2 = plot_x + int(i * plot_width / (len(all_motions) - 1))
            y2 = plot_y + plot_height - int(all_motions[i] / max_motion * plot_height)
            
            # Color gradient from blue (low) to red (high)
            motion_ratio = all_motions[i] / max_motion
            color = (int(255 * (1 - motion_ratio)), 0, int(255 * motion_ratio))
            
            cv2.line(timeline, (x1, y1), (x2, y2), color, 2)
        
        # Draw current position indicator
        if 0 <= current_frame < len(all_motions):
            current_x = plot_x + int(current_frame * plot_width / (len(all_motions) - 1))
            current_y = plot_y + plot_height - int(all_motions[current_frame] / max_motion * plot_height)
            
            # Draw vertical line
            cv2.line(timeline, (current_x, plot_y), (current_x, plot_y + plot_height), (255, 255, 255), 2)
            
            # Draw current position circle
            cv2.circle(timeline, (current_x, current_y), 4, (0, 255, 255), -1)
            cv2.circle(timeline, (current_x, current_y), 4, (255, 255, 255), 1)
        
        # Draw last 10 frames indicators
        last_10_start = max(0, current_frame - 10)
        for i in range(last_10_start, current_frame):
            if 0 <= i < len(all_motions):
                frame_x = plot_x + int(i * plot_width / (len(all_motions) - 1))
                frame_y = plot_y + plot_height - int(all_motions[i] / max_motion * plot_height)
                
                # Draw small circles for last 10 frames
                alpha = (i - last_10_start) / 10.0  # Fade effect
                circle_color = (int(255 * alpha), int(255 * alpha), 0)  # Yellow with fade
                cv2.circle(timeline, (frame_x, frame_y), 2, circle_color, -1)
        
        # Add threshold line
        threshold_y = plot_y + plot_height - int(self.motion_threshold / max_motion * plot_height)
        cv2.line(timeline, (plot_x, threshold_y), (plot_x + plot_width, threshold_y), (255, 255, 0), 1)
        
        return timeline
    
    def create_velocity_plots(self, joint_data: Dict, frame_idx: int, width: int, height: int) -> np.ndarray:
        """Create compact velocity plots for left arm joints."""
        if not joint_data or 'time_seconds' not in joint_data:
            # Return empty plot if no joint data
            plots = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(plots, "No Joint Data", (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return plots
        
        # Create matplotlib figure for plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('black')
        
        time_seconds = joint_data['time_seconds']
        velocities = joint_data['velocities']
        joint_names = joint_data['joint_names']
        aggregated_velocity_magnitude = joint_data['aggregated_velocity_magnitude']
        
        # Align with frame index instead of time for synchronization with optical flow
        # This ensures the plots align vertically with the optical flow timeline
        frame_indices = np.arange(len(time_seconds))
        max_frames = len(time_seconds) - 1 if len(time_seconds) > 1 else 1
        
        # Plot 1: Individual joint velocities (compact)
        for i, joint_name in enumerate(joint_names):
            if i < len(self.joint_colors):
                color = self.joint_colors[i]
                ax1.plot(frame_indices, velocities[:, i], color=color, linewidth=1.5, alpha=0.8)
        
        # Add aggregated velocity line
        aggregated_velocity = np.mean(velocities, axis=1)
        ax1.plot(frame_indices, aggregated_velocity, color='red', linewidth=2, linestyle='--', alpha=0.9)
        
        # Add current frame indicator (aligned with optical flow timeline)
        if 0 <= frame_idx < len(frame_indices):
            ax1.axvline(x=frame_idx, color='yellow', linewidth=2, alpha=0.8)
        
        # Clean up plot - remove labels, grid, ticks
        ax1.set_xlim(0, max_frames)
        ax1.set_facecolor('black')
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        
        # Plot 2: Velocity magnitudes (compact)
        individual_magnitudes = joint_data['individual_velocity_magnitudes']
        for i, joint_name in enumerate(joint_names):
            if i < len(self.joint_colors):
                color = self.joint_colors[i]
                ax2.plot(frame_indices, individual_magnitudes[:, i], color=color, linewidth=1.5, alpha=0.8)
        
        # Add aggregated magnitude
        ax2.plot(frame_indices, aggregated_velocity_magnitude, color='red', linewidth=2, linestyle='--', alpha=0.9)
        ax2.fill_between(frame_indices, aggregated_velocity_magnitude, alpha=0.2, color='red')
        
        # Add current frame indicator (aligned with optical flow timeline)
        if 0 <= frame_idx < len(frame_indices):
            ax2.axvline(x=frame_idx, color='yellow', linewidth=2, alpha=0.8)
        
        # Clean up plot - remove labels, grid, ticks
        ax2.set_xlim(0, max_frames)
        ax2.set_facecolor('black')
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to OpenCV image
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # Convert RGBA to RGB (remove alpha channel)
        plot_image = plot_image[:, :, :3]
        
        # Convert RGB to BGR for OpenCV
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        # Resize to target dimensions
        if plot_image.shape[:2] != (height, width):
            plot_image = cv2.resize(plot_image, (width, height))
        
        return plot_image
    
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
    
    def analyze_trimming_decisions(self, motions: List[float]) -> Dict:
        """Analyze motion sequence to determine trimming decisions."""
        start_threshold = self.motion_threshold * 1.2
        end_threshold = self.motion_threshold * 0.9
        
        # Start trimming analysis
        start_trim = 0
        consecutive_motions = 0
        min_start_frames = max(5, self.min_motion_frames - 3)
        
        for i in range(1, min(len(motions), self.start_analysis_frames)):
            if motions[i] > start_threshold:
                consecutive_motions += 1
                if consecutive_motions >= min_start_frames:
                    start_trim = max(0, i - min_start_frames + 1)
                    break
            else:
                consecutive_motions = 0
        
        # Limit start trimming
        if start_trim > 25:
            start_trim = min(20, len(motions) // 8)
        
        # End trimming analysis
        end_trim = 0
        if len(motions) > self.end_analysis_frames:
            analysis_start = len(motions) - self.end_analysis_frames
            end_motions = motions[analysis_start:]
            
            consecutive_low_motions = 0
            min_end_frames = max(6, self.min_motion_frames - 2)
            
            for i in range(len(end_motions) - 1, -1, -1):
                if end_motions[i] <= end_threshold:
                    consecutive_low_motions += 1
                    if consecutive_low_motions >= min_end_frames:
                        frames_from_end = len(end_motions) - i + min_end_frames - 1
                        end_trim = frames_from_end
                        break
                else:
                    consecutive_low_motions = 0
            
            # Limit end trimming
            if end_trim > 20:
                end_trim = 15
            elif 0 < end_trim < 8:
                if end_trim < 5:
                    end_trim = 0
        
        return {
            'start_trim': start_trim,
            'end_trim': end_trim,
            'start_threshold': start_threshold,
            'end_threshold': end_threshold
        }
    
    def create_optical_flow_video(self, bag_path: str, episode_name: str = None):
        """Create a video with optical flow analysis overlaid on frames."""
        if episode_name is None:
            episode_name = Path(bag_path).stem
        
        print(f"🎬 Creating optical flow video for: {episode_name}")
        
        # Extract frames
        topic = "/sync/emily01/head/color/image_raw/compressed"
        frames = self.extract_frames_from_bag(bag_path, topic)
        
        if len(frames) < 10:
            print(f"❌ Not enough frames for {episode_name}")
            return
        
        # Load left arm joint data
        joint_data = self.load_left_arm_joint_data(bag_path)
        
        # Calculate motion for all frames first (for trimming analysis)
        print("🔍 Calculating motion analysis...")
        motions = [0.0]  # First frame has no motion
        
        for i in range(1, len(frames)):
            motion, _, _, _ = self.calculate_optical_flow_detailed(frames[i-1], frames[i])
            motions.append(motion)
        
        # Determine trimming decisions
        trimming_info = self.analyze_trimming_decisions(motions)
        print(f"✂️ Trimming analysis: Start={trimming_info['start_trim']}, End={trimming_info['end_trim']}")
        
        # Set up video writer
        output_path = self.output_dir / f"{episode_name}_optical_flow_analysis.mp4"
        height, width = frames[0].shape[:2]
        
        # Add space for timeline and velocity plots
        timeline_height = 80
        velocity_plot_height = 200  # Compact height for velocity plots
        final_height = height + timeline_height + velocity_plot_height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, final_height))
        
        print(f"📹 Creating video with {len(frames)} frames...")
        
        # Process each frame
        for i in range(len(frames)):
            if i == 0:
                # First frame - just add info panel without flow
                overlay_frame = self.create_flow_overlay(
                    frames[i], np.zeros_like(frames[i][:,:,0]), np.zeros_like(frames[i][:,:,0]), 
                    np.zeros((height, width, 2)), motions[i], i, len(frames), trimming_info, motions, joint_data
                )
            else:
                # Calculate optical flow for current frame
                motion, magnitude, angle, flow = self.calculate_optical_flow_detailed(frames[i-1], frames[i])
                
                # Create overlay
                overlay_frame = self.create_flow_overlay(
                    frames[i], magnitude, angle, flow, motion, i, len(frames), trimming_info, motions, joint_data
                )
            
            # Write frame to video
            video_writer.write(overlay_frame)
            
            # Progress indicator
            if (i + 1) % 20 == 0 or i == len(frames) - 1:
                print(f"   Processed {i + 1}/{len(frames)} frames ({(i+1)/len(frames)*100:.1f}%)")
        
        # Clean up
        video_writer.release()
        
        print(f"✅ Optical flow video saved: {output_path}")
        print(f"   Duration: {len(frames)/self.fps:.1f} seconds")
        print(f"   Resolution: {width}x{final_height}")
        
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
                output_path = self.create_optical_flow_video(bag_path, episode_name)
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
    parser = argparse.ArgumentParser(description="Create optical flow visualization videos")
    parser.add_argument("bag_path", help="Path to bag file or directory")
    parser.add_argument("--output-dir", "-o", default="./optical_flow_videos", help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--episode-name", help="Custom episode name")
    parser.add_argument("--no-downsize", action="store_true", help="Use full resolution (slower processing)")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to process")
    parser.add_argument("--batch-mode", action="store_true", help="Process multiple bags with episode mapping")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        visualizer = OpticalFlowVideoVisualizer(args.output_dir, args.fps, downsize=not args.no_downsize)
        
        bag_path = Path(args.bag_path)
        
        if bag_path.is_file():
            # Single bag file
            visualizer.create_optical_flow_video(str(bag_path), args.episode_name)
        elif bag_path.is_dir():
            # Check if it's a bag directory or contains bag directories
            db3_files = list(bag_path.glob("*.db3"))
            if db3_files:
                # It's a bag directory
                visualizer.create_optical_flow_video(str(bag_path), args.episode_name)
            else:
                # Look for subdirectories containing bags
                bag_dirs = []
                for item in bag_path.iterdir():
                    if item.is_dir() and list(item.glob("*.db3")):
                        bag_dirs.append(item)
                
                if bag_dirs:
                    if args.batch_mode:
                        print(f"📚 Batch processing {len(bag_dirs)} bag directories with episode mapping")
                        visualizer.process_multiple_bags(str(bag_path), args.max_episodes)
                    else:
                        print(f"📁 Found {len(bag_dirs)} bag directories - processing individually")
                        # Limit episodes if specified
                        if args.max_episodes:
                            bag_dirs = sorted(bag_dirs)[:args.max_episodes]
                        
                        for bag_dir in sorted(bag_dirs):
                            episode_name = args.episode_name or bag_dir.name
                            visualizer.create_optical_flow_video(str(bag_dir), episode_name)
                else:
                    print(f"❌ No bag files found in {bag_path}")
        else:
            print(f"❌ Path not found: {bag_path}")
            
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == "__main__":
    main()
