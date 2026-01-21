#!/usr/bin/env python3
import argparse
import os
import glob
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

# ROS2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("Error: ROS 2 environment not sourced or rosbag2_py/rclpy not installed.")
    print("Please run: source /opt/ros/<distro>/setup.bash")
    exit(1)

def open_reader(path):
    reader = rosbag2_py.SequentialReader()
    
    # Try 1: Auto-detection (rely on metadata.yaml)
    try:
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
        return reader
    except Exception:
        pass
        
    # Try 2: Force MCAP (for raw .mcap files or if metadata is missing)
    try:
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)
        return reader
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Visualize JointTrajectory point[0] from MCAP bags.")
    parser.add_argument("--bag_dir", nargs='+', required=True, help="Directories, files, or glob patterns for MCAP bags")
    parser.add_argument("--topic_name", required=True, help="Name of the topic to parse (e.g., /joint_trajectory)")
    parser.add_argument("--output_dir", default="plots", help="Directory to save generated plots")
    args = parser.parse_args()

    # 1. Identify Bag Folders
    all_inputs = []
    for pattern in args.bag_dir:
        # Expand glob and filter out non-existent paths
        matches = glob.glob(pattern)
        if not matches:
            print(f"Warning: Pattern '{pattern}' matched no files.")
        all_inputs.extend(matches)
    
    bag_paths = []
    for path in all_inputs:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # Check if this dir is a bag (contains metadata.yaml or .mcap)
            if glob.glob(os.path.join(path, "metadata.yaml")) or glob.glob(os.path.join(path, "*.mcap")):
                bag_paths.append(path)
            else:
                # Otherwise, look one level deeper for bag subdirectories
                for entry in os.scandir(path):
                    if entry.is_dir():
                        if glob.glob(os.path.join(entry.path, "metadata.yaml")) or glob.glob(os.path.join(entry.path, "*.mcap")):
                            bag_paths.append(entry.path)
        elif os.path.isfile(path) and path.endswith(".mcap"):
            # Direct path to an mcap file
            bag_paths.append(path)
    
    # Remove duplicates while preserving order
    bag_paths = list(dict.fromkeys(bag_paths))
    
    if not bag_paths:
        print(f"No valid ROS 2 bags found for patterns: {args.bag_dir}")
        return

    print(f"Found {len(bag_paths)} bag(s) to process.")

    # Data structure: data[joint_name][bag_name] = [(time, position)]
    data_by_joint = defaultdict(lambda: defaultdict(list))

    # 2. Parse Bags
    pbar = tqdm(bag_paths, desc="Parsing Bags")
    for bag_path in pbar:
        bag_name = os.path.basename(os.path.normpath(bag_path))
        pbar.set_postfix({"current": bag_name[:20]})
        
        reader = open_reader(bag_path)
        if reader is None:
            tqdm.write(f"  Error: Could not open bag at {bag_path}. Plugin 'mcap' might be missing or the file is corrupted.")
            continue

        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        if args.topic_name not in topic_types:
            tqdm.write(f"  Topic '{args.topic_name}' not found in {bag_name}. Available topics: {list(topic_types.keys())}")
            continue
        
        msg_type_name = topic_types[args.topic_name]
        try:
            msg_type = get_message(msg_type_name)
        except Exception as e:
            tqdm.write(f"  Could not load message type {msg_type_name} for {bag_name}: {e}")
            continue

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == args.topic_name:
                msg = deserialize_message(data, msg_type)
                
                # Check for trajectory_msgs/JointTrajectory attributes
                if hasattr(msg, 'joint_names') and len(msg.points) > 0:
                    t_sec = timestamp / 1e9
                    point0 = msg.points[0] # Focus on point[0] as requested
                    
                    for i, joint_name in enumerate(msg.joint_names):
                        if i < len(point0.positions):
                            data_by_joint[joint_name][bag_name].append((t_sec, point0.positions[i]))

    # 3. Plotting
    if not data_by_joint:
        print("\nNo data extracted. Check if the topic contains JointTrajectory messages with points.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    joint_names = sorted(data_by_joint.keys())
    num_joints = len(joint_names)
    
    # Calculate grid size
    cols = 2 if num_joints > 1 else 1
    rows = (num_joints + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    # Track handles for the shared legend
    handles, labels = [], []

    print("\nGenerating combined plot...")
    for i, joint_name in enumerate(tqdm(joint_names, desc="Plotting Joints")):
        ax = axes[i]
        bags = data_by_joint[joint_name]
        
        for bag_name, values in bags.items():
            values.sort() # Ensure chronological order
            times, positions = zip(*values)
            
            # Normalize time to start at 0 for comparison
            start_t = times[0]
            norm_times = [t - start_t for t in times]
            
            line, = ax.plot(norm_times, positions, label=bag_name, alpha=0.7)
            
            # Collect legend info from the first joint's plot
            if i == 0:
                handles.append(line)
                labels.append(bag_name)

        ax.set_title(f"Joint: {joint_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add shared legend at the bottom or top
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), fontsize='small')
    
    plt.suptitle(f"Joint Trajectory Comparison (Topic: {args.topic_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust for suptitle and legend
    
    save_path = os.path.join(args.output_dir, "combined_trajectory_comparison.png")
    plt.savefig(save_path)
    print(f"\nSuccess. Combined plot saved in: {save_path}")

if __name__ == "__main__":
    main()
