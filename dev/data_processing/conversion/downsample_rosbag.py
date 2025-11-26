#!/usr/bin/env python3
"""
Downsample ROS2 bag files from 60Hz to 20Hz
"""

import sys
import os
from pathlib import Path
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from collections import defaultdict
import shutil

def downsample_rosbag(input_bag_dir: str, output_bag_dir: str, downsample_factor: int = 3):
    """
    Downsample a rosbag from 60Hz to 20Hz by keeping every 3rd message.

    Args:
        input_bag_dir: Path to input rosbag directory
        output_bag_dir: Path to output rosbag directory
        downsample_factor: Factor to downsample by (3 for 60Hz -> 20Hz)
    """

    input_path = Path(input_bag_dir)
    output_path = Path(output_bag_dir)

    # Remove output directory if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downsampling rosbag: {input_path} -> {output_path}")
    print(f"Downsample factor: {downsample_factor} (60Hz -> {60//downsample_factor}Hz)")

    # Set up reader
    storage_options = StorageOptions(uri=str(input_path), storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic metadata
    topic_metadata = reader.get_all_topics_and_types()

    # Remove output directory if it exists (before setting up writer)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    # Set up writer (this will create the directory)
    writer_storage_options = StorageOptions(uri=str(output_path), storage_id='sqlite3')
    writer = SequentialWriter()
    writer.open(writer_storage_options, converter_options)

    # Create topic metadata for writer
    for topic_info in topic_metadata:
        topic_meta = TopicMetadata(
            name=topic_info.name,
            type=topic_info.type,
            serialization_format='cdr'
        )
        writer.create_topic(topic_meta)

    # Read all messages and group by frame index
    frame_messages = defaultdict(list)  # frame_index -> [(topic, data, timestamp), ...]

    message_count = 0
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        frame_index = message_count // len(topic_metadata)  # Assuming synchronized topics
        frame_messages[frame_index].append((topic, data, timestamp))
        message_count += 1

    print(f"Read {message_count} messages across {len(frame_messages)} frames")

    # Downsample: keep every downsample_factor-th frame
    kept_frames = 0
    for frame_index in sorted(frame_messages.keys()):
        if frame_index % downsample_factor == 0:
            # Write all messages in this frame
            for topic, data, timestamp in frame_messages[frame_index]:
                writer.write(topic, data, timestamp)
            kept_frames += 1

    print(f"Kept {kept_frames} frames out of {len(frame_messages)}")
    print(f"Downsampled from {len(frame_messages)} to {kept_frames} frames")

    # Close writer
    writer.close()

    return kept_frames, len(frame_messages)

def downsample_all_rosbags(input_base_dir: str, output_base_dir: str, downsample_factor: int = 3):
    """
    Downsample all rosbags in a directory
    """
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    # Check if input_base itself is a bag directory
    if list(input_base.glob("*.db3")):
        bag_directories = [input_base]
    else:
        # Find all bag directories
        bag_directories = []
        for item in input_base.iterdir():
            if item.is_dir():
                # Check for bag files
                db3_files = list(item.glob("*.db3"))
                if db3_files:
                    bag_directories.append(item)

    bag_directories.sort()
    print(f"Found {len(bag_directories)} rosbag directories to process")

    total_original_frames = 0
    total_kept_frames = 0

    for i, bag_dir in enumerate(bag_directories):
        print(f"\n--- Processing {i+1}/{len(bag_directories)}: {bag_dir.name} ---")

        # Create corresponding output directory
        relative_path = bag_dir.relative_to(input_base)
        output_dir = output_base / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            kept_frames, original_frames = downsample_rosbag(str(bag_dir), str(output_dir), downsample_factor)
            total_kept_frames += kept_frames
            total_original_frames += original_frames
            print(f"✅ Successfully downsampled {bag_dir.name}")
        except Exception as e:
            print(f"❌ Error processing {bag_dir.name}: {e}")
            continue

    print(f"\n🎉 Completed downsampling:")
    print(f"   Total original frames: {total_original_frames}")
    print(f"   Total kept frames: {total_kept_frames}")
    print(f"   Downsample factor: {downsample_factor}")
    print(f"   Expected frequency: {60//downsample_factor}Hz")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python downsample_rosbag.py <input_bag_dir> <output_bag_dir> [downsample_factor]")
        print("  downsample_factor: 3 for 60Hz -> 20Hz (default: 3)")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    downsample_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if not Path(input_dir).exists():
        print(f"❌ Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    downsample_all_rosbags(input_dir, output_dir, downsample_factor)
