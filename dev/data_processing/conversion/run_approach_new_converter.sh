#!/bin/bash

# Convert approach_new rosbags to LeRobot dataset

# Environment setup
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
eval "$(conda shell.bash hook)"
conda activate lerobot

# Configuration
BASE_DIR="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_new"
OUTPUT_DIR="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_lerobot/dataset/input-vision_pos-output-pos_vel"

echo "Converting rosbags to LeRobot dataset"
echo "Source: $BASE_DIR"
echo "Output: $OUTPUT_DIR"

# Count episodes
EPISODE_COUNT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "sync_approach*" | wc -l)
echo "Found $EPISODE_COUNT rosbag episodes"

# Confirmation
read -p "Convert all $EPISODE_COUNT episodes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting conversion..."
    
    python rosbag_to_lerobot_rosbag2.py \
        "$BASE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "input-vision_pos-output-pos_vel" \
        --fps 20 \
        --task "Robot approach task with dual cameras and left arm control" \
        --tolerance 1.0 \
        --no-trim-unmoving \
        --input-mode vision_pos \
        --output-mode pos_vel
    
    if [ $? -eq 0 ]; then
        echo "Conversion completed successfully"
        echo "Dataset saved to: $OUTPUT_DIR"
    else
        echo "Conversion failed"
        exit 1
    fi
else
    echo "Conversion cancelled"
    exit 0
fi
