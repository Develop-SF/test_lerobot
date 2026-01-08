#!/bin/bash

# Convert approach_new rosbags to LeRobot dataset

# Environment setup
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# eval "$(conda shell.bash hook)"
# conda activate lerobot

# Configuration
BASE_DIR="/root/lerobot_ws/src/michael_pusht_q2_down"
OUTPUT_DIR="/root/lerobot_ws/src/dataset"

# Number of parallel workers for bag extraction
# Default: 1 (sequential, memory-safe)
# For faster processing with sufficient RAM: 4-8 workers
# Each worker loads a full episode into memory
NUM_WORKERS=1

echo "Converting rosbags to LeRobot dataset"
echo "Source: $BASE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Parallel workers: $NUM_WORKERS (1=sequential/memory-safe, 4-8=faster/more RAM)"

# Count episodes
EPISODE_COUNT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "michael_pusht_*" | wc -l)
echo "Found $EPISODE_COUNT rosbag episodes"

# Confirmation
read -p "Convert all $EPISODE_COUNT episodes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting conversion..."
    echo ""
    
    python rosbag_to_lerobot_rosbag2.py \
        "$BASE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "michael_pusht_q2_down" \
        --fps 20 \
        --task "Michael pusht task with start from q2 t-down left arm" \
        --tolerance 1.0 \
        --no-trim-unmoving \
        --input-mode vision_pos \
        --output-mode pos_only \
        --num-workers $NUM_WORKERS
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Conversion completed successfully"
        echo "Dataset saved to: $OUTPUT_DIR"
        echo "Processed with $NUM_WORKERS parallel workers"
    else
        echo ""
        echo "Conversion failed"
        exit 1
    fi
else
    echo "Conversion cancelled"
    exit 0
fi
