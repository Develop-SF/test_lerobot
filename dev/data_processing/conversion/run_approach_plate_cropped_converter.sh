#!/bin/bash

# Convert approach_plate rosbags to LeRobot dataset with cropping and trimming

# Environment setup
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# eval "$(conda shell.bash hook)"
# conda activate lerobot

# Configuration
BASE_DIR="/root/nas/rosbags/michael_pusht_cart"
OUTPUT_DIR="/root/nas/dataset/robot_learning/lerobot/michael_pusht_cart_q234"
MAPPING_FILE=""

# suppress verbose SVT logs
export SVT_AV1_LOG=1

# Crop box for top view (head camera)
CROP_X=220
CROP_Y=134
CROP_W=224
CROP_H=224

# Episodes to skip (none)
SKIP_EPISODES=""

# Number of parallel workers for bag extraction
# Default: 1 (sequential, memory-safe)
# For faster processing with sufficient RAM: 4-8 workers
# Each worker loads a full episode into memory
NUM_WORKERS=2

echo "========================================="
echo "ROS Bag to LeRobot Converter"
echo "With Cropping and Trimming (Parallel)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Source: $BASE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Mapping: None (no episode mapping file)"
echo "  Crop box (top view): x=$CROP_X, y=$CROP_Y, w=$CROP_W, h=$CROP_H"
echo "  Skip episodes: None"
echo "  Parallel workers: $NUM_WORKERS (1=sequential/memory-safe, 4-8=faster/more RAM)"
echo ""

# Count episodes
EPISODE_COUNT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "*" | wc -l)
echo "Found $EPISODE_COUNT rosbag episodes"
echo ""

# Confirmation
read -p "Convert all episodes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting conversion..."
    echo ""
    
    python rosbag_to_lerobot_cropped_trimmed.py \
        "$BASE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "michael_pusht_cart_q234" \
        --fps 20 \
        --task "Michael pusht start from q2 & q3 & q4" \
        --tolerance 1.0 \
        --no-trim-unmoving \
        --no-downsize \
        --input-mode vision_pos \
        --output-mode pos_only \
        --crop $CROP_X $CROP_Y $CROP_W $CROP_H \
        --num-workers $NUM_WORKERS \
        --cartesian \
        --observation-topics \
            "/sync/emily01/left_arm/color/image_raw/compressed" \
            "/sync/emily01/head/color/image_raw/compressed" \
            "/sync/joint_states" \
        --action-topics \
            "/sync/la/servo_node/delta_twist_cmds" \
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Conversion completed successfully!"
        echo "========================================="
        echo "Dataset saved to: $OUTPUT_DIR"
        echo ""
        echo "Summary:"
        echo "  - Top view: cropped to ${CROP_W}x${CROP_H}, rotated 90°, final size 224x224"
        echo "  - Left arm: resized to 224x224"
        echo "  - Both cameras at matching resolution: 224x224"
        echo "  - All episodes processed (0-72)"
        echo "  - Episode-specific frame trimming applied"
        echo "  - All topics aligned to same message count"
        echo "  - Processing mode: $([ $NUM_WORKERS -eq 1 ] && echo 'sequential (memory-safe)' || echo "parallel with $NUM_WORKERS workers")"
    else
        echo ""
        echo "========================================="
        echo "Conversion failed"
        echo "========================================="
        exit 1
    fi
else
    echo "Conversion cancelled"
    exit 0
fi
