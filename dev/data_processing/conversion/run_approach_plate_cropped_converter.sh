#!/bin/bash

# Convert approach_plate rosbags to LeRobot dataset with cropping and trimming

# Environment setup
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
eval "$(conda shell.bash hook)"
conda activate lerobot

# Configuration
BASE_DIR="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot_1116/approach_real_bing_20hz"
OUTPUT_DIR="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot_1116/approach_real_bing_20hz_lerobot/dataset/input-vision_pos-output-pos"
MAPPING_FILE="/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot_1116/optical_flow_videos/episode_mapping.json"

# Crop box for top view (head camera)
CROP_X=260
CROP_Y=135
CROP_W=178
CROP_H=224

# Episodes to skip (none)
SKIP_EPISODES=""

echo "========================================="
echo "ROS Bag to LeRobot Converter"
echo "With Cropping and Trimming"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Source: $BASE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Mapping: $MAPPING_FILE"
echo "  Crop box (top view): x=$CROP_X, y=$CROP_Y, w=$CROP_W, h=$CROP_H"
echo "  Skip episodes: None"
echo ""

# Count episodes
EPISODE_COUNT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "plate_real_bing*" | wc -l)
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
        --dataset-name "approach_plate_cropped_trimmed" \
        --fps 20 \
        --task "Robot approach plate task with cropped top view and trimmed frames" \
        --tolerance 1.0 \
        --no-trim-unmoving \
        --no-downsize \
        --input-mode vision_pos \
        --output-mode pos_only \
        --crop $CROP_X $CROP_Y $CROP_W $CROP_H \
        --episode-mapping "$MAPPING_FILE" \
        --observation-topics \
            "/sync/emily01/left_arm/color/image_raw/compressed" \
            "/sync/emily01/head/color/image_raw/compressed" \
            "/sync/joint_states" \
        --action-topics \
            "/sync/la_trajectory_controller/joint_trajectory"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Conversion completed successfully!"
        echo "========================================="
        echo "Dataset saved to: $OUTPUT_DIR"
        echo ""
        echo "Summary:"
        echo "  - Top view: cropped to ${CROP_W}x${CROP_H}, rotated 90°, final size 224x178"
        echo "  - Left arm: resized to 224x178"
        echo "  - Both cameras at matching resolution: 224x178"
        echo "  - All episodes processed (0-25)"
        echo "  - Episode-specific frame trimming applied"
        echo "  - All topics aligned to same message count"
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
