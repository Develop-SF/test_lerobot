#!/bin/bash
# Run the cropped and trimmed video converter
# This script processes ROS bag files to create cropped and trimmed videos

# Configuration
MAPPING_FILE="optical_flow_videos_approach_plate/episode_mapping.json"
OUTPUT_DIR="cropped_trimmed_videos_approach_plate"
CROP_X=247
CROP_Y=122
CROP_W=193
CROP_H=237
FPS=20

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Cropped & Trimmed Video Converter${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Mapping file: $MAPPING_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Crop box: x=$CROP_X, y=$CROP_Y, w=$CROP_W, h=$CROP_H"
echo "  FPS: $FPS"
echo ""

# Check if mapping file exists
if [ ! -f "$MAPPING_FILE" ]; then
    echo -e "${YELLOW}Warning: Mapping file not found: $MAPPING_FILE${NC}"
    echo "Please ensure the episode_mapping.json file exists."
    exit 1
fi

# Run the converter
echo -e "${GREEN}Starting conversion...${NC}"
echo ""

python3 cropped_trimmed_video_viz.py \
    --mapping "$MAPPING_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --crop $CROP_X $CROP_Y $CROP_W $CROP_H \
    --fps $FPS

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Conversion completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Output videos saved to: $OUTPUT_DIR"
else
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Conversion failed or incomplete${NC}"
    echo -e "${YELLOW}========================================${NC}"
    exit 1
fi
