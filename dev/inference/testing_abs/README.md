# LeRobot Inference for Approach Plate Dataset

This directory contains inference code for deploying trained LeRobot models from the approach_plate dataset.

## Key Features

The inference code includes **image preprocessing** that matches the training pipeline:

### Image Preprocessing Pipeline

1. **Top View (Head Camera)**:
   - Crop to bounding box: `(x=260, y=135, w=178, h=224)`
   - Rotate 90° clockwise
   - Final size: **224×178** (width×height)

2. **Left Arm Camera**:
   - Resize to **224×178** (width×height)

Both cameras output the same resolution (224×178) for the model.

## Files

- `lerobot_inference.py` - Core inference module with automatic mode detection and image preprocessing
- `inference_node.py` - ROS2 node for real-time deployment
- `README.md` - This file

## Usage

### Prerequisites: Convert to ONNX

If using the ONNX inference node (`inference_node_onnx.py`), you must first convert the model:

```bash
python convert_to_onnx.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models
```

### 1. Test Inference Module

```bash
python lerobot_inference.py /path/to/checkpoint --device cuda
```

### 2. Run ROS2 Inference Node

#### Continuous Mode (Fixed Frequency)
```bash
python inference_node.py \
    --checkpoint /home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_plate_lerobot \
    --device cuda \
    --frequency 20.0 \
    --mode continuous
```

#### Triggered Mode (Event-Driven)
```bash
python inference_node.py \
    --checkpoint /path/to/checkpoint \
    --device cuda \
    --mode triggered
```

### 3. Run ONNX Inference Node (Optimized)

```bash
python inference_node_onnx.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

### 4. Evaluate Predictions (ONNX)

```bash
python evaluate_predictions_onnx.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/rosbag \
    --onnx-dir ./onnx_models \
    --plot
```

## Input/Output Modes

The inference system automatically detects the model's input and output modes:

### Input Modes
- `vision_only` - Only camera images (no joint states)
- `vision_pos` - Camera images + joint positions (6D)
- `vision_pos_vel` - Camera images + joint positions + velocities (12D)

### Output Modes
- `pos_only` - Joint positions only (6D)
- `pos_vel` - Joint positions + velocities (12D)

## ROS Topics

### Subscribed Topics
- `/sync/emily01/left_arm/color/image_raw/compressed` - Left arm camera
- `/sync/emily01/head/color/image_raw/compressed` - Head camera (top view)
- `/sync/joint_states` - Joint states (if required by model)

### Published Topics
- `/left_arm/joint_trajectory` - Predicted joint trajectory commands

## Differences from Original Pipeline

This inference code differs from the `approach_lerobot/testing` code in the following ways:

1. **Image Cropping**: Top view is cropped to focus on the task area
2. **Image Rotation**: Top view is rotated 90° clockwise after cropping
3. **Image Resizing**: Both cameras are resized to matching 224×178 resolution
4. **No Downsampling**: Images maintain higher resolution compared to the original 180×320

These preprocessing steps match exactly what was done during training with `rosbag_to_lerobot_cropped_trimmed.py`.

## Architecture

```
ROS Messages → Image Preprocessing → Model Inference → Action Publishing
                     ↓
              - Crop (top view)
              - Rotate (top view)
              - Resize (both cameras)
              - Normalize [0,1]
              - Convert to CHW format
```

## Notes

- The inference system automatically validates that incoming images match the expected preprocessing
- Image preprocessing is applied in `decode_compressed_image_msg()` with the `is_top_view` flag
- The model expects images at 224×178 resolution (width×height)
- All preprocessing parameters are hardcoded to match the training pipeline
