# LeRobot Inference - Relative Action Mode

This directory contains inference code for deploying LeRobot models trained with **relative action representation**.

## Key Features

- **Relative Actions**: Handles models that output actions as relative changes from the current state.
- **ONNX + TensorRT**: Optimized inference pipeline using ONNX Runtime with TensorRT backend.
- **Image Preprocessing**: Matches the training pipeline:
  - **Top View**: Crop `(x=260, y=135, w=178, h=224)` -> Rotate 90° CW -> Final `224x178`
  - **Left Arm**: Resize to `224x178`

## Files

- `inference_node_onnx_rel.py`: Main ROS2 node for real-time inference using ONNX/TensorRT.
- `evaluate_predictions_onnx_rel.py`: Script to evaluate model predictions against ground truth from rosbags.
- `convert_to_onnx_rel.py`: Utility to convert PyTorch models to ONNX format with relative action support.

## Prerequisites

Before running inference, you must convert the PyTorch model to ONNX format.

### Convert Model to ONNX

```bash
python convert_to_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models \
    --use-relative-actions \
    --arm-dim 6
```

## Usage

### Run Inference Node

```bash
python inference_node_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

### Evaluate Predictions

```bash
python evaluate_predictions_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/rosbag \
    --onnx-dir ./onnx_models \
    --plot
```

**Note**: The `--plot` flag enables visualization of predicted vs. ground truth trajectories.
