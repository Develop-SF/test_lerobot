# Testing Branch - Complete Workflow Guide

This guide provides a comprehensive walkthrough for converting ROS bags to datasets, training models, converting to ONNX, and running inference to control robots.

## Overview of Changes in Testing Branch

The `testing` branch contains significant enhancements over the main branch:

### Key Commits:
1. **c9e57b0** - Update documentation
2. **136fa06** - Update inference with ONNX support
3. **8b33fe1** - Update LeRobot core with relative action support
4. **987abec** - Add development code (conversion, visualization, inference)
5. **4674f48** - Initial testing setup

### Major Additions (38 files, 14,110+ lines):
- **ROS bag conversion tools** with flexible input/output modes
- **Relative action training** support (UMI/ManiSkill-inspired)
- **ONNX/TensorRT inference** nodes for optimized deployment
- **ROS2 integration** for real robot control
- **Comprehensive testing and evaluation** scripts

---

## Workflow Overview

```
ROS Bags → LeRobot Dataset → Train Model → Convert to ONNX → ROS2 Inference → Robot Control
```

---

## 1. Convert ROS Bag to LeRobot Dataset

### Available Conversion Scripts

#### Option A: Standard Converter (rosbag_to_lerobot_rosbag2.py)
Basic conversion with flexible input/output modes.

**Features:**
- Supports multiple input modes: `vision_only`, `vision_pos`, `vision_pos_vel`
- Supports multiple output modes: `pos_only`, `pos_vel`
- Configurable FPS and image preprocessing
- Automatic trimming of unmoving segments

**Usage:**
```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/data_processing/conversion

python rosbag_to_lerobot_rosbag2.py \
    /path/to/rosbags \
    --output-dir /path/to/output/dataset \
    --dataset-name "my_robot_dataset" \
    --fps 20 \
    --task "Robot manipulation task description" \
    --tolerance 1.0 \
    --input-mode vision_pos \
    --output-mode pos_vel
```

**Using the shell script:**
```bash
# Edit the script to set your paths
vim run_approach_new_converter.sh

# Run the conversion
bash run_approach_new_converter.sh
```

#### Option B: Cropped & Trimmed Converter (rosbag_to_lerobot_cropped_trimmed.py)
Advanced converter with image cropping, rotation, and precise trimming.

**Features:**
- Custom image preprocessing (crop, rotate, resize)
- Top view: Crop → Rotate 90° CW → Resize to 224×178
- Left arm: Resize to 224×178
- Precise episode trimming

**Usage:**
```bash
python rosbag_to_lerobot_cropped_trimmed.py \
    /path/to/rosbags \
    --output-dir /path/to/output/dataset \
    --dataset-name "cropped_dataset" \
    --fps 20 \
    --trim-start 30 \
    --trim-end 30
```

### Input/Output Modes

**Input Modes:**
- `vision_only`: Only camera images (no proprioception)
- `vision_pos`: Camera images + joint positions (6D)
- `vision_pos_vel`: Camera images + joint positions + velocities (12D)

**Output Modes:**
- `pos_only`: Joint positions only (6D)
- `pos_vel`: Joint positions + velocities (12D)

### Expected ROS Topics

The converter expects these ROS2 topics:
- `/sync/emily01/left_arm/color/image_raw/compressed` - Left arm camera
- `/sync/emily01/head/color/image_raw/compressed` - Head/top view camera
- `/sync/joint_states` - Joint state information

### Verify Dataset

After conversion, inspect the dataset:
```bash
cd /home/shinfang-ovx/workspaces/test_lerobot

# Load and inspect dataset
python inspect_dataset.py /path/to/output/dataset

# Or use the provided script
python load_local_dataset.py
```

---

## 2. Train the Model

### Option A: Standard Training (Absolute Actions)

Train with standard absolute action representation:

```bash
cd /home/shinfang-ovx/workspaces/test_lerobot

python lerobot/scripts/train.py \
    --config-path /path/to/config.json \
    --output-dir /path/to/output \
    --dataset-repo-id /path/to/local/dataset
```

**Using the example script:**
```bash
python examples/3_train_policy.py
```

### Option B: Relative Action Training

Train with relative action representation (recommended for better generalization):

```bash
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /path/to/config.json \
    --output_dir /path/to/output \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 2
```

**Key Parameters:**
- `--use_relative_actions`: Enable relative action mode
- `--arm_dim 6`: Number of arm joints (excluding gripper)
- `--obs_horizon 2`: Number of observation frames to use
- `--relative_stats_path`: Optional pre-computed statistics

### Compute Relative Action Statistics (Optional)

Pre-compute normalization statistics for relative actions:

```bash
python lerobot/scripts/compute_relative_action_stats.py \
    --dataset-path /path/to/dataset \
    --arm-dim 6 \
    --obs-horizon 2 \
    --output-path ./relative_stats.json
```

### Training Configuration

Example config structure:
```json
{
  "policy": {
    "name": "diffusion",
    "input_shapes": {
      "observation.images.top": [3, 224, 178],
      "observation.images.left_arm": [3, 224, 178],
      "observation.state": [12]
    },
    "output_shapes": {
      "action": [12]
    },
    "horizon": 16,
    "n_action_steps": 8
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 1000,
    "learning_rate": 1e-4
  }
}
```

### Monitor Training

Training outputs are saved in the output directory:
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs
- `config.json` - Full configuration

---

## 3. Convert Model to ONNX

After training, convert the PyTorch model to ONNX for optimized inference.

### For Absolute Action Models

```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_abs

python convert_to_onnx.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models \
    --opset-version 17
```

### For Relative Action Models

```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_rel

python convert_to_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models \
    --use-relative-actions \
    --arm-dim 6 \
    --opset-version 17
```

**What gets exported:**
- `vision_encoder.onnx` - RGB encoder network
- `noise_pred_net.onnx` - Diffusion UNet
- `config.json` - Model configuration
- `normalization_stats.json` - Data statistics

### Verify ONNX Models

Test the ONNX models before deployment:
```bash
# Test with dummy inputs
python -c "import onnxruntime as ort; \
    session = ort.InferenceSession('./onnx_models/vision_encoder.onnx'); \
    print('Model loaded successfully!')"
```

---

## 4. Run Inference & Control Robot

### Test Inference (Without Robot)

First, test inference with recorded data:

```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_abs

# Test PyTorch inference
python lerobot_inference.py /path/to/checkpoint --device cuda

# Test with rosbag
python test_rosbag.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/test.bag
```

### Evaluate Predictions

Compare model predictions against ground truth:

**For Absolute Actions:**
```bash
python evaluate_predictions_onnx.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/rosbag \
    --onnx-dir ./onnx_models \
    --plot
```

**For Relative Actions:**
```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_rel

python evaluate_predictions_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/rosbag \
    --onnx-dir ./onnx_models \
    --plot
```

### Run ROS2 Inference Node

Deploy the model with ROS2 for real-time robot control.

#### Absolute Action Mode

**PyTorch Inference:**
```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_abs

# Continuous mode (fixed frequency)
python inference_node.py \
    --checkpoint /path/to/checkpoint \
    --device cuda \
    --frequency 20.0 \
    --mode continuous
```

**ONNX Inference (Optimized):**
```bash
python inference_node_onnx.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

#### Relative Action Mode

**ONNX + TensorRT (Recommended):**
```bash
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_rel

python inference_node_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

### ROS2 Topics

**Subscribed Topics:**
- `/sync/emily01/left_arm/color/image_raw/compressed` - Left arm camera
- `/sync/emily01/head/color/image_raw/compressed` - Top view camera
- `/sync/joint_states` - Current joint states (if needed)

**Published Topics:**
- `/left_arm/joint_trajectory` - Predicted joint commands

### Inference Modes

**Continuous Mode:**
- Runs at fixed frequency (e.g., 20 Hz)
- Predictable timing
- Best for closed-loop control

**Triggered Mode:**
- Runs on new sensor data
- Variable frequency
- Uses latest data

---

## 5. Monitoring & Debugging

### Check Topic Rates

```bash
# Camera topics
ros2 topic hz /sync/emily01/left_arm/color/image_raw/compressed
ros2 topic hz /sync/emily01/head/color/image_raw/compressed

# Action commands
ros2 topic hz /left_arm/joint_trajectory
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

### View Logs

```bash
# ROS2 logs
ros2 node info /lerobot_inference_node

# Check for errors
ros2 topic echo /rosout
```

### Common Issues

#### Issue: CUDA Out of Memory
**Solution:**
```bash
# Use CPU
python inference_node.py --device cpu

# Or reduce batch size/frequency
python inference_node.py --frequency 10.0
```

#### Issue: Image Size Mismatch
**Solution:** Verify preprocessing matches training pipeline. Check crop/resize parameters.

#### Issue: Robot Not Moving
**Solutions:**
1. Verify topic names match
2. Check robot controller is subscribed
3. Verify model outputs non-zero actions
4. Check sensor data is publishing

---

## 6. Performance Optimization

### GPU Optimization

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable TensorRT optimizations
export ONNXRUNTIME_PROVIDER=TensorRT
```

### CPU Optimization

```bash
# Use all CPU cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Frequency Tuning

Start conservative and increase gradually:
```bash
# Start at 10 Hz
python inference_node.py --frequency 10.0

# Increase to 20 Hz if stable
python inference_node.py --frequency 20.0

# Maximum (if system can handle)
python inference_node.py --frequency 30.0
```

---

## 7. Complete Example Workflow

Here's a complete end-to-end example:

```bash
# 1. Convert ROS bag to dataset
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/data_processing/conversion

python rosbag_to_lerobot_rosbag2.py \
    /data/rosbags/experiment_01 \
    --output-dir /data/datasets/experiment_01_dataset \
    --dataset-name "experiment_01" \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel

# 2. Verify dataset
cd /home/shinfang-ovx/workspaces/test_lerobot
python inspect_dataset.py /data/datasets/experiment_01_dataset

# 3. Compute relative action statistics (optional)
python lerobot/scripts/compute_relative_action_stats.py \
    --dataset-path /data/datasets/experiment_01_dataset \
    --arm-dim 6 \
    --output-path /data/datasets/experiment_01_stats.json

# 4. Train model with relative actions
python lerobot/scripts/train_with_relative_actions.py \
    --config_path ./configs/diffusion_policy.json \
    --output_dir /data/models/experiment_01 \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 2 \
    --relative_stats_path /data/datasets/experiment_01_stats.json

# 5. Convert to ONNX
cd /home/shinfang-ovx/workspaces/test_lerobot/dev/inference/testing_rel

python convert_to_onnx_rel.py \
    --checkpoint /data/models/experiment_01/checkpoints/best \
    --output-dir /data/models/experiment_01/onnx \
    --use-relative-actions \
    --arm-dim 6

# 6. Evaluate on test rosbag
python evaluate_predictions_onnx_rel.py \
    --checkpoint /data/models/experiment_01/checkpoints/best \
    --rosbag /data/rosbags/test_01 \
    --onnx-dir /data/models/experiment_01/onnx \
    --plot

# 7. Deploy on real robot
python inference_node_onnx_rel.py \
    --checkpoint /data/models/experiment_01/checkpoints/best \
    --onnx-dir /data/models/experiment_01/onnx \
    --device cuda \
    --frequency 20.0
```

---

## 8. Safety Considerations

⚠️ **Important Safety Guidelines:**

1. **Test in simulation first** before deploying on real hardware
2. **Start with low frequencies** (5-10 Hz) and increase gradually
3. **Monitor robot behavior** closely during initial deployment
4. **Have emergency stop ready** at all times
5. **Verify action limits** are within safe ranges
6. **Use triggered mode** initially for better control
7. **Test with rosbag data** before live deployment

---

## 9. Additional Resources

### Documentation Files
- [dev/inference/README.md](dev/inference/README.md) - Inference overview
- [dev/inference/testing_abs/README.md](dev/inference/testing_abs/README.md) - Absolute actions
- [dev/inference/testing_abs/DEPLOYMENT_GUIDE.md](dev/inference/testing_abs/DEPLOYMENT_GUIDE.md) - Deployment guide
- [dev/inference/testing_abs/TESTING_GUIDE.md](dev/inference/testing_abs/TESTING_GUIDE.md) - Testing guide
- [dev/inference/testing_rel/README.md](dev/inference/testing_rel/README.md) - Relative actions

### Utility Scripts
- `inspect_dataset.py` - Inspect LeRobot datasets
- `load_local_dataset.py` - Load and test local datasets
- `dev/data_processing/visualization/` - Visualization tools
- `dev/inference/testing_abs/test_inference.py` - Test inference pipeline

---

## 10. Architecture Overview

### Data Flow

```
ROS Bag (Raw Sensor Data)
    ↓
LeRobot Dataset (Preprocessed)
    ↓
Training (PyTorch)
    ↓
Trained Model (Checkpoint)
    ↓
ONNX Export (Optimized)
    ↓
ROS2 Inference Node
    ↓
Robot Controller
```

### Image Preprocessing Pipeline

```
Camera Image (640×480)
    ↓
[Top View: Crop → Rotate 90° CW]
[Left Arm: Direct Resize]
    ↓
Resized (224×178)
    ↓
Normalize [0, 1]
    ↓
CHW Format (3×224×178)
    ↓
Model Input
```

### Action Processing Pipeline

**Absolute Actions:**
```
Model Output → Denormalize → Joint Commands
```

**Relative Actions:**
```
Model Output (Δ) → Add to Current State → Joint Commands
```

---

## Summary

This testing branch provides a complete pipeline for:
1. ✅ Converting ROS bags to LeRobot datasets
2. ✅ Training with absolute or relative actions
3. ✅ Converting models to ONNX for optimization
4. ✅ Deploying models via ROS2 for real robot control
5. ✅ Comprehensive testing and evaluation tools

The key improvements over main branch:
- **Flexible data conversion** with multiple input/output modes
- **Relative action training** for better generalization
- **ONNX/TensorRT optimization** for faster inference
- **Production-ready ROS2 nodes** with proper error handling
- **Extensive testing and evaluation** tools

Start with the conversion scripts, train your model, and gradually work through testing before deploying on real hardware. Good luck!
