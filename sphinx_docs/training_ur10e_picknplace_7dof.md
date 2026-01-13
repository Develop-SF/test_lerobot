# Training UR10e Pick and Place with 7DoF (Right Arm + Gripper)

This guide explains how to train a diffusion policy for the UR10e robot performing pick and place tasks using the right arm (6 DoF) and Robotiq gripper (1 DoF).

## Overview

The pipeline supports flexible single or dual-arm configurations with the following features:

- **7 DoF Control**: 6 arm joints + 1 gripper joint
- **Relative Action Space**: Following UMI principles (arm joints relative, gripper absolute)
- **Multi-Camera Vision**: Head and front cameras with depth support
- **State Representation**: Position + velocity for both arm and gripper

## Prerequisites

1. **Hardware**:
   - UR10e robot with right arm trajectory controller
   - Robotiq 85 gripper
   - 2x Intel RealSense cameras (emily01/head and emily01/front)

2. **Software**:
   - ROS2 (tested with Humble)
   - LeRobot library
   - CUDA-capable GPU for training

3. **Data**:
   - ROS2 bag files with synchronized topics:
     - `/sync/ra_trajectory_controller/joint_trajectory`
     - `/sync/sns_right_gripper_cmd`
     - `/sync/joint_states`
     - `/sync/emily01/head/color/image_raw/compressed`
     - `/sync/emily01/front/color/image_raw/compressed`

## Pipeline Steps

### 1. Data Conversion

Convert ROS2 bags to LeRobot dataset format:

```bash
cd /home/shinfang-ovx/workspaces/test_lerobot
conda activate lerobot

# Run batch conversion for all bags in normal directory
./dev/data_processing/conversion/convert_normal_bags.sh
```

**Key Arguments**:
- `--arm right`: Process right arm joints
- `--right-arm-joints`: Specify RA joint names (ra_shoulder_pan_joint, etc.)
- `--gripper-joint`: Gripper state joint (ra_robotiq_85_left_knuckle_joint)
- `--gripper-action-topic`: Separate gripper command topic
- `--input-mode vision_pos_vel`: Include position + velocity in observations
- `--output-mode pos_vel`: Include position + velocity in actions

**Expected Output**:
- Dataset directory: `/mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal/`
- Features:
  - `observation.state`: [14] (7 positions + 7 velocities)
  - `action`: [14] (7 positions + 7 velocities)
  - `observation.images.head_cam`: [3, 240, 424]
  - `observation.images.front_cam`: [3, 240, 424]

### 2. Compute Relative Action Statistics

Calculate normalization statistics for relative action space:

```bash
./dev/data_processing/conversion/compute_stats_7dof.sh
```

This script:
- Uses `arm_dim=6` to treat only the 6 arm joints as relative
- Keeps the gripper (7th DoF) in absolute space
- Generates `relative_stats.json` in the dataset directory

**Why `arm_dim=6` for 7DoF?**
Following UMI principles, the gripper should remain in absolute space for pick-and-place tasks, as relative gripper commands can cause issues with precise grasping.

### 3. Train the Policy

Launch training with relative actions:

```bash
./dev/data_processing/conversion/train_7dof.sh
```

**Training Configuration** (`training_config_7dof.json`):
- **Policy Type**: Diffusion
- **Observation Horizon**: 2 steps
- **Action Horizon**: 8 steps
- **Batch Size**: 32
- **Total Steps**: 100,000
- **Device**: CUDA

**Output**:
- Checkpoints: `/mnt/nas/training/picknplace_7dof_normal_rel/`
- Frequency: Every 10,000 steps
- Logs: Every 200 steps

## Configuration for Different Arms

### Left Arm Only (6 DoF, no gripper)

```bash
python3 dev/data_processing/conversion/rosbag_to_lerobot_rosbag2.py \
    /path/to/bags \
    --arm left \
    --action-topics "/sync/la_trajectory_controller/joint_trajectory" \
    --input-mode "vision_pos_vel" \
    --output-mode "pos_vel"
```

Training config updates:
- `observation.state.shape: [12]` (6 pos + 6 vel)
- `action.shape: [12]` (6 pos + 6 vel)
- Use `--arm_dim 6` for stats and training

### Dual Arm (12 DoF + optional gripper)

```bash
python3 dev/data_processing/conversion/rosbag_to_lerobot_rosbag2.py \
    /path/to/bags \
    --arm both \
    --action-topics \
        "/sync/la_trajectory_controller/joint_trajectory" \
        "/sync/ra_trajectory_controller/joint_trajectory" \
        "/sync/sns_right_gripper_cmd" \
    --gripper-joint "ra_robotiq_85_left_knuckle_joint"
```

Training config updates:
- `observation.state.shape: [26]` (13 pos + 13 vel)
- `action.shape: [26]` (13 pos + 13 vel)
- Use `--arm_dim 12` for stats and training (both arms relative)

## Troubleshooting

### Issue: Action dimension mismatch
**Symptom**: Training fails with dimension errors.
**Solution**: Verify your dataset's `meta.json` matches the training config dimensions.

```bash
cat /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal/meta_data/info.json | grep -A 5 "action"
```

### Issue: Gripper not moving during inference
**Symptom**: Arm moves but gripper stays open/closed.
**Solution**: 
1. Check that gripper commands are in the dataset
2. Verify `--gripper-action-topic` is set during conversion
3. Ensure gripper state appears in `observation.state`

### Issue: Images are too dark/bright
**Symptom**: Poor visual performance.
**Solution**: The converter automatically downsizes images to 240x424. For different lighting, you may need to adjust camera exposure in ROS2 before recording.

## Advanced: Custom Joint Configurations

For robots with different joint names:

```bash
python3 dev/data_processing/conversion/rosbag_to_lerobot_rosbag2.py \
    /path/to/bags \
    --arm right \
    --right-arm-joints \
        "custom_shoulder_pan" \
        "custom_shoulder_lift" \
        "custom_elbow" \
        "custom_wrist_1" \
        "custom_wrist_2" \
        "custom_wrist_3" \
    --gripper-joint "custom_gripper_joint"
```

### 4. Inference and Deployment

After training, the model can be optimized and deployed using the generalized ONNX + TensorRT pipeline.

#### Step A: Optimize for Inference (ONNX + TensorRT)

Convert the PyTorch checkpoint to optimized ONNX models. This step now automatically stores hardware-specific metadata in `onnx_config.json`.

```bash
python3 dev/inference/testing_rel/convert_to_onnx_rel.py \
    --checkpoint /mnt/nas/training/picknplace_7dof_normal_rel/checkpoints/last/pretrained_model \
    --output-dir ./onnx_models \
    --arm-dim 6 \
    --joint-names \
        ra_shoulder_pan_joint \
        ra_shoulder_lift_joint \
        ra_elbow_joint \
        ra_wrist_1_joint \
        ra_wrist_2_joint \
        ra_wrist_3_joint \
        ra_robotiq_85_left_knuckle_joint
```

#### Step B: Offline Performance Evaluation

Replay a test ROS bag through the optimized model to verify prediction accuracy. The script automatically discovers camera topics based on the model's configuration.

```bash
python3 dev/inference/testing_rel/evaluate_predictions_onnx_rel.py \
    --checkpoint /mnt/nas/training/picknplace_7dof_normal_rel/checkpoints/last/pretrained_model \
    --onnx-dir ./onnx_models \
    --rosbag /mnt/nas/rosbags/test_eval_bag.mcap \
    --plot
```

**Features:**
- **Dynamic Topic Discovery**: Maps model feature keys (e.g. `head_cam`) to ROS topics in the bag.
- **Auto-Plotting**: Generates a grid of plots matching the robot's DoF (6, 7, etc.).
- **Ground Truth**: Compares predictions against recorded `joint_trajectory` commands.

#### Step C: Live ROS Deployment

Run the optimized model on the physical robot with low-latency TensorRT inference.

```bash
python3 dev/inference/testing_rel/inference_node_onnx_rel.py \
    --checkpoint /mnt/nas/training/picknplace_7dof_normal_rel/checkpoints/last/pretrained_model \
    --onnx-dir ./onnx_models \
    --mode continuous \
    --frequency 20.0
```

**Generalized Deployment Features:**
- **Universal Sensor Support**: Automatically creates subscribers for any number of cameras defined in training.
- **Smart Preprocessing**: Distinguishes between cameras (e.g., `head` gets automatic crop/rotate/resize, others get standard resize).
- **Auto-Publisher**: Guesses the action publication topic (`/left_arm/..` vs `/right_arm/..`) based on joint prefixes, or can be overridden via `--action-topic`.
- **Latency Monitoring**: Debug mode logs receive-to-inference timing for performance tuning.

### Hardware Generalization Matrix

The following table summarizes how to adapt the inference pipeline to different configurations:

| Hardware Change | Action Required during Conversion (`convert_to_onnx_rel.py`) | Result in Inference |
| :--- | :--- | :--- |
| **Change DoF** (e.g. 6 -> 7) | Update `--arm-dim` and `--joint-names`. | Node adapts buffers and message shapes automatically. |
| **New Joint Names** | Pass new names to `--joint-names`. | ROS and internal logs will use these identifiers. |
| **Add/Remove Camera** | No conversion args required; script reads model features. | Node automatically subscribes to the new topics. |
| **Change Robot Role** | Update `--joint-names` with correct prefix (`la_` vs `ra_`). | Node publishes to the correct arm controller automatically. |

## References

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [UMI Paper](https://umi-gripper.github.io/) - Relative action space motivation
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Policy architecture
