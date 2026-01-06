# Quick Reference - Testing Branch Workflow

## Setup, Tips & Troubleshooting

### Environment Setup
- **Baseline Setup**: Follow LeRobot's README to set up the conda environment and install LeRobot locally.
- **FFmpeg & TorchCodec (Critical Fixes)**:
  - Do **not** use `apt-get` for ffmpeg inside Conda.
  - Install FFmpeg 7+ via Conda-Forge: `conda install -y -c conda-forge 'ffmpeg>=7.0'`
  - Install compatible TorchCodec: `pip install torchcodec==0.2.1`
  - *Verified Stack*: FFmpeg 7 + TorchCodec 0.2.1 + PyTorch 2.6.0.

### Runtime Tips
- **Suppress Verbose Logs**: Set `export SVT_LOG=1` to suppress distracting info logs.
- **Parallel Conversion**: Use `run_parallel_conversion.sh` for speed.
- **Script Config**: Edit simpler parameters (`BASE_DIR`, `--dataset-name`) in conversion scripts rather than deep logic.

### Known Issues & Fixes
- **`lerobot_dataset.py` Torch Input Error**:
  - *Issue*: `torch.stack` fails on lists of floats (e.g., timestamps).
  - *Fix*: Code modified to use `torch.tensor()` which accepts lists, or robustly check types before stacking.
- **`libtorchcodec` Loading Errors**:
  - *Issue*: `RuntimeError` due to missing `libavutil.so` or symbol mismatch.
  - *Fix*: See "FFmpeg & TorchCodec" setup above.

### Validated Command Examples

**Compute Relative Stats:**
```bash
python lerobot/scripts/compute_relative_action_stats.py \
    --config_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/training_config.json \
    --output_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json \
    --arm_dim 6
```

**Train with Relative Actions:**
```bash
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/training_config.json \
    --output_dir /mnt/SF-Shared/training/eric_plating_v2_rel \
    --use_relative_actions \
    --arm_dim 6 \
    --relative_stats_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json
```

## 🎯 Quick Commands

### 1️⃣ Convert ROS Bag → Dataset

```bash
cd dev/data_processing/conversion

# Standard conversion
python rosbag_to_lerobot_rosbag2.py \
    /path/to/rosbags \
    --output-dir /path/to/dataset \
    --dataset-name "my_dataset" \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel

# OR use shell script
bash run_approach_new_converter.sh
```

### 2️⃣ Train Model

```bash
# Absolute actions
python lerobot/scripts/train.py \
    --config-path config.json \
    --output-dir /path/to/output

# Relative actions (recommended)
python lerobot/scripts/train_with_relative_actions.py \
    --config_path config.json \
    --output_dir /path/to/output \
    --use_relative_actions \
    --arm_dim 6
```

### 3️⃣ Convert to ONNX

```bash
# Absolute actions
cd dev/inference/testing_abs
python convert_to_onnx.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models

# Relative actions
cd dev/inference/testing_rel
python convert_to_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models \
    --use-relative-actions \
    --arm-dim 6
```

### 4️⃣ Run Inference

```bash
# Test with rosbag
python test_rosbag.py \
    --checkpoint /path/to/checkpoint \
    --rosbag /path/to/test.bag

# Deploy on robot
python inference_node_onnx.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

---

## 📋 Input/Output Modes

| Mode | Description | Observation Dim | Action Dim |
|------|-------------|----------------|------------|
| `vision_only` | Images only | Images | 6 or 12 |
| `vision_pos` | Images + positions | Images + 6 | 6 or 12 |
| `vision_pos_vel` | Images + pos + vel | Images + 12 | 12 |
| `pos_only` | Position outputs | - | 6 |
| `pos_vel` | Position + velocity | - | 12 |

---

## 📡 ROS2 Topics

**Subscribe:**
- `/sync/emily01/left_arm/color/image_raw/compressed`
- `/sync/emily01/head/color/image_raw/compressed`
- `/sync/joint_states`

**Publish:**
- `/left_arm/joint_trajectory`

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `--device cpu` or `--frequency 10.0` |
| Image size mismatch | Check crop/resize parameters |
| Robot not moving | Verify topics and controller |
| Slow inference | Use ONNX, reduce frequency |

---

## 📂 Directory Structure

```
dev/
├── data_processing/
│   ├── conversion/          # ROS bag converters
│   └── visualization/       # Visualization tools
└── inference/
    ├── testing_abs/         # Absolute action inference
    └── testing_rel/         # Relative action inference

lerobot/
├── scripts/
│   ├── train.py                        # Standard training
│   ├── train_with_relative_actions.py  # Relative training
│   └── compute_relative_action_stats.py
└── common/
    └── datasets/
        └── relative_action_dataset.py
```

---

## 🚀 Complete Pipeline Example

```bash
# Step 1: Convert
python rosbag_to_lerobot_rosbag2.py /data/bags --output-dir /data/dataset --dataset-name exp1 --fps 20

# Step 2: Train
python train_with_relative_actions.py --config_path config.json --output_dir /data/model --use_relative_actions

# Step 3: Convert to ONNX
python convert_to_onnx_rel.py --checkpoint /data/model/checkpoints/best --output-dir /data/model/onnx --use-relative-actions

# Step 4: Test
python evaluate_predictions_onnx_rel.py --checkpoint /data/model/checkpoints/best --rosbag /data/test.bag --onnx-dir /data/model/onnx

# Step 5: Deploy
python inference_node_onnx_rel.py --checkpoint /data/model/checkpoints/best --onnx-dir /data/model/onnx --frequency 20.0
```

---

## ⚠️ Safety Checklist

- [ ] Test in simulation first
- [ ] Start with low frequency (5-10 Hz)
- [ ] Monitor robot behavior
- [ ] Emergency stop ready
- [ ] Verify action limits
- [ ] Test with rosbag data first

---

## 🔍 Useful Commands

```bash
# Check git diff
git diff main...testing --stat

# Inspect dataset
python inspect_dataset.py /path/to/dataset

# Monitor ROS topics
ros2 topic hz /left_arm/joint_trajectory

# Watch GPU
watch -n 1 nvidia-smi

# Test ONNX model
python -c "import onnxruntime; print(onnxruntime.get_device())"
```

---

For detailed information, see [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md)

### 5️⃣ Generate Training Config (Official/Hydra Method)

Instead of writing the JSON manually, use the standard LeRobot training script to generate a valid base configuration (YAML), then convert it to JSON and adapt it.

**1. Generate Base Configuration**
Run the standard training script for 1 step to generate the config file:
```bash
python lerobot/scripts/train.py \
    policy=diffusion \
    dataset.repo_id=eric_plating_v2 \
    env.task=eric_plating_v2 \
    device=cpu \
    steps=1 \
    hydra.run.dir=outputs/config_gen
```
> **Flags Explanation:**
> *   `device=cpu`: Use CPU to avoid unnecessary GPU initialization overhead for just generating a config.
> *   `steps=1`: Run for only 1 step so the script exits immediately (side-effect: generating the config).
> *   `hydra.run.dir=...`: Save outputs to a known, fixed directory (`outputs/config_gen`) instead of a timestamped folder.

**2. Retrieve and Convert Config**
The configuration will be saved at `outputs/config_gen/config.yaml`. Convert this YAML to JSON (e.g., using an online tool or script) to create `training_config.json`.

**3. Manual Adjustments for Custom Script**
Our custom script (`train_with_relative_actions.py`) requires specific keys that might be missing or different in the standard config:

- **`policy.repo_id`**: This is **REQUIRED** but often missing in local generic runs. Add it manually to the `policy` section:
  ```json
  "policy": {
      "repo_id": "eric_plating_v2_policy",
      ...
  }
  ```
- **Feature Matching**: Ensure `input_features` and `output_features` match your dataset exactly (check image keys like `observation.images.sync_head_cam`). The standard script attempts to guess them, but verify they are correct.
- **Paths**: Ensure `dataset.root` points to your local dataset folders if not using Hub.

---


