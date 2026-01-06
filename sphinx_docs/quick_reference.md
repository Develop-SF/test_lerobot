# Quick Reference - Testing Branch Workflow

## Setup and Tips

- **Environment Setup**: Follow LeRobot's README to set up the conda environment and install LeRobot locally.
- **Suppress Verbose Logs**: Verbose Svt[info] logs can be distracting. Set `export SVT_LOG=1` in the terminal to suppress them except in case of errors.
- **Parallel Conversion**: Use `run_parallel_conversion.sh` to speed up the conversion process via parallel processing of multiple episodes.
- **Script Configuration**: For conversion scripts, typically only need to edit `BASE_DIR`, `OUTPUT_DIR`, episode count's bag root folder name, `--dataset-name`, and `--task` parameters.

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
