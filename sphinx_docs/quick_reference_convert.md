# Quick Reference — Convert Data

Convert ROS bags to LeRobot datasets.

> **Camera & Arm Configuration:** Standard converter defaults to **left arm camera** (`left_arm_cam`) and **left arm joints** (`la_*`). The front camera is commented out. Use the 7-DoF converter for systems with 6 arm joints + 1 gripper joint.

## Standard Converter

```bash
cd dev/data_processing/conversion
python rosbag_to_lerobot_rosbag2.py \
    /path/to/rosbags \
    --output-dir /path/to/dataset \
    --dataset-name exp1 \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel
```

## 7-DoF Converter (UR10e + Gripper)

Use this specialized converter for 7-DoF systems to handle 6 arm joints + 1 gripper joint.

```bash
cd dev/data_processing/conversion
python3 rosbag_to_lerobot_rosbag2_7DoF.py \
    /path/to/rosbags \
    --output-dir /path/to/dataset \
    --dataset-name picknplace_7dof \
    --task "picknplace_7dof" \
    --input-mode vision_pos \
    --output-mode pos_only
```

### 7-DoF Real Example
```bash
python3 dev/data_processing/conversion/rosbag_to_lerobot_rosbag2_7DoF.py \
    /mnt/nas/rosbags/20260109_picknplace/normal/ \
    --output-dir /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal \
    --dataset-name picknplace_7dof_normal \
    --task "picknplace_7dof_normal" \
    --input-mode vision_pos \
    --output-mode pos_only
```

### 7-DoF Multiple Bags Example
```bash
python3 dev/data_processing/conversion/rosbag_to_lerobot_rosbag2_7DoF.py \
    /mnt/nas/rosbags/20260109_picknplace/normal \
    /mnt/nas/rosbags/20260109_picknplace/tilted \
    /mnt/nas/rosbags/20260109_picknplace/tipover \
    --output-dir /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof \
    --dataset-name picknplace_7dof \
    --task "picknplace_7dof" \
    --input-mode vision_pos \
    --output-mode pos_only
```

## Standard Converter Real Examples
```bash
cd dev/data_processing/conversion
python rosbag_to_lerobot_rosbag2.py \
    /mnt/nas/rosbags/20251222_eric_plating_v2/ \
    --output-dir /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2 \
    --dataset-name eric_plating_v2 \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel

python rosbag_to_lerobot_rosbag2.py \
    /mnt/nas/rosbags/20260109_picknplace/normal/ \
    --output-dir /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof \
    --dataset-name picknplace_6dof \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel
```

## Cropped/Trimmed Converter (optional)

```bash
python rosbag_to_lerobot_cropped_trimmed.py \
    /path/to/rosbags \
    --output-dir /path/to/dataset_cropped \
    --dataset-name exp1_cropped \
    --fps 20
```

## Verify Dataset

```bash
python inspect_dataset.py /path/to/dataset
```

**Real example:**
```bash
python inspect_dataset.py /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2
```

## Review Dataset

### Single Episode (Rerun Viewer)

First, install the required dependency:

```bash
pip install rerun-sdk
```

Then, visualize a single episode:

```bash
python3 -m lerobot.scripts.visualize_dataset --root /path/to/dataset --episode-index 0 --repo-id dataset_name
```

**Real example:**
```bash
python3 -m lerobot.scripts.visualize_dataset --root /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/ --episode-index 0 --repo-id picknplace_6dof
```

### Multiple Episodes (HTML Viewer)

Visualize multiple episodes interactively:

```bash
python3 -m lerobot.scripts.visualize_dataset_html --root /path/to/dataset --repo-id /dataset_name --port 9099
```

**Real example:**
```bash
python3 -m lerobot.scripts.visualize_dataset_html --root /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/ --repo-id /picknplace_6dof --port 9099
```

Then open `http://localhost:9099` in your browser to navigate between episodes.

| Mode | Description | Observation Dim | Action Dim |
|------|-------------|----------------|------------|
| vision_only | Images only | Images | 6 or 12 |
| vision_pos | Images + positions | Images + 6 | 6 or 12 |
| vision_pos_vel | Images + pos + vel | Images + 12 | 12 |
| pos_only | Position outputs | - | 6 |
| pos_vel | Position + velocity | - | 12 |

## ROS2 Topics
- Subscribe: `/sync/emily01/left_arm/color/image_raw/compressed`, `/sync/emily01/head/color/image_raw/compressed`, `/sync/joint_states`
- Publish: `/left_arm/joint_trajectory`
