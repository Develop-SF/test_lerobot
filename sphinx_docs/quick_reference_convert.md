# Quick Reference — Convert Data

Convert ROS bags to LeRobot datasets.

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

**Real example:**
```bash
cd dev/data_processing/conversion
python rosbag_to_lerobot_rosbag2.py \
    /mnt/SF-Shared/rosbags/20251222_eric_plating_v2/ \
    --output-dir /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2 \
    --dataset-name eric_plating_v2 \
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
python inspect_dataset.py /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2
```

## Input/Output Modes

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
