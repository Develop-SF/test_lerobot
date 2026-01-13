# Quick Reference — Deploy

Run ROS2 inference nodes to control the robot.

## Absolute (ONNX)

```bash
python dev/inference/testing_abs/inference_node_onnx.py \
    --checkpoint /path/to/run_abs \
    --onnx-dir /path/to/run_abs/onnx_models \
    --frequency 20.0
```

## Relative (ONNX)

```bash
python dev/inference/testing_rel/inference_node_onnx_rel.py \
    --checkpoint /path/to/run_rel \
    --onnx-dir /path/to/run_rel/onnx_models \
    --frequency 20.0
```

**Real example:**
```bash
python dev/inference/testing_rel/inference_node_onnx_rel.py \
    --checkpoint /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/checkpoints/last \
    --onnx-dir /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/onnx_models \
    --frequency 20.0
```

## Monitor & Debug

```bash
ros2 topic hz /sync/emily01/left_arm/color/image_raw/compressed
ros2 topic hz /sync/emily01/head/color/image_raw/compressed
ros2 topic hz /left_arm/joint_trajectory
watch -n 1 nvidia-smi
```

## Troubleshooting
- CUDA OOM: use `--device cpu` or lower `--frequency`.
- Image mismatch: verify crop/resize matches training pipeline.
- Robot not moving: check topic names, controller subscription, non-zero outputs.

## Safety Checklist
- Test in simulation first and start at 5–10 Hz.
- Monitor behavior and keep an emergency stop ready.
- Verify action limits and topic rates.

## Useful Commands

```bash
git diff main...testing --stat
python -c "import onnxruntime; print(onnxruntime.get_device())"
```
