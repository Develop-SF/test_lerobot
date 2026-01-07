# Quick Reference — Evaluate

Compare predictions against ground truth.

## Absolute

```bash
python dev/inference/testing_abs/evaluate_predictions_onnx.py \
    --checkpoint /path/to/run_abs \
    --rosbag /path/to/test.bag \
    --onnx-dir /path/to/run_abs/onnx_models \
    --plot
```

## Relative

```bash
python dev/inference/testing_rel/evaluate_predictions_onnx_rel.py \
    --checkpoint /path/to/run_rel \
    --rosbag /path/to/test.bag \
    --onnx-dir /path/to/run_rel/onnx_models \
    --plot
```

**Real example:**
```bash
python dev/inference/testing_rel/evaluate_predictions_onnx_rel.py \
    --checkpoint /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/checkpoints/last \
    --rosbag /mnt/SF-Shared/rosbags/20251222_eric_plating_v2/eric_plating_v2_2025_12_22-20_22_05/ \
    --onnx-dir /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/onnx_models \
    --plot
```

## Tips
- Ensure the test rosbag matches training preprocessing.
- Use `--plot` to generate visual comparisons.
