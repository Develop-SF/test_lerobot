# Quick Reference — Evaluate

Compare predictions against ground truth.

## Absolute (supports 7 DoF)

For high-performance evaluation with TensorRT accelerator:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
python3 dev/inference/testing_abs/evaluate_predictions_onnx_7DoF.py \
    --checkpoint /path/to/run_abs \
    --onnx-dir /path/to/run_abs/onnx_models \
    --rosbag /path/to/test.bag \
    --num-samples 248 \
    --plot
```

**Key Features of `evaluate_predictions_onnx_7DoF.py`:**
- **Proper 7-DoF Support:** Explicitly handles 6 arm joints + 1 gripper in the order expected by the 7-DoF dataset.
- **Interleaved Gripper:** Handles gripper position at index 5 and wrist_3 at index 6.
- **Angle Wraparound:** Automatically handles revolute joint wraparound (e.g., ±π) for `ra_wrist_3_joint`.
- **MCAP & front camera:** Defaulted to Front camera and MCAP storage for the 7-DoF system.


**Real example (Absolute 7 DoF):**
```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
python3 dev/inference/testing_abs/evaluate_predictions_onnx_7DoF.py \
    --checkpoint /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal/models/picknplace_7dof_diffusion/baseline/checkpoints/last \
    --onnx-dir /mnt/nas/models/picknplace_7dof_diffusion/baseline/onnx \
    --rosbag /mnt/nas/rosbags/20260109_picknplace/normal/picknplace_2026_01_09-14_10_48/ \
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
    --checkpoint /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/checkpoints/last \
    --rosbag /mnt/nas/rosbags/20251222_eric_plating_v2/eric_plating_v2_2025_12_22-20_22_05/ \
    --onnx-dir /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/onnx_models \
    --plot
```

## Tips
- Ensure the test rosbag matches training preprocessing.
- Use `--plot` to generate visual comparisons.
