# Quick Reference — Export to ONNX

Export trained models for optimized inference.

## Absolute Model (supports 7 DoF)

```bash
cd dev/inference/testing_abs
python3 convert_to_onnx_7DoF.py \
    --checkpoint /path/to/run_abs/checkpoints/last \
    --output /path/to/run_abs/onnx
```

> **Note:** Use `convert_to_onnx_7DoF.py` for 7-DoF models (6 arm + 1 gripper). It handles the interleaved gripper at index 5 and specific image shapes correctly.

## TensorRT Setup Requirement

To use TensorRT-accelerated inference, you must set the `LD_LIBRARY_PATH` (e.g., in your `~/.bashrc`):

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

**Real example (Absolute 7 DoF):**
```bash
cd dev/inference/testing_abs
python3 convert_to_onnx_7DoF.py \
    --checkpoint /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal/models/picknplace_7dof_diffusion/baseline/checkpoints/last \
    --output /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal/models/picknplace_7dof_diffusion/baseline/onnx
```

## Relative Model

```bash
cd dev/inference/testing_rel
python convert_to_onnx_rel.py \
    --checkpoint /path/to/run_rel/checkpoints/best \
    --output-dir /path/to/run_rel/onnx_models \
    --arm-dim 6 \
    --opset-version 17
```

**Real example:**
```bash
cd dev/inference/testing_rel
python convert_to_onnx_rel.py \
    --checkpoint /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/checkpoints/last \
    --output-dir /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/onnx_models \
    --arm-dim 6 \
    --opset-version 17
```

## Verify ONNX (optional)

```bash
python -c "import onnxruntime as ort; print(ort.get_device())"
```
