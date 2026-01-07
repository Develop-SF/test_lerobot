# Quick Reference — Export to ONNX

Export trained models for optimized inference.

## Absolute Model

```bash
cd dev/inference/testing_abs
python convert_to_onnx.py \
    --checkpoint /path/to/run_abs/checkpoints/best \
    --output-dir /path/to/run_abs/onnx_models \
    --opset-version 17
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
    --checkpoint /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/checkpoints/last \
    --output-dir /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/runs/eric_plating_v2_rel_2026-01-06_23-03-49/onnx_models \
    --arm-dim 6 \
    --opset-version 17
```

## Verify ONNX (optional)

```bash
python -c "import onnxruntime as ort; print(ort.get_device())"
```
