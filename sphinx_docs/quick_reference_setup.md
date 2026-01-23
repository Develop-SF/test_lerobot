# Quick Reference — Setup

Use this page to get the environment ready fast.

## Install Basics
- Install LeRobot locally:
  ```bash
  conda create -y -n lerobot python=3.10
  conda activate lerobot
  conda install ffmpeg -c conda-forge
  pip install -e .
  ```
  See the main project README for more details.


<!-- ## Additional Packages
```bash
pip install numpy==1.26.4 matplotlib
pip uninstall -y pynvml
pip install nvidia-ml-py
pip install poetry-core
pip uninstall pyarrow -y
pip install pyarrow
``` -->

## ONNX Runtime

```bash
# GPU
pip install onnx onnxruntime-gpu

# CPU-only
pip install onnx onnxruntime
```

## Tips
- Reduce verbose logs: `export SVT_LOG=1`.
- Prefer editing script parameters (e.g., `--dataset-name`) over deep code changes.

## Known Issues (Quick Fixes)
- ONNX not found: `pip install onnx onnxruntime[-gpu]`.
- OpenSSL mismatch: `conda install -c conda-forge openssl`.
