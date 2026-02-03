# Quick Reference — Tips & Troubleshooting

Use this page for runtime tips and quick fixes for common issues.

## Runtime Tips
- **Suppress Verbose Logs**: Set `export SVT_LOG=1` to suppress distracting info logs.
- **Parallel Conversion**: Use `run_parallel_conversion.sh` for speed.
- **Overwrite Existing Dataset**: Use the `--force` flag with conversion scripts (e.g., `rosbag_to_lerobot_rosbag2.py --force`) to overwrite an existing output directory. Use this with caution as it overwrites the previous content.
- **Script Config**: Edit simpler parameters (`BASE_DIR`, `--dataset-name`) in conversion scripts rather than deep logic.

## Known Issues & Fixes
- **`lerobot_dataset.py` Torch Input Error**:
  - *Issue*: `torch.stack` fails on lists of floats (e.g., timestamps).
  - *Fix*: Code modified to use `torch.tensor()` which accepts lists, or robustly check types before stacking.
- **`libtorchcodec` Loading Errors**:
  - *Issue*: `RuntimeError` due to missing `libavutil.so` or symbol mismatch.
  - *Fix*: See "FFmpeg & TorchCodec" setup in [Setup](quick_reference_setup.md).
- **OpenSSL Version Mismatch**:
  - *Issue*: `ImportError: /lib/x86_64-linux-gnu/libcrypto.so.3: version 'OPENSSL_3.3.0' not found` (occurs when importing `datasets` or `pyarrow`)
  - *Fix*: Reinstall `pyarrow` and `datasets` from conda-forge (not pip): `conda install -c conda-forge pyarrow datasets`. This ensures all dependencies use compatible system libraries natively.
- **ONNX Module Not Found**:
  - *Issue*: `ModuleNotFoundError: No module named 'onnx'` when converting to ONNX.
  - *Fix*: Install ONNX packages: `pip install onnx onnxruntime-gpu`
- **OpenCV Image Decoding & libjpeg Symbol Mismatch**:
  - *Issue*: `ImportError: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0` when importing cv2, or cv2.imdecode() fails with "buf is not a numpy array" error.
  - *Root Cause*: OpenCV installed from pip (`~/.local/lib`) conflicts with conda environment dependencies. Libjpeg/libtiff version mismatches cause symbol resolution failures.
  - *Fix*: Remove all pip-installed opencv versions and reinstall from conda-forge:
    ```bash
    pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
    conda install -c conda-forge opencv libjpeg-turbo libtiff -y
    ```
  - This ensures all image codec dependencies (libjpeg, libtiff, libpng) are consistent within the conda environment.

## Docker Training Optimization
### Slow Training/Dataloading in Docker
- **Problem**: `train/dataloading_s` is high (e.g., > 0.1s) and GPU utilization drops to 0% intermittently.
- **Verification**: Run `nvidia-smi dmon -s u`.
  - ✅ **Good**: `sm` (Streaming Multiprocessor) stays high (90-100%), `dec` is 0% (CPU decoding is fast enough).
  - ❌ **Bad**: `sm` fluctuates (100% $\to$ 0%), indicating the CPU is still the bottleneck. Check the CPU throttling in Docker stats `cat /sys/fs/cgroup/cpu.stat | grep throttled_usec` to see if this value increases over time radpidly. If so, the cpu is being throttled due to insufficient allocation.
- **Root Cause**: Misunderstanding Docker CPU limits. The `--cpus N` flag limits the container to $N$ logical threads, not physical cores. If `num_workers` is too high relative to $N$, the CPU cannot decode video fast enough (AVX-512 workload), causing GPU starvation.
- **Solution (Formula)**: Allocate enough Logical Threads to cover all workers plus system overhead.
  - **Docker CPUs** $\approx$ `num_workers + 2`
- **Recommended Configuration**:
  - **Docker Limit**: Set `--cpus 10` (or higher) to provide enough raw power for `num_workers: 8`.
  - **Shared Memory**: Use `--shm-size=16g` to prevent DataLoader crashes (Bus Error).


## 7-DoF Training & Conversion Troubleshooting
- **Dataset Timestamp Validation Error**:
  - *Issue*: `ValueError: One or several timestamps unexpectedly violate the tolerance`
  - *Fix*: Increase tolerance in the conversion command (e.g., `--tolerance 2.0`).
- **Out of Memory (OOM) during Training**:
  - *Issue*: GPU memory exceeded.
  - *Fix*: Reduce `batch_size` (e.g., to 32 or 16) and `num_workers` (e.g., to 2) in the config.
- **Model Not Learning**:
  - *Issue*: Loss not decreasing.
  - *Fix*: Check learning rate (try 1e-4 range), verify data normalization, and ensure `observation.state` and `action` shapes match your 7-DoF setup.

## Verified Package Versions
- **FFmpeg**: 7.0 or higher (from conda-forge)
- **TorchCodec**: 0.2.1
- **PyTorch**: 2.6.0
- **PyArrow**: Latest (from conda-forge for native SSL compatibility)
- **Datasets**: Latest (from conda-forge for native SSL compatibility)
- **ONNX**: Latest (auto-installed with pip)
- **ONNX Runtime**: Latest GPU version (or CPU version for non-GPU systems)
- **OpenSSL**: Latest from conda-forge

## Quick Install Script
```bash
# All-in-one setup after LeRobot base installation
conda activate lerobot
conda install -y -c conda-forge 'ffmpeg>=7.0' openssl pyarrow datasets
pip install torchcodec==0.2.1 onnx onnxruntime-gpu
```