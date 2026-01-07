# Quick Reference — Tips & Troubleshooting

Use this page for runtime tips and quick fixes for common issues.

## Runtime Tips
- **Suppress Verbose Logs**: Set `export SVT_LOG=1` to suppress distracting info logs.
- **Parallel Conversion**: Use `run_parallel_conversion.sh` for speed.
- **Script Config**: Edit simpler parameters (`BASE_DIR`, `--dataset-name`) in conversion scripts rather than deep logic.

## Known Issues & Fixes
- **`lerobot_dataset.py` Torch Input Error**:
  - *Issue*: `torch.stack` fails on lists of floats (e.g., timestamps).
  - *Fix*: Code modified to use `torch.tensor()` which accepts lists, or robustly check types before stacking.
- **`libtorchcodec` Loading Errors**:
  - *Issue*: `RuntimeError` due to missing `libavutil.so` or symbol mismatch.
  - *Fix*: See "FFmpeg & TorchCodec" setup in [Setup](quick_reference_setup.md).
- **OpenSSL Version Mismatch**:
  - *Issue*: `ImportError: /lib/x86_64-linux-gnu/libcrypto.so.3: version 'OPENSSL_3.3.0' not found`
  - *Fix*: Install OpenSSL from conda-forge: `conda install -c conda-forge openssl`
- **ONNX Module Not Found**:
  - *Issue*: `ModuleNotFoundError: No module named 'onnx'` when converting to ONNX.
  - *Fix*: Install ONNX packages: `pip install onnx onnxruntime-gpu`

## Verified Package Versions
- **FFmpeg**: 7.0 or higher (from conda-forge)
- **TorchCodec**: 0.2.1
- **PyTorch**: 2.6.0
- **ONNX**: Latest (auto-installed with pip)
- **ONNX Runtime**: Latest GPU version (or CPU version for non-GPU systems)
- **OpenSSL**: Latest from conda-forge (fixes SSL import errors)

## Quick Install Script
```bash
# All-in-one setup after LeRobot base installation
conda activate lerobot
conda install -y -c conda-forge 'ffmpeg>=7.0' openssl
pip install torchcodec==0.2.1 onnx onnxruntime-gpu
```