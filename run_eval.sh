#!/bin/bash
# Wrapper script to run evaluation with correct OpenSSL libraries

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Set LD_LIBRARY_PATH to use conda's OpenSSL
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run the evaluation script with all passed arguments
python dev/inference/testing_rel/evaluate_predictions_onnx_rel.py "$@"
