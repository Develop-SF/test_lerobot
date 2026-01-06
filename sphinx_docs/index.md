# LeRobot Testing Branch Documentation

Welcome to the comprehensive documentation for the LeRobot Testing Branch!

```{toctree}
:maxdepth: 2
:caption: Contents:

overview
testing_branch
quick_reference
workflow_diagram
branch_changes
```

## Quick Links

- {doc}`testing_branch` - Complete workflow guide
- {doc}`quick_reference` - Command cheat sheet  
- {doc}`workflow_diagram` - Visual diagrams
- {doc}`branch_changes` - Detailed changes

## Overview

The testing branch contains significant enhancements over the main branch:

- **38 new files** with **14,110+ lines** of code
- ROS bag conversion tools with flexible input/output modes
- Relative action training support (UMI/ManiSkill-inspired)
- ONNX/TensorRT inference nodes (5-10x faster)
- Complete ROS2 integration for real robot control
- Comprehensive testing and evaluation tools

## Quick Start

```bash
# 1. Convert ROS bag
python dev/data_processing/conversion/rosbag_to_lerobot_rosbag2.py \
    /path/to/rosbags --output-dir /path/to/dataset

# 2. Train model
python lerobot/scripts/train_with_relative_actions.py \
    --use_relative_actions --arm_dim 6

# 3. Convert to ONNX
python dev/inference/testing_rel/convert_to_onnx_rel.py \
    --checkpoint /path/to/checkpoint --use-relative-actions

# 4. Deploy
python dev/inference/testing_rel/inference_node_onnx_rel.py \
    --checkpoint /path/to/checkpoint --frequency 20.0
```

## Getting Help

- Check the {doc}`testing_branch` for detailed instructions
- Use the {doc}`quick_reference` for quick command lookup
- Review {doc}`workflow_diagram` for visual understanding
- Read {doc}`branch_changes` for complete change list
