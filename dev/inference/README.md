# LeRobot Inference Development

This directory contains development code for LeRobot inference, split into two main subdirectories based on the action representation used.

## Directory Structure

### 1. `testing_abs` (Absolute Actions)
Contains inference code for models trained with **absolute action representation**.
- Uses standard position/velocity control.
- Includes ROS2 nodes and evaluation scripts.
- See [testing_abs/README.md](testing_abs/README.md) for details.

### 2. `testing_rel` (Relative Actions)
Contains inference code for models trained with **relative action representation**.
- Uses relative changes in position/velocity.
- Includes specialized ONNX+TensorRT inference nodes.
- See [testing_rel/README.md](testing_rel/README.md) for details.

## Common Scripts

- `test_rosbag.py`: Utility to test inference pipelines using recorded rosbag data.
- `test_inference.py`: Unit tests for the inference module and image preprocessing.
