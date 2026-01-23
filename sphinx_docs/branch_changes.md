# Testing Branch - Summary of Changes

## Branch Information

**Branch:** `testing`  
**Base:** `main`  
**Commits ahead:** 5  
**Files changed:** 38 files  
**Lines added:** 14,110+  

## Commit History

```
* c9e57b0 - update docs
* 136fa06 - update inference onnx
* 8b33fe1 - update lerobot core
* 987abec - add dev code
* 4674f48 - testing
```

---

## Key Additions

### 1. Data Processing & Conversion (14 files)

#### Conversion Scripts
- **rosbag_to_lerobot_rosbag2.py** (777 lines)
- **rosbag_to_lerobot_cropped_trimmed.py** (873 lines)
- **downsample_rosbag.py** (162 lines)

#### Visualization Tools
- **cropped_trimmed_video_viz.py** (396 lines)
- **extract_frames.py** (279 lines)
- **left_arm_video_extractor.py** (351 lines)
- **optical_flow_video_viz.py** (790 lines)

---

### 2. Relative Action Training (4 files)

#### Core Implementation
- **lerobot/common/datasets/relative_action_dataset.py** (344 lines)
- **lerobot/common/utils/relative_actions.py** (253 lines)

#### Training Scripts
- **lerobot/scripts/train_with_relative_actions.py** (499 lines)
- **lerobot/scripts/compute_relative_action_stats.py** (334 lines)

---

### 3. Enhanced Policy Support (2 files)

#### Diffusion Policy Updates
- **lerobot/common/policies/diffusion/modeling_diffusion.py**
- **lerobot/common/policies/diffusion/configuration_diffusion.py**

---

### 4. Inference & Deployment - Absolute Actions (10 files)

Located in `dev/inference/testing_abs/`

#### Core Inference
- **lerobot_inference.py** (421 lines)
- **inference_node.py** (407 lines)
- **inference_node_onnx.py** (1,079 lines)

#### Model Conversion
- **convert_to_onnx.py** (585 lines)

#### Evaluation & Testing
- **evaluate_predictions_onnx.py** (876 lines)
- **test_rosbag.py** (258 lines)

---

### 5. Inference & Deployment - Relative Actions (5 files)

Located in `dev/inference/testing_rel/`

#### ROS2 Node
- **inference_node_onnx_rel.py** (1,141 lines)

#### Model Conversion
- **convert_to_onnx_rel.py** (615 lines)

#### Evaluation
- **evaluate_predictions_onnx_rel.py** (961 lines)

---

### 6. Utility Scripts (3 files)

- **inspect_dataset.py** (100 lines)

---

## Next Steps

### For New Users:
1. Read {doc}`testing_branch`
2. Follow {doc}`quick_reference`
3. Start with conversion → training → inference

---

## Support & Resources

- **Full Guide:** {doc}`testing_branch`
- **Quick Commands:** {doc}`quick_reference`
- **Diagrams:** {doc}`workflow_diagram`
- **Absolute Actions:** (see `dev/inference/testing_abs` folder)
- **Relative Actions:** (see `dev/inference/testing_rel` folder)

---

**Created:** January 5, 2026  
**Branch:** testing  
**Status:** Active Development  
**Stability:** Production Ready
