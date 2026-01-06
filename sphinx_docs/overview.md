# 📚 Testing Branch Documentation Index

Welcome to the comprehensive documentation for the LeRobot Testing Branch!

---

## 🚀 Quick Start

**New to this branch?** Start here:
1. Read the [Branch Changes Summary](#branch-changes-summary) to understand what's new
2. Follow the [Complete Workflow Guide](#workflow-guide) for step-by-step instructions
3. Use the [Quick Reference](#quick-reference) for command-line cheat sheet
4. Check the [Workflow Diagrams](#workflow-diagrams) for visual understanding

---

## 📖 Documentation Files

### 1. Branch Changes Summary
**File:** [BRANCH_CHANGES_SUMMARY.md](BRANCH_CHANGES_SUMMARY.md)

**What's inside:**
- Complete list of changes (38 files, 14,110+ lines)
- Commit history
- Feature comparison table
- Performance improvements
- Migration guide from main branch

**Read this if:**
- You want to understand what's different from main
- You need to see the complete feature list
- You're migrating from main branch

---

### 2. Testing Branch Workflow Guide
**File:** [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md)

**What's inside:**
- Complete end-to-end workflow (ROS bag → Dataset → Training → ONNX → Inference)
- Detailed instructions for each step
- Configuration examples
- Troubleshooting guide
- Safety considerations
- Performance optimization tips

**Read this if:**
- You're setting up the complete pipeline
- You need detailed explanations
- You're encountering issues
- You want to understand best practices

---

### 3. Quick Reference
**File:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**What's inside:**
- One-liner commands for each step
- Input/output modes table
- ROS2 topics reference
- Troubleshooting quick fixes
- Complete pipeline example
- Safety checklist

**Read this if:**
- You need quick command references
- You know what to do but forgot the syntax
- You want a command cheat sheet

---

### 4. Workflow Diagrams
**File:** [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)

**What's inside:**
- Visual pipeline flow
- Data format transformations
- Image preprocessing pipeline
- Action processing flow
- System architecture
- Performance optimization levels
- Branch differences diagram

**Read this if:**
- You're a visual learner
- You want to understand data flow
- You need architecture overview
- You're presenting to others

---

## 📂 Component-Specific Documentation

### Data Processing & Conversion

**Location:** `dev/data_processing/conversion/`

**Key Files:**
- `rosbag_to_lerobot_rosbag2.py` - Standard converter
- `rosbag_to_lerobot_cropped_trimmed.py` - Advanced converter
- `run_approach_new_converter.sh` - Shell script example

**What it does:**
- Converts ROS2 bags to LeRobot datasets
- Supports multiple input/output modes
- Custom image preprocessing
- Episode trimming

**Quick command:**
```bash
python rosbag_to_lerobot_rosbag2.py \
    /path/to/rosbags \
    --output-dir /path/to/dataset \
    --dataset-name "my_dataset" \
    --fps 20 \
    --input-mode vision_pos \
    --output-mode pos_vel
```

---

### Training

**Location:** `lerobot/scripts/`

**Key Files:**
- `train.py` - Standard training (absolute actions)
- `train_with_relative_actions.py` - Relative action training
- `compute_relative_action_stats.py` - Statistics computation

**What it does:**
- Train models with absolute or relative actions
- Flexible input/output configurations
- Pre-computed statistics support

**Quick commands:**
```bash
# Absolute actions
python lerobot/scripts/train.py \
    --config-path config.json \
    --output-dir /path/to/output

# Relative actions
python lerobot/scripts/train_with_relative_actions.py \
    --config_path config.json \
    --output_dir /path/to/output \
    --use_relative_actions \
    --arm_dim 6
```

---

### Inference - Absolute Actions

**Location:** `dev/inference/testing_abs/`

**Documentation:**
- [README.md](dev/inference/testing_abs/README.md) - Overview
- [DEPLOYMENT_GUIDE.md](dev/inference/testing_abs/DEPLOYMENT_GUIDE.md) - Deployment steps
- [TESTING_GUIDE.md](dev/inference/testing_abs/TESTING_GUIDE.md) - Testing procedures

**Key Files:**
- `convert_to_onnx.py` - ONNX conversion
- `inference_node_onnx.py` - ONNX inference node
- `evaluate_predictions_onnx.py` - Evaluation

**What it does:**
- Convert PyTorch models to ONNX
- Real-time ROS2 inference
- Performance evaluation

**Quick commands:**
```bash
# Convert to ONNX
python convert_to_onnx.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models

# Run inference
python inference_node_onnx.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

---

### Inference - Relative Actions

**Location:** `dev/inference/testing_rel/`

**Documentation:**
- [README.md](dev/inference/testing_rel/README.md) - Overview

**Key Files:**
- `convert_to_onnx_rel.py` - ONNX conversion with relative actions
- `inference_node_onnx_rel.py` - ONNX inference node
- `evaluate_predictions_onnx_rel.py` - Evaluation

**What it does:**
- Convert relative action models to ONNX
- Real-time inference with action deltas
- State accumulation and prediction

**Quick commands:**
```bash
# Convert to ONNX
python convert_to_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --output-dir ./onnx_models \
    --use-relative-actions \
    --arm-dim 6

# Run inference
python inference_node_onnx_rel.py \
    --checkpoint /path/to/checkpoint \
    --onnx-dir ./onnx_models \
    --device cuda \
    --frequency 20.0
```

---

## 🎯 Common Workflows

### Workflow 1: Complete Pipeline (Absolute Actions)

```bash
# 1. Convert ROS bag
cd dev/data_processing/conversion
python rosbag_to_lerobot_rosbag2.py /data/bags --output-dir /data/dataset --dataset-name exp1 --fps 20

# 2. Train model
cd /home/shinfang-ovx/workspaces/test_lerobot
python lerobot/scripts/train.py --config-path config.json --output-dir /data/model

# 3. Convert to ONNX
cd dev/inference/testing_abs
python convert_to_onnx.py --checkpoint /data/model/checkpoints/best --output-dir /data/model/onnx

# 4. Evaluate
python evaluate_predictions_onnx.py --checkpoint /data/model/checkpoints/best --rosbag /data/test.bag --onnx-dir /data/model/onnx

# 5. Deploy
python inference_node_onnx.py --checkpoint /data/model/checkpoints/best --onnx-dir /data/model/onnx --frequency 20.0
```

### Workflow 2: Complete Pipeline (Relative Actions)

```bash
# 1. Convert ROS bag
cd dev/data_processing/conversion
python rosbag_to_lerobot_rosbag2.py /data/bags --output-dir /data/dataset --dataset-name exp1 --fps 20

# 2. Compute statistics (optional)
cd /home/shinfang-ovx/workspaces/test_lerobot
python lerobot/scripts/compute_relative_action_stats.py --dataset-path /data/dataset --arm-dim 6 --output-path /data/stats.json

# 3. Train with relative actions
python lerobot/scripts/train_with_relative_actions.py --config_path config.json --output_dir /data/model --use_relative_actions --arm_dim 6 --relative_stats_path /data/stats.json

# 4. Convert to ONNX
cd dev/inference/testing_rel
python convert_to_onnx_rel.py --checkpoint /data/model/checkpoints/best --output-dir /data/model/onnx --use-relative-actions --arm-dim 6

# 5. Evaluate
python evaluate_predictions_onnx_rel.py --checkpoint /data/model/checkpoints/best --rosbag /data/test.bag --onnx-dir /data/model/onnx

# 6. Deploy
python inference_node_onnx_rel.py --checkpoint /data/model/checkpoints/best --onnx-dir /data/model/onnx --frequency 20.0
```

---

## 🔍 Finding Information

### By Task

| Task | Document | Section |
|------|----------|---------|
| Understanding changes | [BRANCH_CHANGES_SUMMARY.md](BRANCH_CHANGES_SUMMARY.md) | Feature Comparison |
| Converting ROS bags | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) | Section 1 |
| Training models | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) | Section 2 |
| Converting to ONNX | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) | Section 3 |
| Running inference | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) | Section 4 |
| Quick commands | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | All sections |
| Visual diagrams | [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) | All sections |
| Deployment | [dev/inference/testing_abs/DEPLOYMENT_GUIDE.md](dev/inference/testing_abs/DEPLOYMENT_GUIDE.md) | All sections |
| Troubleshooting | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) | Section 5 |

### By Question

| Question | Answer Location |
|----------|----------------|
| What's new in this branch? | [BRANCH_CHANGES_SUMMARY.md](BRANCH_CHANGES_SUMMARY.md) |
| How do I convert ROS bags? | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) § 1 |
| What are input/output modes? | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § Input/Output Modes |
| How do I train with relative actions? | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) § 2B |
| How do I convert to ONNX? | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) § 3 |
| How do I deploy on a robot? | [dev/inference/testing_abs/DEPLOYMENT_GUIDE.md](dev/inference/testing_abs/DEPLOYMENT_GUIDE.md) |
| What ROS topics are used? | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) § ROS2 Topics |
| How do I troubleshoot issues? | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) § 5 |
| How do I optimize performance? | [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md) § 6 |
| What's the architecture? | [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) § System Architecture |

---

## 📊 Key Features at a Glance

### ✅ Data Processing
- ✅ Multi-mode ROS bag conversion
- ✅ Custom image preprocessing
- ✅ Episode trimming
- ✅ Visualization tools

### ✅ Training
- ✅ Absolute action training
- ✅ Relative action training (UMI-inspired)
- ✅ Flexible input/output modes
- ✅ Pre-computed statistics

### ✅ Inference
- ✅ PyTorch inference
- ✅ ONNX inference (3-5x faster)
- ✅ TensorRT optimization (5-10x faster)
- ✅ Real-time ROS2 integration

### ✅ Evaluation
- ✅ Prediction vs ground truth
- ✅ Performance metrics
- ✅ Visualization plots
- ✅ Timing analysis

### ✅ Testing
- ✅ Unit tests
- ✅ Integration tests
- ✅ Rosbag testing
- ✅ Image preprocessing validation

---

## 🛠️ Tools & Utilities

| Tool | Location | Purpose |
|------|----------|---------|
| Dataset inspector | `inspect_dataset.py` | Inspect LeRobot datasets |
| Dataset loader | `load_local_dataset.py` | Load local datasets |
| ROS bag downsampler | `dev/data_processing/conversion/downsample_rosbag.py` | Reduce frame rate |
| Video visualizer | `dev/data_processing/visualization/*.py` | Visualize data |
| Test inference | `dev/inference/testing_abs/test_inference.py` | Test pipeline |
| Test rosbag | `dev/inference/testing_abs/test_rosbag.py` | Test with data |

---

## 📈 Performance Guide

| Backend | Relative Speed | Use Case |
|---------|---------------|----------|
| PyTorch CPU | 1x (baseline) | Testing only |
| PyTorch CUDA | 2-3x | Development |
| ONNX CPU | 1.5-2x | Production (no GPU) |
| ONNX CUDA | 3-5x | Production |
| ONNX+TensorRT | 5-10x | Real-time control |

**Recommendation:** Use ONNX+TensorRT for real robot deployment.

---

## 🎓 Learning Path

### Beginner
1. Read [BRANCH_CHANGES_SUMMARY.md](BRANCH_CHANGES_SUMMARY.md)
2. View [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)
3. Follow [QUICK_REFERENCE.md](QUICK_REFERENCE.md) examples

### Intermediate
1. Study [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md)
2. Read component-specific documentation
3. Try both absolute and relative action workflows

### Advanced
1. Read [DEPLOYMENT_GUIDE.md](dev/inference/testing_abs/DEPLOYMENT_GUIDE.md)
2. Optimize performance with TensorRT
3. Customize preprocessing pipeline

---

## 💡 Tips & Best Practices

1. **Always test with rosbag data before deploying on real robot**
2. **Start with low inference frequency (5-10 Hz) and increase gradually**
3. **Use relative actions for better generalization**
4. **Convert to ONNX for production deployment**
5. **Monitor GPU usage and inference timing**
6. **Keep emergency stop accessible**
7. **Verify image preprocessing matches training**
8. **Use continuous mode for predictable control**

---

## 🤝 Support

**Having issues?**
1. Check troubleshooting sections in guides
2. Verify your setup matches examples
3. Test with provided rosbag data first
4. Review ROS topic names and types

**Need clarification?**
1. Start with visual diagrams
2. Read relevant guide sections
3. Try example commands
4. Check component documentation

---

## 📝 Document Versions

| Document | Lines | Last Updated |
|----------|-------|--------------|
| BRANCH_CHANGES_SUMMARY.md | ~400 | 2026-01-05 |
| TESTING_BRANCH_GUIDE.md | ~650 | 2026-01-05 |
| QUICK_REFERENCE.md | ~150 | 2026-01-05 |
| WORKFLOW_DIAGRAM.md | ~400 | 2026-01-05 |
| INDEX.md (this file) | ~400 | 2026-01-05 |

---

## 🎯 Next Steps

**Choose your path:**

**Path 1: Quick Start** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)  
**Path 2: Complete Guide** → [TESTING_BRANCH_GUIDE.md](TESTING_BRANCH_GUIDE.md)  
**Path 3: Visual Learning** → [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)  
**Path 4: Understanding Changes** → [BRANCH_CHANGES_SUMMARY.md](BRANCH_CHANGES_SUMMARY.md)

---

**Happy Robot Learning! 🤖**

---

*Generated: January 5, 2026*  
*Branch: testing*  
*Status: Production Ready*
