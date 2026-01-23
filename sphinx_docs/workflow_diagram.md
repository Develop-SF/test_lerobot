# Testing Branch Workflow Diagram

Tip: Use {doc}`quick_reference` for the exact commands. This page focuses on visuals and system flow.

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TESTING BRANCH PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│   1. DATA COLLECTION     │
│  ──────────────────────  │
│  🤖 Record ROS2 Bags     │
│  • Camera streams        │
│  • Joint states          │
│  • Commands              │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│   2. CONVERT TO LEROBOT DATASET                                  │
│  ─────────────────────────────────────────────────────────────── │
│                                                                   │
│  📂 dev/data_processing/conversion/                              │
│                                                                   │
│  Option A: Standard                                              │
│  python rosbag_to_lerobot_rosbag2.py                            │
│    --input-mode vision_pos                                       │
│    --output-mode pos_vel                                         │
│                                                                   │
│  Option B: Cropped/Trimmed                                       │
│  python rosbag_to_lerobot_cropped_trimmed.py                    │
│    (with custom preprocessing)                                   │
│                                                                   │
│  Output: LeRobot Dataset (.safetensors + metadata)              │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│   3. TRAIN MODEL                                                 │
│  ─────────────────────────────────────────────────────────────── │
│                                                                   │
│  📚 lerobot/scripts/                                             │
│                                                                   │
│  Option A: Absolute Actions                                      │
│  python train.py                                                 │
│    --config-path config.json                                     │
│    --output-dir /path/to/output                                  │
│                                                                   │
│  Option B: Relative Actions (Recommended)                        │
│  python train_with_relative_actions.py                           │
│    --use_relative_actions                                        │
│    --arm_dim 6                                                   │
│    --obs_horizon 2                                               │
│                                                                   │
│  Output: PyTorch Checkpoint (.safetensors)                       │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│   4. CONVERT TO ONNX                                             │
│  ─────────────────────────────────────────────────────────────── │
│                                                                   │
│  🔄 dev/inference/testing_abs/ OR testing_rel/                   │
│                                                                   │
│  python convert_to_onnx.py                                       │
│    --checkpoint /path/to/checkpoint                              │
│    --output-dir ./onnx_models                                    │
│                                                                   │
│  Output:                                                         │
│  • vision_encoder.onnx                                           │
│  • noise_pred_net.onnx                                           │
│  • config.json                                                   │
│  • normalization_stats.json                                      │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│   5. EVALUATE (Optional but Recommended)                         │
│  ─────────────────────────────────────────────────────────────── │
│                                                                   │
│  📊 Test predictions against ground truth                        │
│                                                                   │
│  python evaluate_predictions_onnx.py                             │
│    --checkpoint /path/to/checkpoint                              │
│    --rosbag /path/to/test.bag                                    │
│    --onnx-dir ./onnx_models                                      │
│    --plot                                                        │
│                                                                   │
│  Output: Prediction plots, metrics, analysis                     │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│   6. DEPLOY WITH ROS2                                            │
│  ─────────────────────────────────────────────────────────────── │
│                                                                   │
│  🚀 dev/inference/testing_abs/ OR testing_rel/                   │
│                                                                   │
│  python inference_node_onnx.py                                   │
│    --checkpoint /path/to/checkpoint                              │
│    --onnx-dir ./onnx_models                                      │
│    --device cuda                                                 │
│    --frequency 20.0                                              │
│                                                                   │
│  ROS2 Topics:                                                    │
│  Subscribe: /sync/.../image_raw/compressed                       │
│  Subscribe: /sync/joint_states                                   │
│  Publish:   /left_arm/joint_trajectory                           │
└────────────┬─────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│   7. ROBOT CONTROL       │
│  ──────────────────────  │
│  🤖 Real Robot           │
│  • Receives commands     │
│  • Executes actions      │
│  • Provides feedback     │
└──────────────────────────┘
```

---

## Data Format Flow

```
┌─────────────────────┐
│   ROS2 Messages     │
│  ─────────────────  │
│  CompressedImage    │
│  JointState         │
│  JointTrajectory    │
└──────────┬──────────┘
           │
           ▼ CONVERSION
┌─────────────────────┐
│  LeRobot Dataset    │
│  ─────────────────  │
│  .safetensors       │
│  • observations     │
│  • actions          │
│  • metadata         │
└──────────┬──────────┘
           │
           ▼ TRAINING
┌─────────────────────┐
│  PyTorch Model      │
│  ─────────────────  │
│  • Vision Encoder   │
│  • Diffusion UNet   │
│  • Checkpoints      │
└──────────┬──────────┘
           │
           ▼ EXPORT
┌─────────────────────┐
│   ONNX Models       │
│  ─────────────────  │
│  • vision_encoder   │
│  • noise_pred_net   │
│  + TensorRT opt.    │
└──────────┬──────────┘
           │
           ▼ INFERENCE
┌─────────────────────┐
│  Robot Commands     │
│  ─────────────────  │
│  JointTrajectory    │
│  → Robot Control    │
└─────────────────────┘
```

---

## Image Preprocessing Pipeline

```
┌───────────────────────────────────────────────────────────┐
│                   CAMERA IMAGES                           │
├──────────────────────────┬────────────────────────────────┤
│   Top View (Head Cam)    │    Left Arm Camera            │
│   640×480 RGB            │    640×480 RGB                 │
└──────────┬───────────────┴────────────┬───────────────────┘
           │                            │
           ▼                            ▼
     ┌──────────┐                 ┌──────────┐
     │   CROP   │                 │  RESIZE  │
     │ 260,135  │                 │ 224×178  │
     │ 178×224  │                 └────┬─────┘
     └────┬─────┘                      │
          │                            │
          ▼                            │
     ┌──────────┐                      │
     │ ROTATE   │                      │
     │  90° CW  │                      │
     │ 224×178  │                      │
     └────┬─────┘                      │
          │                            │
          └────────────┬───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   NORMALIZE    │
              │   [0, 1]       │
              │   3×224×178    │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  MODEL INPUT   │
              │   Batch×3×     │
              │   224×178      │
              └────────────────┘
```

---

## Action Processing Flow

### Absolute Action Mode
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Model Output │ ──> │ Denormalize  │ ──> │ Joint Cmds   │
│  [-1, 1]     │     │   Stats      │     │   Absolute   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Relative Action Mode
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Model Output │ ──> │ Denormalize  │ ──> │   Add Δ to   │
│  Δ [-1, 1]   │     │   Stats      │     │ Current Pos  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │ Joint Cmds   │
                                          │   Absolute   │
                                          └──────────────┘
```

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        ROS2 ECOSYSTEM                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐        ┌──────────────┐                    │
│  │   Camera     │ ────>  │ Image Topics │                    │
│  │   Drivers    │        │  Compressed  │                    │
│  └──────────────┘        └──────┬───────┘                    │
│                                  │                            │
│  ┌──────────────┐        ┌──────┴───────┐                    │
│  │   Robot      │ ────>  │ Joint State  │                    │
│  │  Controller  │        │    Topics    │                    │
│  └──────┬───────┘        └──────┬───────┘                    │
│         │                       │                            │
│         │                       ▼                            │
│         │          ┌─────────────────────────┐              │
│         │          │  Inference Node         │              │
│         │          │  ─────────────────────  │              │
│         │          │  • Image preprocessing  │              │
│         │          │  • ONNX inference       │              │
│         │          │  • Action postprocess   │              │
│         │          └───────────┬─────────────┘              │
│         │                      │                            │
│         │                      ▼                            │
│         │          ┌─────────────────────────┐              │
│         └──────────│  Action Publisher       │              │
│                    │  /left_arm/joint_traj   │              │
│                    └─────────────────────────┘              │
│                                                                │
└────────────────────────────────────────────────────────────────┘

                              ▼

┌────────────────────────────────────────────────────────────────┐
│                     PHYSICAL ROBOT                             │
├────────────────────────────────────────────────────────────────┤
│  • Receives trajectory commands                                │
│  • Executes motion                                             │
│  • Returns state feedback                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimization Levels

```
Level 1: PyTorch CPU
├─ Slowest
├─ No dependencies
└─ Good for testing

Level 2: PyTorch CUDA
├─ Fast
├─ Requires CUDA
└─ Good for development

Level 3: ONNX CPU
├─ Faster than PyTorch
├─ Optimized operators
└─ Production ready

Level 4: ONNX CUDA
├─ Very fast
├─ CUDA optimizations
└─ Recommended

Level 5: ONNX + TensorRT
├─ Fastest
├─ Maximum optimization
└─ Best for real-time
```

---

## Key Branch Differences

```
MAIN BRANCH                    TESTING BRANCH
────────────                   ──────────────
Basic conversion     ───────>  Multi-mode conversion
                               • vision_only
                               • vision_pos
                               • vision_pos_vel

Absolute actions     ───────>  Absolute + Relative
only                           • UMI-inspired
                               • Better generalization

PyTorch inference    ───────>  PyTorch + ONNX + TensorRT
only                           • 2-5x faster
                               • Production ready

Basic ROS            ───────>  Advanced ROS2 nodes
integration                    • Continuous/triggered modes
                               • Error handling
                               • Monitoring

Limited testing      ───────>  Comprehensive testing
                               • Unit tests
                               • Integration tests
                               • Evaluation scripts
```

---

For detailed commands and configuration, see:
- {doc}`testing_branch`
- {doc}`quick_reference`
