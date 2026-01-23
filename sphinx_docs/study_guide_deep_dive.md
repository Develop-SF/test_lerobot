# Deep Dive: Diffusion Policy for Robot Learning

## 📚 Paper Background

### Original Diffusion Policy Paper
**Title:** "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"  
**Authors:** Columbia Artificial Intelligence, Robotics Lab  
**Paper:** https://arxiv.org/abs/2303.04137  
**Code:** https://github.com/real-stanford/diffusion_policy  

### Key Insights

#### What is Diffusion Policy?
Diffusion Policy reframes robot visuomotor policy learning as a **conditional denoising diffusion process**. Instead of directly predicting actions from observations, it:

1. **Learns the gradient of the action-distribution score function**
2. **Iteratively optimizes** this gradient field during inference via stochastic Langevin dynamics steps
3. **Refines noise into actions** through a learned gradient field

#### Why Diffusion for Robotics?

The diffusion formulation offers several powerful advantages:

- ✅ **Multimodal Action Distributions**: Gracefully handles scenarios where multiple valid actions exist
- ✅ **High-Dimensional Action Spaces**: Suitable for complex robot manipulation with many degrees of freedom
- ✅ **Training Stability**: More stable training compared to GANs and VAEs
- ✅ **Robustness**: Impressive robustness against perturbations and visual distractions
- ✅ **Performance**: Average improvement of **46.9%** over existing state-of-the-art methods across 12 tasks

#### Technical Contributions

To fully leverage diffusion models for physical robots, the paper introduces:

1. **Receding Horizon Control**: Predicts an action trajectory (horizon) but only executes a subset (n_action_steps)
2. **Visual Conditioning**: Efficient image encoding with ResNet + SpatialSoftmax
3. **Time-Series Diffusion Transformer**: U-Net architecture for denoising across the temporal action dimension

---

## 🚀 Successor Techniques: Flow Matching

### Flow Matching Overview

**Flow Matching** (specifically Conditional Flow Matching - CFM) is emerging as a powerful successor to Diffusion Policy with several key advantages:

#### Advantages over Diffusion Policy

| Feature | Diffusion Policy | Flow Matching |
|---------|-----------------|---------------|
| **Framework** | Stochastic (noisy process) | Deterministic (flow trajectories) |
| **Inference Speed** | Slower (multiple steps) | **7x faster** |
| **Training Stability** | Stable | **More stable** |
| **Hyperparameters** | More tuning required | **Fewer hyperparameters** |
| **Performance** | Strong baseline | **Comparable or better** |
| **Generalization** | Diffusion formulation | **More flexible framework** |

#### Notable Flow Matching Methods

1. **FlowPolicy**: Achieved 7x inference speedup while maintaining competitive success rates
2. **PointFlowMatch**: Uses CFM with point cloud observations, **doubled performance** in some manipulation tasks
3. **Streaming Flow Policy**: Treats action trajectories as flow trajectories for faster, more reactive execution

#### Why Flow Matching Matters

- **Real-time Robotics**: Faster inference is critical for reactive control
- **Simpler Framework**: Better suited for robotics applications
- **Unified Theory**: Both diffusion and flow matching are linked under a unified framework
- **Real-world Success**: Demonstrated on humanoid whole-body control and complex manipulation

### When to Use What?

- **Use Diffusion Policy** if:
  - You have existing infrastructure/codebase (like LeRobot)
  - Training stability is paramount
  - You don't need ultra-fast inference
  
- **Consider Flow Matching** if:
  - Inference speed is critical (real-time control)
  - You want simpler training with fewer hyperparameters
  - You're building a new system from scratch

---

## 🔧 Training Command Arguments Explained

Let's break down your training command argument by argument:

### Example Training Commands

We'll show two common configurations. Most teams use **6-DoF reach tasks**, but we also include 7-DoF for pick-and-place scenarios.

#### 6-DoF Reach Task (Most Common)
```bash
python3 -m [lerobot.scripts.train](../lerobot/scripts/train.py) \
    --dataset.repo_id plating_6dof \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/plating_6dof \
    --policy.type diffusion \
    --policy.horizon 32 \
    --policy.n_action_steps 16 \
    --policy.n_obs_steps 2 \
    --policy.crop_shape "null" \
    --policy.vision_backbone resnet18 \
    --policy.push_to_hub False \
    --policy.device cuda \
    --batch_size 64 \               # Larger batch for 6DoF
    --num_workers 4 \
    --wandb.enable True \
    --wandb.project plating_6dof \
    --wandb.entity shennongshi \
    --wandb.disable_artifact True \
    --job_name plating_diffusion_baseline \
    --output_dir /mnt/nas/models/plating_6dof/diffusion/baseline
```
**Note**: `obs.state` and `action` dimensions are **(6,)** for 6-DoF arms

#### 7-DoF Pick-and-Place (With Gripper)
```bash
python3 -m [lerobot.scripts.train](../lerobot/scripts/train.py) \
    --dataset.repo_id picknplace_7dof_normal \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal \
    --policy.type diffusion \
    --policy.horizon 32 \
    --policy.n_action_steps 16 \
    --policy.n_obs_steps 2 \
    --policy.crop_shape "null" \
    --policy.vision_backbone resnet18 \
    --policy.push_to_hub False \
    --policy.device cuda \
    --batch_size 16 \               # Smaller batch if memory-constrained
    --num_workers 2 \
    --wandb.enable True \
    --wandb.project picknplace_7dof_normal \
    --wandb.entity shennongshi \
    --wandb.disable_artifact True \
    --job_name picknplace_7dof_diffusion_baseline \
    --output_dir /mnt/nas/models/picknplace_7dof_diffusion/baseline
```
**Note**: `obs.state` and `action` dimensions are **(7,)** for 7-DoF arms with gripper

---

### Dataset Arguments

#### `--dataset.repo_id picknplace_7dof_normal`
- **What**: Identifier for your dataset
- **Why**: Used to locate and load the correct dataset
- **Typical Values**: Your dataset name (e.g., `eric_plating_v2`, `pusht`, etc.)

#### `--dataset.root /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal`
- **What**: Local filesystem path where your dataset is stored
- **Why**: LeRobot needs to know where to read the episodes/data from
- **Structure**: Should contain `meta/` folder and episode parquet files

---

### Policy Type

#### `--policy.type diffusion`
- **What**: Specifies which policy architecture to use
- **Why**: LeRobot supports multiple policies (ACT, Diffusion, TDMPC, VQ-BeT, etc.)
- **Options**: `diffusion`, `act`, `tdmpc`, `vqbet`
- **Your Choice**: You're using the Diffusion Policy architecture

---

### Temporal Parameters (CRITICAL!)

#### `--policy.horizon 32`
**What**: The number of future action timesteps the diffusion model predicts  
**Why**: The model learns to denoise a **trajectory of actions**, not just a single action  
**How it Works**:
- At each policy invocation, the model predicts 32 future actions
- These actions form a temporal sequence: `[a_t, a_{t+1}, ..., a_{t+31}]`

**Mathematical Formulation**:
```
Input:  observations from t-1 to t (n_obs_steps=2)
Output: action trajectory of length 32
        Actions shape: (batch_size, horizon=32, action_dim=7)
```

**Typical Values**: 16, 32, 64 (must be divisible by 2^len(down_dims))

---

#### `--policy.n_action_steps 16`
**What**: How many actions from the predicted horizon are actually executed  
**Why**: **Receding Horizon Control** - predict more than you execute for smoothness

**How it Works**:
1. Model predicts 32 actions: `[a_0, a_1, ..., a_31]`
2. Robot executes only the first 16: `[a_0, a_1, ..., a_15]`
3. On the next policy call, predict another 32 actions starting from the new state
4. Execute the next 16, and so on...

**Benefits**:
- **Smoother trajectories**: Future context helps current actions
- **Better long-term planning**: Model sees further ahead
- **Temporal consistency**: Overlapping predictions reduce jitter

**Common Patterns**:
- `horizon=32, n_action_steps=16` → Execute first half
- `horizon=64, n_action_steps=64` → Execute all (used in some configs)
- `horizon=16, n_action_steps=8` → Smaller model, less compute

---

#### `--policy.n_obs_steps 2`
**What**: Number of past observation frames to condition the policy on  
**Why**: Temporal context helps the model understand **velocity and dynamics**

**How it Works**:
- `n_obs_steps=2` means the model sees observations from timesteps `[t-1, t]`
- Enables the model to infer velocities: `v_t ≈ (s_t - s_{t-1}) / Δt`
- Helps with moving objects, dynamic scenes, and momentum

**Input Shape**:
```python
observation.state:  (batch, n_obs_steps=2, state_dim=7)
observation.images: (batch, n_obs_steps=2, num_cams, C, H, W)
```

**Typical Values**: 1 (single frame), 2 (velocity info), 3-4 (acceleration info)

**Tradeoff**:
- More obs_steps → Better dynamics understanding, but more memory
- Fewer obs_steps → Less memory, faster, but may miss motion cues

---

### Vision Backbone Arguments

#### `--policy.vision_backbone resnet18`
**What**: The CNN architecture used to encode camera images  
**Why**: Extract visual features from high-dimensional pixel observations

**Available Options** (in LeRobot):
- `resnet18` ← **Your choice** (lightweight, 11M params, fast)
- `resnet34` (21M params, more capacity)
- `resnet50` (25M params, very deep)

**Architecture Flow**:
```
Raw Image (3, 180, 320)
    ↓ [ResNet18 backbone]
Feature Map (512, H', W')
    ↓ [SpatialSoftmax with 32 keypoints]
Keypoint Coordinates (32, 2) = 64-dim vector
    ↓ [Concatenate with state]
Global Conditioning Vector
```

**Why ResNet18?**:
- Good balance between capacity and speed
- Pre-trained on ImageNet (better feature extraction)
- Works well for manipulation tasks with moderate visual complexity

---

#### `--policy.crop_shape "null"`
**What**: Crop images to (H, W) before passing to vision backbone  
**Why**: Focus on region of interest, reduce computation

**Your Setting**: `"null"` means **no cropping**, use full image
- Good if your camera view is well-framed
- Max information retention

**Alternative**: `--policy.crop_shape [153, 200]`
- Crops to 153×200 pixels
- `crop_is_random=True` → Random crop during training (data augmentation)
- `crop_is_random=False` → Center crop during evaluation

---

### Hardware Arguments

#### `--policy.device cuda`
- **What**: Use GPU for training
- **Why**: Diffusion training is compute-intensive (U-Net + multiple cameras)
- **Options**: `cuda`, `cpu`, `cuda:0`, `cuda:1` (specific GPU)

#### `--policy.push_to_hub False`
- **What**: Don't upload the trained model to Hugging Face Hub
- **Why**: You're training locally and want to keep models private
- **Alternative**: `True` to share models publicly

---

### Training Hyperparameters

#### `--batch_size 16`
**What**: Number of transition samples per training batch  
**Why**: Balances GPU memory usage vs. gradient stability

**Shape in Training**:
```python
batch = {
    "observation.state": (16, 2, 7),        # batch, n_obs_steps, state_dim
    "observation.images.front_cam": (16, 2, 3, 180, 320),
    "observation.images.head_cam": (16, 2, 3, 180, 320),
    "action": (16, 32, 7),                  # batch, horizon, action_dim
}
```

**Typical Values**:
- Small GPU (12GB): batch_size = 8-16
- Medium GPU (24GB): batch_size = 32-64
- Large GPU (48GB): batch_size = 128+

**Tradeoff**:
- Larger batch → More stable gradients, slower iteration
- Smaller batch → Faster iteration, potentially noisier gradients

---

#### `--num_workers 2`
**What**: Number of parallel data loading processes  
**Why**: Speed up data preprocessing (image decoding, augmentation)

**How DataLoader Works**:
```
Main Process (GPU training)
     ↑
     | (batches)
     |
DataLoader
  ├─ Worker 0: Load episodes, decode images
  ├─ Worker 1: Load episodes, decode images
  └─ ...
```

**Typical Values**:
- CPU-bound: `num_workers = min(4-8, num_cpu_cores)`
- Fast SSD: More workers help
- Slow HDD: Fewer workers (I/O bottleneck)

**Your Setting**: `num_workers=2` is conservative, safe for most systems

---

### Weights & Biases (W&B) Arguments

#### `--wandb.enable True`
- **What**: Enable experiment tracking with Weights & Biases
- **Why**: Track loss curves, metrics, hyperparameters, and system utilization
- **Requires**: `wandb login` beforehand

#### `--wandb.project picknplace_7dof_normal`
- **What**: W&B project name to organize runs
- **Why**: Group related experiments together
- **Result**: Visible at `https://wandb.ai/shennongshi/picknplace_7dof_normal`

#### `--wandb.entity shennongshi`
- **What**: Your W&B username or team name
- **Why**: Determines who owns the logged experiments

#### `--wandb.disable_artifact True`
- **What**: Don't upload dataset artifacts to W&B
- **Why**: Saves time and storage (datasets can be large)
- **Alternative**: `False` to version-control datasets

---

### Output Arguments

#### `--job_name picknplace_7dof_diffusion_baseline`
- **What**: Human-readable name for this training run
- **Why**: Easier to identify in logs, W&B, and saved checkpoints
- **Used in**: Checkpoint folder names, W&B run names

#### `--output_dir /mnt/nas/models/picknplace_7dof_diffusion/baseline`
- **What**: Directory where checkpoints and logs are saved
- **Contents**:
  ```
  baseline/
  ├── checkpoint_001000/
  │   ├── model.safetensors
  │   ├── config.json
  │   └── optimizer.bin
  ├── checkpoint_002000/
  └── ...
  ```

---

## 🧠 Advanced Diffusion Policy Concepts

### The U-Net Architecture

The core of Diffusion Policy is a **1D U-Net** that operates over the action trajectory temporal dimension:

```
Input: Noisy actions (batch, horizon=32, action_dim=7)
       + Timestep embedding
       + Global condition (visual + state features)

           ┌─────────────────┐
           │  Encoder Block  │ ─┐
           └────────┬────────┘  │
                    ↓            │ Skip Connection
           ┌─────────────────┐  │
           │ Downsample (÷2) │  │
           └────────┬────────┘  │
                    ↓            │
           ┌─────────────────┐  │
           │  Encoder Block  │ ─┤
           └────────┬────────┘  │
                    ↓            │
           ┌─────────────────┐  │
           │     Bottleneck  │  │
           └────────┬────────┘  │
                    ↓            │
           ┌─────────────────┐  │
           │  Decoder Block  │ ←┘
           └────────┬────────┘
                    ↓
           ┌─────────────────┐
           │   Upsample (×2) │
           └────────┬────────┘
                    ↓
Output: Denoised actions (batch, 32, 7)
```

**Key Parameters**:
- `down_dims = (512, 1024, 2048)`: Feature dimensions at each downsampling stage
- `kernel_size = 5`: Temporal convolution size
- `n_groups = 8`: Group normalization groups
- `diffusion_step_embed_dim = 128`: Embedding size for diffusion timestep

---

### Noise Scheduler

The noise scheduler defines the forward diffusion process (adding noise) and reverse process (denoising):

#### `--policy.noise_scheduler_type DDPM`
- **DDPM**: Denoising Diffusion Probabilistic Models (slower, more steps)
- **DDIM**: Denoising Diffusion Implicit Models (faster, deterministic)

#### `--policy.num_train_timesteps 100`
- **What**: Number of diffusion steps during training
- **Why**: More steps → finer denoising, but slower training

#### `--policy.beta_schedule squaredcos_cap_v2`
- **What**: How noise variance increases from step 0 to T
- **Options**: `linear`, `squaredcos_cap_v2`
- **Why**: `squaredcos_cap_v2` is the default from the paper

#### `--policy.beta_start 0.0001` and `--policy.beta_end 0.02`
- **What**: Range of noise variance across diffusion steps

#### `--policy.prediction_type epsilon`
- **What**: What the U-Net predicts during denoising
- **Options**:
  - `epsilon`: Predict the noise (recommended)
  - `sample`: Predict the clean action directly

---

### Loss Computation

During training, the diffusion model learns to predict noise:

```python
# Forward diffusion: Add noise to ground-truth actions
actions_gt = batch["action"]  # (batch, horizon, action_dim)
noise = torch.randn_like(actions_gt)
timesteps = torch.randint(0, num_train_timesteps, (batch,))

noisy_actions = sqrt(alpha_t) * actions_gt + sqrt(1 - alpha_t) * noise

# Model prediction
predicted_noise = diffusion_model(
    noisy_actions,
    timesteps,
    global_cond=encoded_observations
)

# MSE loss between predicted and actual noise
loss = F.mse_loss(predicted_noise, noise)
```

**Key Insight**: The model learns to **remove noise**, not directly predict actions!

---

## 📊 Normalization Strategy

Your config uses:
```json
"normalization_mapping": {
    "VISUAL": "MEAN_STD",
    "STATE": "MIN_MAX",
    "ACTION": "MIN_MAX"
}
```

**Why Different Normalization?**
- **Images**: MEAN_STD with ImageNet stats leverages pre-trained features
- **State/Action**: MIN_MAX ensures values fit in [-1, 1] for `clip_sample`

**Critical**: Actions MUST be normalized to [-1, 1] if `clip_sample=True`!

---

## 🎯 Next Steps for Your Study

### 1. Explore the Modeling Code
```bash
# View the main policy class
code [lerobot/common/policies/diffusion/modeling_diffusion.py](../lerobot/common/policies/diffusion/modeling_diffusion.py)
```
**Focus on**:
- `DiffusionPolicy.__init__`: How the model is constructed
- `DiffusionPolicy.select_action`: Inference logic with action queue
- `DiffusionModel.compute_loss`: Training loss computation

### 2. Understand the Data Flow
```bash
# View how datasets are loaded
code [lerobot/common/datasets/lerobot_dataset.py](../lerobot/common/datasets/lerobot_dataset.py)
```
**Focus on**:
- How `n_obs_steps` and `horizon` affect data loading
- How images and states are batched
- How padding is handled

### 3. Run a Small Experiment
Modify your command to do a quick overfitting test:
```bash
python3 -m [lerobot.scripts.train](../lerobot/scripts/train.py) \
    --dataset.repo_id picknplace_7dof_normal \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal \
    --policy.type diffusion \
    --dataset.episodes "[0, 1, 2]" \  # Only 3 episodes!
    --steps 1000 \                     # Short training
    --eval_freq 100 \
    --save_freq 500 \
    --output_dir /tmp/diffusion_overfit_test
```

If the model can overfit to 3 episodes, your setup is correct!

### 4. Compare with Flow Matching
Keep an eye on the research:
- **PointFlowMatch**: https://arxiv.org/abs/2410.xxxxx
- **FlowPolicy**: Search for recent papers on flow-based robot policies

---

## 📖 Recommended Reading Order

1. ✅ **Diffusion Policy Paper**: https://arxiv.org/abs/2303.04137
2. ⏳ **DDPM Paper** (foundational): "Denoising Diffusion Probabilistic Models"
3. ⏳ **Flow Matching Papers**: Search for "Conditional Flow Matching" + robotics
4. ⏳ **Code Walkthrough**: Start with [`modeling_diffusion.py`](../lerobot/common/policies/diffusion/modeling_diffusion.py)

---

## 🤔 Common Questions

### Q: Why `horizon=32` but `n_action_steps=16`?
**A**: Predicting longer trajectories provides temporal context, making current actions smoother. You only execute the first half, then re-plan.

### Q: What if I have limited GPU memory?
**A**: Reduce `batch_size`, use smaller `vision_backbone` (resnet18), reduce `horizon`, or disable `crop_is_random`.

### Q: How do I tune these hyperparameters?
**A**: Start with the paper defaults, then:
1. Tune `batch_size` for your GPU
2. Tune `horizon` and `n_action_steps` based on task dynamics
3. Tune learning rate if loss doesn't converge
4. Experiment with `n_obs_steps` if your task needs velocity info

### Q: Should I use Diffusion or Flow Matching?
**A**: For now, stick with Diffusion in LeRobot. Once you understand it deeply, you can experiment with flow matching implementations (not yet in LeRobot).

---

## 🎉 Summary

You now have:
✅ Understanding of the Diffusion Policy paper and technique  
✅ Knowledge of successor methods (Flow Matching)  
✅ Detailed explanation of every training argument  
✅ Insight into the model architecture and training process  
✅ Next steps for deeper code exploration  

Happy learning! 🚀
