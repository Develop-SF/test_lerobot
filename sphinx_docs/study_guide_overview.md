# Your Learning Path: Mastering Diffusion Policy Study Guide

## What You Have Now

I've created comprehensive documentation for you in the `.agent/` directory:

1. {doc}`study_guide_deep_dive` ← Start here
   - Paper background & key insights
   - Successor techniques (Flow Matching)
   - **Detailed explanation of EVERY training argument**
   - Architecture concepts

2. {doc}`study_guide_papers`
   - Complete paper references with links
   - Recommended reading order
   - Quick summaries of each paper

3. {doc}`study_guide_code_walkthrough`
   - Annotated code explanations
   - Step-by-step through the implementation
   - Training & inference loops

4. **`diffusion_policy_architecture.png`**
   - Visual diagram of the complete architecture
   - Shows data flow from cameras → ResNet → U-Net → actions

---

## Quick Reference: Your Training Command

```bash
python3 -m [lerobot.scripts.train](../lerobot/scripts/train.py) \
    --dataset.repo_id picknplace_7dof_normal \       # Dataset name
    --dataset.root /mnt/nas/.../picknplace_7dof_normal \  # Data location
    --policy.type diffusion \                         # Use Diffusion Policy
    
    # ⏱️ TEMPORAL PARAMETERS (most important!)
    --policy.horizon 32 \              # Predict 32 future actions
    --policy.n_action_steps 16 \       # Execute first 16 (receding horizon)
    --policy.n_obs_steps 2 \           # Use 2 past observations
    
    # 🖼️ VISION
    --policy.crop_shape "null" \       # No cropping, use full image
    --policy.vision_backbone resnet18 \ # ResNet-18 for image encoding
    
    # 💻 HARDWARE
    --policy.device cuda \             # Use GPU
    --batch_size 16 \                  # Batch size (GPU memory dependent)
    --num_workers 2 \                  # Data loading workers
    
    # 📊 LOGGING
    --wandb.enable True \              # Track experiments
    --wandb.project picknplace_7dof_normal \
    --wandb.entity shennongshi \
    --wandb.disable_artifact True \    # Don't upload dataset
    
    # 💾 OUTPUT
    --job_name picknplace_7dof_diffusion_baseline \
    --output_dir /mnt/nas/models/.../baseline
```

---

## 📖 Recommended Study Order

### Week 1: Understand the Theory

1. **Read the Deep Dive** ({doc}`study_guide_deep_dive`)
   - Focus on "How Diffusion Works" section
   - Understand the analogy: noise → clean actions
   - **Key insight**: Model learns to predict noise, not actions!

2. **Read the Main Paper** ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137))
   - Start with intro, skip heavy math initially
   - Focus on Section 3 (Method)
   - Look at the figures and diagrams
   - **Don't get stuck on equations!** Come back later.

3. **Watch a Tutorial** (optional)
   - Search YouTube for "Diffusion Models explained"
   - Understand the forward/reverse process visually

### Week 2: Dive Into Code

4. **Trace Through [`modeling_diffusion.py`](../lerobot/common/policies/diffusion/modeling_diffusion.py)**
   - Start with `DiffusionPolicy.__init__` (line 62)
   - Follow `select_action()` (line 130) - this is the inference flow
   - Use the code walkthrough document as your guide

5. **Print Tensor Shapes**
   - Add print statements in key methods
   - Run a small training to see shapes flow through
   - Example:
   ```python
   print(f"obs.state shape: {batch['observation.state'].shape}")
   print(f"actions shape: {actions.shape}")
   ```

6. **Modify a Hyperparameter**
   - Try `--policy.horizon 64` instead of 32
   - Observe how it affects training
   - Read the error messages carefully

### Week 3: Understand Your Configuration

7. **Examine Default Values**
   - Open [`configuration_diffusion.py`](../lerobot/common/policies/diffusion/configuration_diffusion.py)
   - Compare defaults to your CLI overrides
   - Understand why certain values are used

8. **Read About Noise Schedulers**
   - DDPM vs DDIM
   - Why `beta_schedule = "squaredcos_cap_v2"`
   - How `num_train_timesteps` affects training

9. **Explore Normalization**
   - Why images use MEAN_STD (ImageNet stats)
   - Why actions use MIN_MAX (need to fit in [-1, 1])
   - What happens if you get this wrong?

### Week 4: Advanced Topics

10. **Flow Matching (Successor Technique)**
    - Read papers listed in {doc}`study_guide_papers`
    - Compare performance/speed with Diffusion
    - Consider if you want to implement this later

11. **Relative Actions**
    - Understand the `use_relative_actions` flag
    - How velocity-based control works
    - When to use absolute vs relative

12. **Receding Horizon Control**
    - Why predict more than you execute?
    - Relationship between `horizon` and `n_action_steps`
    - Impact on smoothness and planning

---

## 🎓 Learning Checkpoints

After each week, test yourself:

### ✅ Week 1 Checklist
- [ ] Can you explain diffusion in 3 sentences?
- [ ] Do you understand forward vs reverse diffusion?
- [ ] Can you explain why we predict noise instead of actions?
- [ ] Do you know what "receding horizon" means?

### ✅ Week 2 Checklist
- [ ] Can you trace a batch through `select_action()`?
- [ ] Do you understand the observation queue?
- [ ] Do you understand the action queue?
- [ ] Can you modify `horizon` without breaking the code?

### ✅ Week 3 Checklist
- [ ] Can you explain each argument in your training command?
- [ ] Do you know why `horizon` must be divisible by 8?
- [ ] Can you explain the normalization strategy?
- [ ] Do you understand the U-Net architecture?

### ✅ Week 4 Checklist
- [ ] Can you compare Diffusion vs Flow Matching?
- [ ] Do you understand when to use relative actions?
- [ ] Can you explain the full data flow from camera → action?
- [ ] Are you ready to make your own modifications?

---

## 🧪 Hands-On Experiments

### Experiment 1: Overfit Test
**Goal**: Verify your setup works

```bash
python3 -m [lerobot.scripts.train](../lerobot/scripts/train.py) \
    --dataset.repo_id picknplace_7dof_normal \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal \
    --policy.type diffusion \
    --dataset.episodes "[0, 1, 2]" \  # Only 3 episodes
    --policy.horizon 32 \
    --policy.n_action_steps 16 \
    --policy.n_obs_steps 2 \
    --policy.crop_shape "null" \
    --policy.vision_backbone resnet18 \
    --batch_size 8 \
    --num_workers 2 \
    --steps 1000 \                    # Short training
    --eval_freq 250 \
    --save_freq 500 \
    --output_dir /tmp/overfit_test \
    --wandb.enable False              # Disable for quick test
```

**Success Criteria**: Loss should drop to near zero. If not, something is wrong!

---

### Experiment 2: Horizon Ablation
**Goal**: Understand the impact of `horizon`

Run 3 experiments with different horizons:
- `--policy.horizon 16 --policy.n_action_steps 8`
- `--policy.horizon 32 --policy.n_action_steps 16` ← your current
- `--policy.horizon 64 --policy.n_action_steps 32`

**Compare**:
- Training speed (iterations/sec)
- GPU memory usage
- Final performance

---

### Experiment 3: Vision Backbone Comparison
**Goal**: Understand model capacity tradeoffs

```bash
# Baseline
--policy.vision_backbone resnet18   # 11M params

# Heavier model
--policy.vision_backbone resnet34   # 21M params
```

**Compare**:
- Training time
- GPU memory
- Generalization performance

---

### Experiment 4: Observation Steps
**Goal**: See impact of temporal context

```bash
# Single frame
--policy.n_obs_steps 1

# Your current config
--policy.n_obs_steps 2

# More history
--policy.n_obs_steps 3
```

**Hypothesis**: More obs_steps helps with moving objects but costs memory

---

## 🔧 Debugging Tips

### If training loss doesn't decrease:
1. Check normalization: Are actions in [-1, 1]?
2. Check data loading: Are images decoded correctly?
3. Reduce batch size, try overfitting to 1 episode
4. Check learning rate (try 1e-5 or 5e-4)

### If you run out of GPU memory:
1. Reduce `batch_size`
2. Reduce `horizon`
3. Use smaller vision backbone (`resnet18` → custom lighter model)
4. Disable `crop_is_random` during eval

### If actions look jittery:
1. Increase `horizon` (more smoothing)
2. Increase `n_action_steps` (execute more at once)
3. Check if you need to use relative actions

---

## 🚀 Next Level: Going Beyond

Once you've mastered Diffusion Policy:

1. **Implement Flow Matching**
   - Start from Diffusion, modify to use flows
   - Aim for 5-7x speedup

2. **Try Different Backbones**
   - Vision Transformers (ViT)
   - ConvNeXt
   - EfficientNet

3. **Multi-Modal Inputs**
   - Add force/torque sensors
   - Add language instructions
   - Add audio (for contact-rich tasks)

4. **Scale Up**
   - Train on multiple tasks
   - Use larger datasets
   - Fine-tune pre-trained models

---

## 📚 Additional Resources

### Community
- **LeRobot Discord**: Ask questions to the community
- **GitHub Issues**: Search for similar problems
- **Paper Discussions**: Twitter/X threads on the paper

### Tools
- **WandB**: Track hyperparameter sweeps
- **TensorBoard**: Visualize internal activations
- **Netron**: Visualize the model architecture

### Advanced Reading
- ACT (Action Chunking Transformer)
- Behavior Transformers (BeT)
- VQ-BeT (Vector Quantized BeT)
- TDMPC (Temporal Difference Model Predictive Control)

---

## 💡 Key Takeaways

1. **Diffusion = Learned Denoising**
   - Train: Learn to remove noise from actions
   - Inference: Iteratively denoise random noise → clean actions

2. **Receding Horizon = Predict More, Execute Less**
   - Predicting longer trajectories provides context
   - Only execute first `n_action_steps`, then re-plan

3. **Temporal Observations = Velocity Information**
   - `n_obs_steps > 1` lets the model infer dynamics
   - Critical for tasks with moving objects

4. **Normalization = Critical for Success**
   - Images: MEAN_STD with ImageNet stats
   - Actions: MIN_MAX to fit in [-1, 1]
   - Get this wrong → training fails

5. **Flow Matching = Future**
   - Same expressiveness as diffusion
   - 5-7x faster inference
   - Simpler training

---

## 🎉 You're Ready!

You now have:
✅ Complete understanding of Diffusion Policy theory  
✅ Knowledge of every training argument  
✅ Code walkthrough with examples  
✅ Structured learning path  
✅ Hands-on experiments to try  
✅ Debugging strategies  
✅ Next steps for advanced work  

**Start with the deep dive document and work through the checklist!**

Good luck on your Diffusion Policy journey! 🚀🤖

---

## Quick Links

- [Main Paper (arXiv)](https://arxiv.org/abs/2303.04137)
- [Original Code](https://github.com/real-stanford/diffusion_policy)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- Refer to the {doc}`quick_reference` for full training CLI examples.

---

*Last updated: 2026-01-20*
