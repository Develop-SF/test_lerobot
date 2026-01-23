# Essential Reading: Diffusion Policy & Flow Matching Papers

## Core Paper: Diffusion Policy

### Main Paper
- **Title**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **arXiv**: [2303.04137](https://arxiv.org/abs/2303.04137)
- **PDF**: https://arxiv.org/pdf/2303.04137.pdf
- **Project Page**: https://diffusion-policy.cs.columbia.edu/
- **Code**: https://github.com/real-stanford/diffusion_policy
- **Conference**: CoRL 2023 (Conference on Robot Learning)
- **Authors**: Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song

### Key Contributions
1. First to apply diffusion models to action prediction for robotics
2. 46.9% average improvement over prior methods across 12 tasks
3. Handles multimodal action distributions elegantly
4. Introduced receding horizon control for diffusion policies
5. Time-series diffusion transformer for temporal action trajectories

---

## Foundation: Understanding Diffusion Models

### DDPM (Foundational)
- **Title**: Denoising Diffusion Probabilistic Models
- **arXiv**: [2006.11239](https://arxiv.org/abs/2006.11239)
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Why Read**: Understand the core diffusion process
- **Key Concepts**: Forward diffusion, reverse diffusion, noise scheduling

### DDIM (Faster Inference)
- **Title**: Denoising Diffusion Implicit Models
- **arXiv**: [2010.02502](https://arxiv.org/abs/2010.02502)
- **Why Read**: Understand deterministic sampling for faster inference

---

## Successors & Extensions

### 1. Flow Matching for Robotics

#### PointFlowMatch
- **Title**: Learning Robotic Manipulation Policies from Point Cloud Observations with Conditional Flow Matching
- **PDF**: Search on arXiv (2024)
- **Key Advantage**: 2x performance on some tasks, works with point clouds
- **Why Read**: Best-in-class flow matching for manipulation

#### FlowPolicy
- **Title**: Flow Matching Policy: Learning Robot Policies with Flow Matching
- **Conference**: AAAI 2024
- **Key Advantage**: 7x faster inference than diffusion
- **Why Read**: Practical speedups for real-time control

#### Dynamics-Aligned Flow Matching
- **Venue**: Embodied AI Workshop
- **Key Innovation**: Integrates dynamics predictions into flow generation
- **Why Read**: Better generalization through mutual correction

### 2. Conditional Flow Matching (Theory)
- **Title**: Flow Matching for Generative Modeling
- **arXiv**: [2210.02747](https://arxiv.org/abs/2210.02747)
- **Why Read**: Understand the theoretical foundation of flow matching
- **Key Insight**: Flow matching is a generalization of diffusion models

---

## Related Diffusion Techniques for Robotics

### Diff-Control
- **Title**: Diff-Control: A Stateful Diffusion-based Policy for Imitation Learning
- **Year**: 2024
- **Key Innovation**: Adds statefulness to diffusion policies
- **Improvement**: Better robustness and success rates

### Trajectory Transformers
- **Title**: Offline Reinforcement Learning as One Big Sequence Modeling Problem
- **arXiv**: [2106.02039](https://arxiv.org/abs/2106.02039)
- **Why Read**: Alternative approach using transformers for trajectory modeling

---

## Complementary Techniques

### SpatialSoftmax
- **Title**: Deep Spatial Autoencoders for Visuomotor Learning
- **arXiv**: [1509.06113](https://arxiv.org/abs/1509.06113)
- **Authors**: Chelsea Finn et al.
- **Why Read**: Understand how visual features are extracted (used in Diffusion Policy)

### FiLM (Feature-wise Linear Modulation)
- **Title**: FiLM: Visual Reasoning with a General Conditioning Layer
- **arXiv**: [1709.07871](https://arxiv.org/abs/1709.07871)
- **Why Read**: Understand the conditioning mechanism in the U-Net

---

## Reading Order (Recommended)

### For Understanding Diffusion Policy:
1. ✅ **Start**: Diffusion Policy main paper (read intro + method sections)
2. ⏳ **Foundation**: DDPM (skim for intuition, don't get bogged down in math)
3. ⏳ **Deep Dive**: Diffusion Policy (full read with code)
4. ⏳ **Extensions**: Diff-Control, DDIM (optional)

### For Understanding Flow Matching:
1. ✅ **Theory**: Flow Matching for Generative Modeling
2. ⏳ **Application**: PointFlowMatch or FlowPolicy
3. ⏳ **Comparison**: Read discussions comparing diffusion vs flow matching

### For Implementation:
1. ✅ **Code First**: LeRobot's `modeling_diffusion.py`
2. ⏳ **Original Code**: Diffusion Policy GitHub repo
3. ⏳ **Experiments**: Try modifications and ablations

---

## Key Takeaways from Each Paper

### Diffusion Policy
- **Problem**: Robot policies struggle with multimodal action distributions
- **Solution**: Model actions as a denoising diffusion process
- **Innovation**: Temporal action diffusion with visual conditioning
- **Result**: SOTA on multiple benchmarks

### Flow Matching
- **Problem**: Diffusion is slow for real-time control
- **Solution**: Deterministic flow trajectories instead of stochastic diffusion
- **Innovation**: Simpler training, faster inference
- **Result**: 7x speedup with comparable performance

### DDPM
- **Problem**: How to model complex distributions?
- **Solution**: Gradually add noise, then learn to reverse it
- **Innovation**: Simplified training via noise prediction
- **Result**: Foundation for generative modeling

---

## Quick Links

### Download PDFs Directly:
```bash
# Diffusion Policy
wget https://arxiv.org/pdf/2303.04137.pdf -O diffusion_policy.pdf

# DDPM
wget https://arxiv.org/pdf/2006.11239.pdf -O ddpm.pdf

# Flow Matching
wget https://arxiv.org/pdf/2210.02747.pdf -O flow_matching.pdf
```

### Clone Code Repositories:
```bash
# Original Diffusion Policy
git clone https://github.com/real-stanford/diffusion_policy.git

# LeRobot (what you're using)
git clone https://github.com/huggingface/lerobot.git
```

---

## Active Research Areas (2024-2026)

1. **Flow Matching Policies**: Faster inference for real-time control
2. **Consistency Models**: Even faster than flow matching (1-step generation)
3. **Diffusion + World Models**: Combine with predictive models
4. **Hierarchy**: Multi-level diffusion for long-horizon tasks
5. **Language Conditioning**: Natural language instructions → diffusion policies

---

## Community & Resources

- **Discord**: Hugging Face LeRobot Discord
- **Twitter**: Follow @cheng_chi_, @columbia_ai_robotics
- **Blog Posts**: 
  - Hugging Face blog on diffusion policies
  - Lil'Log: "What are Diffusion Models?"
- **Videos**: 
  - CoRL 2023 presentation
  - YouTube tutorials on diffusion models

---

## Summary Table

| Paper | Year | Key Metric | Best For |
|-------|------|------------|----------|
| DDPM | 2020 | Foundation | Understanding diffusion |
| Diffusion Policy | 2023 | +46.9% over baselines | Robot manipulation |
| FlowPolicy | 2024 | 7x faster inference | Real-time control |
| PointFlowMatch | 2024 | 2x success rate | Point cloud tasks |
| Diff-Control | 2024 | Better robustness | Stateful policies |

---

Good luck with your reading! Start with the Diffusion Policy paper and come back when you have questions. 🚀
