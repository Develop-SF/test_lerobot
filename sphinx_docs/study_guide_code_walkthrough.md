# Code Walkthrough: Diffusion Policy Implementation

## File Structure

```
lerobot/common/policies/diffusion/
├── [`configuration_diffusion.py`](../lerobot/common/policies/diffusion/configuration_diffusion.py)   # Config dataclass with all hyperparameters
├── [`modeling_diffusion.py`](../lerobot/common/policies/diffusion/modeling_diffusion.py)        # Main implementation (DiffusionPolicy + DiffusionModel)
└── __init__.py
```

---

## Part 1: Configuration (`configuration_diffusion.py`)

### Key Configuration Parameters

```python
@dataclass
class DiffusionConfig(PreTrainedConfig):
    # Temporal parameters - MOST IMPORTANT
    n_obs_steps: int = 2          # How many past observations to use
    horizon: int = 16             # Action sequence length to predict
    n_action_steps: int = 8       # How many actions to actually execute
    
    # Vision backbone
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    spatial_softmax_num_keypoints: int = 32
    
    # U-Net architecture
    down_dims: tuple[int, ...] = (512, 1024, 2048)  # Feature dims at each stage
    kernel_size: int = 5                             # Conv kernel size
    n_groups: int = 8                                # GroupNorm groups
    diffusion_step_embed_dim: int = 128              # Timestep embedding dim
    
    # Noise scheduler
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # Predict noise, not action
    
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
```

### Important Validation

The config validates that `horizon` is compatible with U-Net downsampling:
```python
downsampling_factor = 2 ** len(self.down_dims)  # 2^3 = 8
if self.horizon % downsampling_factor != 0:
    raise ValueError(...)
```

For `down_dims = (512, 1024, 2048)`, horizon must be divisible by 8:
- ✅ Valid: 16, 32, 64
- ❌ Invalid: 15, 30, 100

---

## Part 2: Policy Wrapper (`DiffusionPolicy` in `modeling_diffusion.py`)

This is the main interface for training and inference.

### Initialization

```python
class DiffusionPolicy(PreTrainedPolicy):
    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        use_relative_actions: bool = False,  # Your use case!
        arm_dim: int = 6,
    ):
        # 1. Store config
        self.config = config
        
        # 2. Create normalization layers
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            dataset_stats,
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats,
        )
        self.unnormalize_outputs = Unnormalize(...)
        
        # 3. Create the core diffusion model
        self.diffusion_model = DiffusionModel(config)
        
        # 4. Setup for relative actions (if enabled)
        self.use_relative_actions = use_relative_actions
        self.arm_dim = arm_dim
        
        # 5. Initialize action and observation queues
        self.reset()
```

---

### The `reset()` Method

Called at the start of each episode:

```python
def reset(self):
    """Clear observation and action queues."""
    # Queue to store past observations
    self._queues = {
        f"observation.state": deque(maxlen=self.config.n_obs_steps),
        f"observation.images.{cam_name}": deque(maxlen=self.config.n_obs_steps),
        # ... for each camera
    }
    
    # Cache for predicted action trajectory
    self._action_queue = deque(maxlen=self.config.n_action_steps)
```

**Why queues?**
- Observation queue: Collect `n_obs_steps` past frames
- Action queue: Cache predicted actions for reuse (receding horizon)

---

### The `select_action()` Method (Inference)

This is where the magic happens!

```python
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    """Select a single action given environment observations."""
    
    # Step 1: Populate observation queues
    # Add current observation to the queue (pop oldest if full)
    for key in batch:
        self._queues[key].append(batch[key])
    
    # Step 2: Check if we can reuse cached actions
    if len(self._action_queue) > 0:
        # We still have predicted actions from previous step
        return self._action_queue.popleft()  # Return next cached action
    
    # Step 3: Need to generate new action trajectory
    # Stack observations to create temporal dimension
    batch = {
        key: torch.stack(list(self._queues[key]), dim=1)
        # Shape: (batch=1, n_obs_steps, ...)
    }
    
    # Step 4: Handle relative actions (if enabled)
    if self.use_relative_actions:
        # Get current arm state from observations
        current_arm_state = get_current_arm_state(batch, self.arm_dim)
        
        # Convert observations to relative (velocity-based)
        batch = convert_observation_state_to_relative(batch, self.config.n_obs_steps)
    
    # Step 5: Normalize inputs
    batch = self.normalize_inputs(batch)
    
    # Step 6: Generate action trajectory using diffusion model
    with torch.no_grad():
        actions = self.diffusion_model.generate_actions(batch)
        # Shape: (1, horizon, action_dim)
    
    # Step 7: Unnormalize actions
    actions = self.unnormalize_outputs({"action": actions})["action"]
    
    # Step 8: Convert back to absolute actions (if using relative)
    if self.use_relative_actions:
        actions = convert_actions_to_absolute(
            actions,
            current_arm_state,
            self.arm_dim
        )
    
    # Step 9: Cache the action trajectory
    # Only keep first n_action_steps (receding horizon!)
    for action in actions[0, :self.config.n_action_steps]:
        self._action_queue.append(action)
    
    # Step 10: Return the first action
    return self._action_queue.popleft()
```

**Key Insights**:
1. **Observation Queue**: Maintains history for temporal reasoning
2. **Action Queue**: Caches predictions to avoid redundant computation
3. **Receding Horizon**: Predict 32, execute 16, re-predict
4. **Relative Actions**: Enable velocity-based control for better generalization

---

### The `forward()` Method (Training)

```python
def forward(self, batch: dict[str, Tensor]) -> dict:
    """Compute loss for training."""
    
    # Step 1: Handle relative actions
    if self.use_relative_actions:
        batch = convert_observation_state_to_relative(batch, self.config.n_obs_steps)
        # Convert absolute actions to relative (velocities)
        # ... similar processing as select_action
    
    # Step 2: Normalize inputs and targets
    batch = self.normalize_inputs(batch)
    batch = self.normalize_targets(batch)
    
    # Step 3: Compute diffusion loss
    loss = self.diffusion_model.compute_loss(batch)
    
    return {"loss": loss}
```

---

## Part 3: Diffusion Model Core (`DiffusionModel` in `modeling_diffusion.py`)

This implements the actual diffusion process.

### Architecture Components

```python
class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        
        # 1. Vision encoders (one per camera or shared)
        if config.use_separate_rgb_encoder_per_camera:
            self.rgb_encoders = nn.ModuleDict({
                cam_name: self._make_rgb_encoder(config)
                for cam_name in image_features
            })
        else:
            self.rgb_encoder = self._make_rgb_encoder(config)
        
        # 2. State encoder (if robot state is provided)
        if robot_state_feature:
            self.state_encoder = nn.Linear(state_dim, state_embed_dim)
        
        # 3. Noise prediction network (U-Net)
        self.unet = ConditionalUnet1D(
            global_cond_dim=global_cond_dim,  # From vision + state
            input_dim=action_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )
        
        # 4. Noise scheduler
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            # ... other params
        )
```

---

### Vision Encoder Details

```python
def _make_rgb_encoder(self, config):
    """Create ResNet + SpatialSoftmax encoder."""
    
    # 1. ResNet backbone
    backbone = getattr(torchvision.models, config.vision_backbone)(
        pretrained=config.pretrained_backbone_weights is not None
    )
    
    # 2. Replace BatchNorm with GroupNorm (better for small batches)
    if config.use_group_norm:
        backbone = replace_bn_with_gn(backbone)
    
    # 3. Remove final classification layers
    # Keep only convolutional feature extractor
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    
    # 4. Add SpatialSoftmax
    # Converts (B, C, H, W) → (B, num_keypoints * 2)
    spatial_softmax = SpatialSoftmax(
        input_shape=backbone_output_shape,
        num_kp=config.spatial_softmax_num_keypoints,
    )
    
    return nn.Sequential(
        backbone,
        spatial_softmax,
    )
```

**Why SpatialSoftmax?**
- Reduces spatial features to keypoint coordinates
- More interpretable than global pooling
- Better for spatial reasoning in manipulation

---

### Global Conditioning Preparation

```python
def _prepare_global_conditioning(self, batch):
    """Encode observations into global conditioning vector."""
    
    feature_list = []
    
    # 1. Encode each camera view
    for cam_name, cam_images in batch["observation.images"].items():
        # cam_images shape: (B, n_obs_steps, C, H, W)
        B, T = cam_images.shape[:2]
        
        # Flatten batch and time: (B * T, C, H, W)
        flat_images = cam_images.flatten(0, 1)
        
        # Optional: Random crop (data augmentation during training)
        if self.training and config.crop_is_random:
            flat_images = random_crop(flat_images, config.crop_shape)
        else:
            flat_images = center_crop(flat_images, config.crop_shape)
        
        # Encode: (B * T, C, H, W) → (B * T, feature_dim)
        features = self.rgb_encoder(flat_images)
        
        # Reshape: (B * T, feature_dim) → (B, T * feature_dim)
        features = features.reshape(B, T * features.shape[-1])
        
        feature_list.append(features)
    
    # 2. Encode robot state (if provided)
    if "observation.state" in batch:
        state = batch["observation.state"]  # (B, n_obs_steps, state_dim)
        # Flatten: (B, n_obs_steps * state_dim)
        state_flat = state.flatten(1)
        state_encoded = self.state_encoder(state_flat)
        feature_list.append(state_encoded)
    
    # 3. Concatenate all features
    global_cond = torch.cat(feature_list, dim=-1)
    # Shape: (B, total_feature_dim)
    
    return global_cond
```

---

### Action Generation (Inference)

```python
def generate_actions(self, batch):
    """Generate actions via reverse diffusion."""
    
    B = batch["observation.state"].shape[0]
    device = self.device
    
    # Step 1: Encode observations
    global_cond = self._prepare_global_conditioning(batch)
    
    # Step 2: Initialize with pure noise
    actions = torch.randn(
        B, self.config.horizon, self.action_dim,
        device=device
    )
    
    # Step 3: Reverse diffusion process
    num_inference_steps = self.config.num_inference_steps or self.config.num_train_timesteps
    self.noise_scheduler.set_timesteps(num_inference_steps)
    
    for t in self.noise_scheduler.timesteps:
        # Predict noise at timestep t
        noise_pred = self.unet(
            actions,                      # Noisy actions
            timestep=t,                   # Current diffusion timestep
            global_cond=global_cond,      # Observation features
        )
        
        # Remove predicted noise (denoising step)
        actions = self.noise_scheduler.step(
            noise_pred,
            t,
            actions,
            generator=None,
        ).prev_sample
        
        # Optional: Clip actions to [-1, 1] (important!)
        if self.config.clip_sample:
            actions = actions.clamp(
                -self.config.clip_sample_range,
                self.config.clip_sample_range
            )
    
    return actions  # (B, horizon, action_dim)
```

**Reverse Diffusion Process**:
```
Noise (t=100) → Slightly denoised (t=99) → ... → Clean actions (t=0)
```

---

### Loss Computation (Training)

```python
def compute_loss(self, batch):
    """Compute diffusion training loss."""
    
    B = batch["observation.state"].shape[0]
    device = self.device
    
    # Step 1: Get ground-truth actions
    actions_gt = batch["action"]  # (B, horizon, action_dim)
    
    # Step 2: Encode observations
    global_cond = self._prepare_global_conditioning(batch)
    
    # Step 3: Sample random timesteps
    timesteps = torch.randint(
        0, self.config.num_train_timesteps,
        (B,),
        device=device
    )
    
    # Step 4: Sample noise
    noise = torch.randn_like(actions_gt)
    
    # Step 5: Forward diffusion: Add noise to actions
    # actions_t = sqrt(α_t) * actions_gt + sqrt(1 - α_t) * noise
    noisy_actions = self.noise_scheduler.add_noise(
        actions_gt,
        noise,
        timesteps
    )
    
    # Step 6: Predict the noise
    noise_pred = self.unet(
        noisy_actions,
        timestep=timesteps,
        global_cond=global_cond,
    )
    
    # Step 7: MSE loss between predicted and actual noise
    if self.config.do_mask_loss_for_padding:
        # Mask out padded actions
        mask = ~batch["action_is_pad"]
        loss = F.mse_loss(noise_pred[mask], noise[mask])
    else:
        loss = F.mse_loss(noise_pred, noise)
    
    return loss
```

**Key Insight**: We DON'T predict actions directly!  
We predict the **noise**, which is easier to learn.

---

## Part 4: The U-Net (`ConditionalUnet1D`)

The U-Net is defined elsewhere, but here's the conceptual structure:

```python
class ConditionalUnet1D(nn.Module):
    """1D U-Net with FiLM conditioning."""
    
    def forward(self, x, timestep, global_cond):
        # x: (B, horizon, action_dim)
        
        # 1. Embed the diffusion timestep
        t_emb = self.time_embed(timestep)  # (B, embed_dim)
        
        # 2. Combine with global conditioning
        cond = torch.cat([t_emb, global_cond], dim=-1)
        
        # 3. Encoder path (downsample)
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, cond)  # FiLM conditioning
            skip_connections.append(x)
            x = downsample(x)  # Reduce temporal dimension by 2
        
        # 4. Bottleneck
        x = self.bottleneck(x, cond)
        
        # 5. Decoder path (upsample)
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)  # Double temporal dimension
            x = torch.cat([x, skip], dim=-1)  # Skip connection
            x = decoder_block(x, cond)  # FiLM conditioning
        
        # 6. Final output
        return x  # (B, horizon, action_dim)
```

---

## Part 5: Putting It All Together

### Training Loop (Simplified)

```python
# Initialize
policy = DiffusionPolicy(config, dataset_stats, use_relative_actions=True, arm_dim=7)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = LeRobotDataLoader(dataset, batch_size=16)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch contains:
        # - observation.state: (16, 2, 7)
        # - observation.images.front_cam: (16, 2, 3, 180, 320)
        # - observation.images.head_cam: (16, 2, 3, 180, 320)
        # - action: (16, 32, 7)
        
        # Forward pass
        output = policy.forward(batch)
        loss = output["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
```

### Inference Loop (Simplified)

```python
# Load trained policy
policy = DiffusionPolicy.from_pretrained("path/to/checkpoint")
policy.eval()

# Reset at episode start
policy.reset()
obs = env.reset()

# Rollout
while not done:
    # Create batch
    batch = {
        "observation.state": obs["state"].unsqueeze(0),
        "observation.images.front_cam": obs["front_cam"].unsqueeze(0),
        "observation.images.head_cam": obs["head_cam"].unsqueeze(0),
    }
    
    # Get action
    with torch.no_grad():
        action = policy.select_action(batch)
    
    # Execute in environment
    obs, reward, done, info = env.step(action)
```

---

## Key Takeaways

1. **DiffusionPolicy** is a wrapper that handles:
   - Normalization
   - Observation/action queues
   - Relative action conversion
   - Interface to the core model

2. **DiffusionModel** implements:
   - Vision encoding (ResNet + SpatialSoftmax)
   - U-Net for noise prediction
   - Forward/reverse diffusion process

3. **Training** learns to predict noise, not actions directly

4. **Inference** uses reverse diffusion to denoise random noise into actions

5. **Receding Horizon** predicts long trajectories but only executes a portion

---

## Next Steps

1. **Read the code**: Start with `DiffusionPolicy.select_action()`
2. **Trace a batch**: Follow a single batch through the entire pipeline
3. **Visualize**: Print shapes at each step to understand dimensions
4. **Experiment**: Modify `horizon` or `n_obs_steps` and observe effects

Good luck diving into the code! 🚀
