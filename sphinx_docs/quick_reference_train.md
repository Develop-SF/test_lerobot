# Quick Reference — Train

Train with absolute or relative actions.

## Weights & Biases (W&B) Setup

To enable experiment tracking with W&B, you first need to log in and provide your API token.

1.  **Get your API Token:**
    *   Log in to [wandb.ai](https://wandb.ai/authorize).
    *   Copy your API key from the authorization page.
2.  **Login from the terminal:**
    ```bash
    wandb login
    ```
    Paste your API key when prompted. Alternatively, you can set the environment variable:
    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

## Baseline Training Configurations

The following are standard baseline configurations used for training.
It is recommended to use the default settings from `train.py` and only overwrite necessary parameters via command-line arguments. These configurations are provided just for reference as a workable baseline but not guaranteed to be optimal.

<details>
<summary><b>Jay's Baseline (Diffusion)</b></summary>

Jay's baseline configuration (`train_config.json`) for a diffusion policy with W&B enabled.

```json
{
    "dataset": {
        "repo_id": "path/to/your/dataset",
        "episodes": null,
        "image_transforms": {
            "enable": true,
            "max_num_transforms": 1,
            "random_order": false
        },
        "use_imagenet_stats": true,
        "video_backend": "pyav"
    },
    "policy": {
        "type": "diffusion",
        "n_obs_steps": 2,
        "device": "cuda",
        "use_amp": false,
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX"
        },
        "input_features": {
            "observation.images.head_cam": {
                "type": "VISUAL",
                "shape": [3, 183, 230]
            },
            "observation.images.left_arm_cam": {
                "type": "VISUAL",
                "shape": [3, 183, 230]
            },
            "observation.state": {
                "type": "STATE",
                "shape": [6]
            }
        },
        "output_features": {
            "action": {
                "type": "ACTION",
                "shape": [6]
            }
        },
        "horizon": 64,
        "n_action_steps": 64,
        "drop_n_last_frames": 7,
        "vision_backbone": "resnet18",
        "crop_shape": [153, 200],
        "crop_is_random": true,
        "pretrained_backbone_weights": null,
        "use_group_norm": true,
        "spatial_softmax_num_keypoints": 32,
        "use_separate_rgb_encoder_per_camera": false,
        "down_dims": [512, 1024, 2048],
        "kernel_size": 5,
        "n_groups": 8,
        "diffusion_step_embed_dim": 128,
        "use_film_scale_modulation": true,
        "noise_scheduler_type": "DDPM",
        "num_train_timesteps": 100,
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
        "clip_sample": true,
        "clip_sample_range": 1.0,
        "num_inference_steps": null,
        "do_mask_loss_for_padding": false,
        "optimizer_lr": 0.0001,
        "optimizer_betas": [0.95, 0.999],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 1e-06,
        "scheduler_name": "cosine",
        "scheduler_warmup_steps": 500
    },
    "job_name": "lerobot_diffusion_baseline",
    "resume": false,
    "seed": 100000,
    "num_workers": 16,
    "batch_size": 128,
    "steps": 100000,
    "eval_freq": 100,
    "log_freq": 100,
    "save_checkpoint": true,
    "save_freq": 10000,
    "use_policy_training_preset": true,
    "optimizer": {
        "type": "adam",
        "lr": 0.0001,
        "betas": [0.95, 0.999],
        "eps": 1e-08,
        "weight_decay": 1e-06,
        "grad_clip_norm": 10.0
    },
    "scheduler": {
        "type": "diffuser",
        "num_warmup_steps": 250,
        "name": "cosine"
    },
    "wandb": {
        "enable": true,
        "disable_artifact": true,
        "project": "lerobot",
        "entity": null,
        "notes": null
    }
}
```
</details>

<details>
<summary><b>Pick-N-Place 7DoF Diffusion Baseline</b></summary>

Baseline configuration for the 7DoF pick-and-place task.

```json
{
    "dataset": {
        "repo_id": "picknplace_7dof_normal",
        "root": "/mnt/nas/dataset/robot_learning/lerobot/picknplace_7dof_normal",
        "episodes": null,
        "image_transforms": {
            "enable": false,
            "max_num_transforms": 3,
            "random_order": false,
            "tfs": {
                "brightness": { "weight": 1.0, "type": "ColorJitter", "kwargs": { "brightness": [0.8, 1.2] } },
                "contrast": { "weight": 1.0, "type": "ColorJitter", "kwargs": { "contrast": [0.8, 1.2] } },
                "saturation": { "weight": 1.0, "type": "ColorJitter", "kwargs": { "saturation": [0.5, 1.5] } },
                "hue": { "weight": 1.0, "type": "ColorJitter", "kwargs": { "hue": [-0.05, 0.05] } },
                "sharpness": { "weight": 1.0, "type": "SharpnessJitter", "kwargs": { "sharpness": [0.5, 1.5] } }
            }
        },
        "revision": null,
        "use_imagenet_stats": true,
        "video_backend": "torchcodec"
    },
    "policy": {
        "type": "diffusion",
        "n_obs_steps": 2,
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX"
        },
        "input_features": {
            "observation.images.sync_front_cam": { "type": "VISUAL", "shape": [3, 180, 320] },
            "observation.images.sync_head_cam": { "type": "VISUAL", "shape": [3, 180, 320] },
            "observation.state": { "type": "STATE", "shape": [7] }
        },
        "output_features": {
            "action": { "type": "ACTION", "shape": [7] }
        },
        "device": "cuda",
        "use_amp": false,
        "push_to_hub": false,
        "horizon": 32,
        "n_action_steps": 16,
        "drop_n_last_frames": 7,
        "vision_backbone": "resnet18",
        "crop_is_random": true,
        "use_group_norm": true,
        "spatial_softmax_num_keypoints": 32,
        "down_dims": [512, 1024, 2048],
        "kernel_size": 5,
        "n_groups": 8,
        "diffusion_step_embed_dim": 128,
        "use_film_scale_modulation": true,
        "noise_scheduler_type": "DDPM",
        "num_train_timesteps": 100,
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
        "clip_sample": true,
        "clip_sample_range": 1.0,
        "optimizer_lr": 0.0001,
        "optimizer_betas": [0.95, 0.999],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 1e-06,
        "scheduler_name": "cosine",
        "scheduler_warmup_steps": 500
    },
    "job_name": "diffusion",
    "resume": false,
    "seed": 1000,
    "num_workers": 2,
    "batch_size": 16,
    "steps": 100000,
    "eval_freq": 20000,
    "log_freq": 200,
    "save_checkpoint": true,
    "save_freq": 20000,
    "use_policy_training_preset": true,
    "optimizer": {
        "type": "adam",
        "lr": 0.0001,
        "weight_decay": 1e-06,
        "grad_clip_norm": 10.0,
        "betas": [0.95, 0.999],
        "eps": 1e-08
    },
    "scheduler": {
        "type": "diffuser",
        "num_warmup_steps": 500,
        "name": "cosine"
    },
    "wandb": {
        "enable": true,
        "project": "picknplace_7dof_normal",
        "entity": "shennongshi"
    }
}
```
</details>


## Absolute Actions

```bash
python lerobot/scripts/train.py \
    --config-path /path/to/config.json \
    --output-dir /path/to/run_abs
```

**Real example:**
```bash
# 6 DoF Plating
python3 -m lerobot.scripts.train \
    --dataset.repo_id eric_plating_v2 \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2 \
    --policy.type diffusion \
    --policy.horizon 32 \
    --policy.n_action_steps 16 \
    --policy.n_obs_steps 2 \
    --policy.crop_shape "null" \
    --policy.vision_backbone resnet18 \
    --policy.push_to_hub False \
    --policy.device cuda \
    --batch_size 128 \
    --num_workers 8 \
    --wandb.enable True \
    --wandb.project eric_plating_v2 \
    --wandb.entity shennongshi \
    --wandb.disable_artifact True \
    --job_name eric_plating_v2_diffusion_baseline \
    --output_dir  /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/models/eric_plating_v2_diffusion/baseline

# 7 DoF Pick-N-Place
python3 -m lerobot.scripts.train \
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
    --batch_size 16 \
    --num_workers 2 \
    --wandb.enable True \
    --wandb.project picknplace_7dof_normal \
    --wandb.entity shennongshi \
    --wandb.disable_artifact True \
    --job_name picknplace_7dof_diffusion_baseline \
    --output_dir /mnt/nas/models/picknplace_7dof_diffusion/baseline
```

## Relative Actions (recommended)

```bash
# Generate training config (see below)
# Then compute stats for normalization
python lerobot/scripts/compute_relative_action_stats.py \
    --config_path /path/to/training_config.json \
    --output_path /path/to/relative_stats.json \
    --arm_dim 6

# Train
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /path/to/config.json \
    --output_dir /path/to/run_rel \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 2 \
    --relative_stats_path /path/to/relative_stats.json
```

**Real example:**
```bash
# Compute stats
python lerobot/scripts/compute_relative_action_stats.py \
    --config_path /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/training_config.json \
    --output_path /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json \
    --arm_dim 6

# Train
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/training_config.json \
    --output_dir /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/models/relative \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 2 \
    --relative_stats_path /mnt/nas/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json

# Compute stats (picknplace_6dof)
python lerobot/scripts/compute_relative_action_stats.py \
    --dataset-path /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof \
    --arm-dim 6 \
    --output-path /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/relative_stats.json

# Train (picknplace_6dof)
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/training_config.json \
    --output_dir /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/runs \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 15 \
    --relative_stats_path /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof/relative_stats.json
```

## Notes
- `--use_relative_actions` enables delta actions.
- `--arm_dim` excludes gripper; typically 6.
- `--obs_horizon` controls frames used for observation.

## Protect the models
Once the training is done, protect the model files to avoid accidental overwriting:

```bash
chmod -R a-w /path/to/output_dir
```

## Generate training_config.json

Create a valid base config using the standard training script, then save it as JSON.

1) Run a 1-step training to generate the config:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id your_dataset_repo_id \
    --dataset.root /path/to/your/dataset \
    --policy.type diffusion \
    --policy.device cpu \
    --policy.push_to_hub false \
    --steps 1 \
    --output_dir /path/to/your/dataset

# Real example (picknplace_6dof)
python lerobot/scripts/train.py \
    --dataset.repo_id picknplace_6dof \
    --dataset.root /mnt/nas/dataset/robot_learning/lerobot/picknplace_6dof \
    --policy.type diffusion \
    --policy.device cpu \
    --policy.push_to_hub false \
    --steps 1 \
    --output_dir /tmp/config_gen
```


2) Minimal manual adjustments (common):
- Ensure `dataset.repo_id` and `dataset.root` match your local dataset.
- Set `policy.push_to_hub` to false for local training.
- CLI flags like `--use_relative_actions`, `--arm_dim`, `--obs_horizon` remain as command-line overrides for `train_with_relative_actions.py`.
