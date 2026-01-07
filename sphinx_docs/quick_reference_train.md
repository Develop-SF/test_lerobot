# Quick Reference — Train

Train with absolute or relative actions.

## Absolute Actions

```bash
python lerobot/scripts/train.py \
    --config-path /path/to/config.json \
    --output-dir /path/to/run_abs
```

## Relative Actions (recommended)

```bash
# Optional stats for normalization
python lerobot/scripts/compute_relative_action_stats.py \
    --dataset-path /path/to/dataset \
    --arm-dim 6 \
    --output-path /path/to/relative_stats.json

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
    --dataset-path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2 \
    --arm-dim 6 \
    --output-path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json

# Train
python lerobot/scripts/train_with_relative_actions.py \
    --config_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/training_config.json \
    --output_dir /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/runs \
    --use_relative_actions \
    --arm_dim 6 \
    --obs_horizon 2 \
    --relative_stats_path /mnt/SF-Shared/dataset/robot_learning/lerobot/eric_plating_v2/relative_stats.json
```

## Notes
- `--use_relative_actions` enables delta actions.
- `--arm_dim` excludes gripper; typically 6.
- `--obs_horizon` controls frames used for observation.

## Generate training_config.json (Hydra method)

Create a valid base config using the standard training script, then convert it to JSON.

1) Emit a YAML config with a 1-step CPU run:

```bash
python lerobot/scripts/train.py \
    policy=diffusion \
    device=cpu \
    steps=1 \
    hydra.run.dir=outputs/config_gen
```

The YAML will be saved at `outputs/config_gen/config.yaml`.

2) Convert YAML → JSON (requires PyYAML):

```bash
pip install pyyaml
python -c "import yaml, json; print(json.dumps(yaml.safe_load(open('outputs/config_gen/config.yaml')), indent=2))" \
    > /path/to/training_config.json
```

3) Minimal manual adjustments (common):
- Add `policy.repo_id` if missing for local runs.
- Check dataset-related fields match your local dataset path.
- CLI flags like `--use_relative_actions`, `--arm_dim`, `--obs_horizon` remain as command-line overrides for `train_with_relative_actions.py`.
