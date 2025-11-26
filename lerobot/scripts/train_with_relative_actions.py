#!/usr/bin/env python

# Copyright 2024 The LeRobot Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Extended training script with relative action support.

This script extends the standard lerobot training pipeline to support relative action
training following UMI and ManiSkill principles (PD2.1 + PD2.2).

Usage:
    python train_with_relative_actions.py \\
        --config_path path/to/config.json \\
        --output_dir path/to/output \\
        --use_relative_actions \\
        --arm_dim 6 \\
        --obs_horizon 2

The script automatically wraps the dataset and policy to handle relative action conversion.
"""

import argparse
import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from termcolor import colored

from lerobot.common.datasets.relative_action_dataset import RelativeActionDataset
from lerobot.common.policies.factory import make_policy
from lerobot.scripts.train import train, parser, TrainPipelineConfig
from lerobot.common.datasets.factory import make_dataset


def parse_args():
    """Parse command-line arguments for relative action training."""
    arg_parser = argparse.ArgumentParser(
        description="Train lerobot policies with relative action support"
    )
    
    # Standard lerobot arguments
    arg_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration JSON file"
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs (checkpoints, logs, etc.)"
    )
    
    # Relative action arguments
    arg_parser.add_argument(
        "--use_relative_actions",
        action="store_true",
        help="Enable relative action training (PD2.1 + PD2.2)"
    )
    arg_parser.add_argument(
        "--arm_dim",
        type=int,
        default=6,
        help="Number of arm joints (excluding gripper). Default: 6"
    )
    arg_parser.add_argument(
        "--obs_horizon",
        type=int,
        default=None,
        help="Observation horizon. If not specified, uses value from config."
    )
    arg_parser.add_argument(
        "--relative_stats_path",
        type=str,
        default=None,
        help="Path to JSON file with pre-computed normalization statistics from relative representations. "
             "If not provided, statistics will be computed on-the-fly following UMI's approach "
             "(iterating through the dataset after conversion to relative). "
             "Pre-computing can be done using: python lerobot/scripts/compute_relative_action_stats.py"
    )
    
    # Additional policy arguments (optional overrides)
    arg_parser.add_argument(
        "--policy.push_to_hub",
        dest="policy_push_to_hub",
        type=lambda x: (str(x).lower() == 'true'),
        default=None,
        help="Whether to push the policy to Hugging Face Hub (true/false)"
    )
    
    return arg_parser.parse_args()


def load_config_from_json(config_path: str) -> DictConfig:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config_dict)
    
    return cfg


def train_with_relative_actions(
    cfg: DictConfig,
    use_relative_actions: bool,
    arm_dim: int,
    obs_horizon: int | None = None,
    relative_stats_path: str | None = None,
    config_path: str | None = None,
):
    """
    Extended training function that wraps the dataset and policy for relative actions.
    
    Args:
        cfg: Training configuration
        use_relative_actions: Whether to use relative action mode
        arm_dim: Number of arm joints (excluding gripper)
        obs_horizon: Observation horizon (uses config value if None)
        relative_stats_path: Path to pre-computed statistics file
        config_path: Path to the config file (for cache file generation)
    """
    # Parse config into TrainPipelineConfig
    # Note: We need to work with the existing training infrastructure
    # The easiest way is to patch the dataset after creation
    
    logging.info("="*80)
    if use_relative_actions:
        logging.info(colored("RELATIVE ACTION MODE ENABLED", "green", attrs=["bold"]))
        logging.info(f"  - Arm dimension: {arm_dim}")
        logging.info(f"  - PD2.1: Actions relative to current position")
        logging.info(f"  - PD2.2: Observations relative to current position")
    else:
        logging.info(colored("ABSOLUTE ACTION MODE", "yellow", attrs=["bold"]))
    logging.info("="*80)
    
    # Import train function components
    from lerobot.scripts.train import update_policy
    from lerobot.common.datasets.sampler import EpisodeAwareSampler
    from lerobot.common.datasets.utils import cycle
    from lerobot.common.optim.factory import make_optimizer_and_scheduler
    from lerobot.common.envs.factory import make_env
    from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.common.utils.random_utils import set_seed
    from lerobot.common.utils.train_utils import (
        get_step_checkpoint_dir,
        get_step_identifier,
        load_training_state,
        save_checkpoint,
        update_last_checkpoint,
    )
    from lerobot.common.utils.utils import format_big_number, get_safe_torch_device
    from lerobot.common.utils.wandb_utils import WandBLogger
    from lerobot.scripts.eval import eval_policy
    from torch.cuda.amp import GradScaler
    from contextlib import nullcontext
    import torch
    import time
    from pprint import pformat
    import draccus
    
    # Validate and create TrainPipelineConfig
    # Convert OmegaConf to dict, then use draccus to properly instantiate the dataclass
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with draccus.config_type("json"):
        cfg_obj = draccus.decode(TrainPipelineConfig, cfg_dict)
    cfg_obj.validate()
    logging.info(pformat(cfg_obj.to_dict()))
    
    if cfg_obj.wandb.enable and cfg_obj.wandb.project:
        wandb_logger = WandBLogger(cfg_obj)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    if cfg_obj.seed is not None:
        set_seed(cfg_obj.seed)
    
    # Check device
    device = get_safe_torch_device(cfg_obj.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create dataset
    logging.info("Creating dataset")
    dataset = make_dataset(cfg_obj)
    
    # Wrap dataset with RelativeActionDataset if relative actions are enabled
    if use_relative_actions:
        # Use obs_horizon from config if not specified
        if obs_horizon is None:
            obs_horizon = cfg_obj.policy.n_obs_steps
        
        logging.info(f"Wrapping dataset with RelativeActionDataset (arm_dim={arm_dim}, obs_horizon={obs_horizon})")
        
        # Determine which stats file to use (priority: manual > cache > compute)
        stats_file_to_use = None
        
        if relative_stats_path is not None:
            # Priority 1: User-provided stats file
            logging.info(f"Using pre-computed statistics from: {relative_stats_path}")
            logging.info("  - Statistics loaded from file (manual override)")
            stats_file_to_use = relative_stats_path
        else:
            # Generate cache file path in the same directory as config
            config_path = Path(cfg_obj.config_path) if hasattr(cfg_obj, 'config_path') else None
            if config_path is None:
                # Fallback: use output_dir
                cache_stats_path = Path(cfg_obj.output_dir) / "relative_stats_cache.json"
            else:
                # Save in same directory as config file
                cache_stats_path = config_path.parent / f"relative_stats_cache_arm{arm_dim}_obs{obs_horizon}.json"
            
            if cache_stats_path.exists():
                # Priority 2: Use cached stats from previous run
                logging.info(f"Found cached statistics at: {cache_stats_path}")
                logging.info("  - Using cached stats (computed previously)")
                stats_file_to_use = str(cache_stats_path)
            else:
                # Priority 3: Compute on-the-fly and save to cache
                logging.info(f"Computing normalization statistics on-the-fly (UMI approach)")
                logging.info("  - Following UMI's get_normalizer() method")
                logging.info("  - Statistics will be computed from relative representations")
                logging.info(f"  - Results will be cached to: {cache_stats_path}")
                
                # Create temporary dataset wrapper with identity normalization for stats collection
                temp_dataset = RelativeActionDataset(
                    base_dataset=dataset,
                    arm_dim=arm_dim,
                    obs_horizon=obs_horizon,
                    skip_normalization=True,  # Use identity for data collection
                    stats_path=None,
                )
                
                # Compute statistics on-the-fly by iterating through dataset
                stats = temp_dataset.compute_normalizer_on_the_fly(
                    batch_size=cfg_obj.batch_size,
                    num_workers=cfg_obj.num_workers,
                )
                
                # Save computed stats to cache file for future use
                logging.info(f"Saving computed statistics to cache: {cache_stats_path}")
                cache_stats_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert numpy arrays to lists for JSON serialization
                stats_serializable = {}
                for key, stat_dict in stats.items():
                    stats_serializable[key] = {
                        stat_name: stat_value.tolist() if hasattr(stat_value, 'tolist') else stat_value
                        for stat_name, stat_value in stat_dict.items()
                    }
                
                # Add metadata
                cache_data = {
                    "metadata": {
                        "arm_dim": arm_dim,
                        "obs_horizon": obs_horizon,
                        "num_samples_processed": len(temp_dataset),
                        "dataset_info": {
                            "num_frames": temp_dataset.num_frames,
                            "num_episodes": temp_dataset.num_episodes,
                        },
                        "computed_on": str(Path(__file__).name),
                    },
                    "stats": stats_serializable,
                }
                
                with open(cache_stats_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                logging.info("  ✓ Statistics computed and saved to cache")
                logging.info("  ✓ This is the most principled approach (UMI Option 1)")
                
                # Use the cached file we just created
                stats_file_to_use = str(cache_stats_path)
        
        # Now create the final RelativeActionDataset wrapper ONCE with the determined stats
        dataset = RelativeActionDataset(
            base_dataset=dataset,
            arm_dim=arm_dim,
            obs_horizon=obs_horizon,
            skip_normalization=False,
            stats_path=stats_file_to_use,
        )
        logging.info("  ✓ RelativeActionDataset wrapper created with computed statistics")
    
    # Create environment for evaluation
    eval_env = None
    if cfg_obj.eval_freq > 0 and cfg_obj.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg_obj.env, n_envs=cfg_obj.eval.batch_size, use_async_envs=cfg_obj.eval.use_async_envs)
    
    # Create policy
    logging.info("Creating policy")
    
    # Use standard policy creation with additional kwargs for relative actions if needed
    policy_kwargs = {}
    if use_relative_actions:
        policy_kwargs["use_relative_actions"] = use_relative_actions
        policy_kwargs["arm_dim"] = arm_dim
    
    policy = make_policy(
        cfg=cfg_obj.policy,
        ds_meta=dataset.meta,
        **policy_kwargs,
    )
    
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg_obj, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg_obj.policy.use_amp)
    
    step = 0
    
    if cfg_obj.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg_obj.checkpoint_path, optimizer, lr_scheduler)
    
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg_obj.output_dir}")
    if cfg_obj.env is not None:
        logging.info(f"{cfg_obj.env.task=}")
    logging.info(f"{cfg_obj.steps=} ({format_big_number(cfg_obj.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # Create dataloader
    if hasattr(cfg_obj.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg_obj.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg_obj.num_workers,
        batch_size=cfg_obj.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    
    policy.train()
    
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        cfg_obj.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg_obj.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg_obj.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg_obj.policy.use_amp,
        )
        
        step += 1
        train_tracker.step()
        is_log_step = cfg_obj.log_freq > 0 and step % cfg_obj.log_freq == 0
        is_saving_step = step % cfg_obj.save_freq == 0 or step == cfg_obj.steps
        is_eval_step = cfg_obj.eval_freq > 0 and step % cfg_obj.eval_freq == 0
        
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
        
        if cfg_obj.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg_obj.output_dir, cfg_obj.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg_obj, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
        
        if cfg_obj.env and is_eval_step:
            step_id = get_step_identifier(step, cfg_obj.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg_obj.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg_obj.eval.n_episodes,
                    videos_dir=cfg_obj.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg_obj.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg_obj.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg_obj.policy.push_to_hub:
        policy.push_model_to_hub(cfg_obj)


def main():
    """Main entry point for training with relative actions."""
    args = parse_args()
    
    # Load configuration
    cfg = load_config_from_json(args.config_path)
    
    # Apply output_dir override
    cfg.output_dir = args.output_dir
    
    # Apply policy.push_to_hub override if specified
    if args.policy_push_to_hub is not None:
        if "policy" not in cfg:
            cfg.policy = {}
        cfg.policy.push_to_hub = args.policy_push_to_hub
    
    # Run training with relative action support
    train_with_relative_actions(
        cfg=cfg,
        use_relative_actions=args.use_relative_actions,
        arm_dim=args.arm_dim,
        obs_horizon=args.obs_horizon,
        relative_stats_path=args.relative_stats_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

