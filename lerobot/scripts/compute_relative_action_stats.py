#!/usr/bin/env python

"""
Compute normalization statistics for relative action representation.

This script computes normalization statistics (min, max, mean, std) from relative
action and observation representations, following UMI's approach. The statistics
are computed AFTER converting to relative, which is important because:

1. Relative values have different distributions than absolute values
2. Using absolute stats on relative values causes poor normalization
3. UMI computes stats from relative representations (Option 1)

Usage:
    python lerobot/scripts/compute_relative_action_stats.py \
        --config_path=/path/to/train_config.json \
        --output_path=/path/to/relative_stats.json \
        --arm_dim=6 \
        --num_samples=10000

The output JSON file contains statistics that can be loaded into the dataset
for proper normalization during training with relative actions.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils.relative_actions import (
    convert_actions_to_relative,
    convert_observation_state_to_relative,
    get_current_arm_state,
)
from lerobot.scripts.train import TrainPipelineConfig
import draccus


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for relative action representation"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the computed statistics (JSON format)"
    )
    parser.add_argument(
        "--arm_dim",
        type=int,
        default=6,
        help="Number of arm joints (excluding gripper). Default: 6"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for computing statistics. Default: use all data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading. Default: 32"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> TrainPipelineConfig:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to OmegaConf then to TrainPipelineConfig
    cfg = OmegaConf.create(config_dict)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    with draccus.config_type("json"):
        cfg_obj = draccus.decode(TrainPipelineConfig, cfg_dict)
    
    return cfg_obj


def compute_stats_from_data(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute normalization statistics from data.
    
    Args:
        data: numpy array of shape (N, D) where N is number of samples
        
    Returns:
        Dictionary with 'min', 'max', 'mean', 'std' keys
    """
    return {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
    }


def collect_relative_data(
    dataset,
    arm_dim: int,
    obs_horizon: int,
    num_samples: int = None,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Collect relative action and observation data from the dataset.

    This function iterates through the dataset, converts entire batches to relative
    representation, and collects all the relative values for statistics computation.

    Args:
        dataset: LeRobotDataset instance
        arm_dim: Number of arm joints (excluding gripper)
        obs_horizon: Observation horizon (n_obs_steps)
        num_samples: Maximum number of samples to process (None = all)
        batch_size: Batch size for data loading

    Returns:
        Dictionary with 'action' and 'observation.state' keys containing
        numpy arrays of all collected relative values
    """
    logging.info("Collecting relative action and observation data...")

    # Determine number of samples to process
    total_samples = len(dataset)
    if num_samples is not None:
        total_samples = min(num_samples, total_samples)

    # Storage for collected data (will be concatenated at the end)
    all_relative_actions = []
    all_relative_obs_states = []

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    samples_processed = 0

    for batch in tqdm(dataloader, desc="Processing batches"):
        # Check if observation.state exists
        if "observation.state" not in batch:
            logging.warning("No observation.state in batch, skipping...")
            continue

        # Get batch data
        obs_state = batch["observation.state"]  # (B, obs_horizon, state_dim)
        actions = batch["action"]  # (B, horizon, action_dim)

        batch_size_actual = obs_state.shape[0]

        # Check if we would exceed total_samples with this batch
        remaining_samples = total_samples - samples_processed
        if remaining_samples <= 0:
            break

        # If this batch would exceed total_samples, truncate it
        if samples_processed + batch_size_actual > total_samples:
            actual_batch_size = remaining_samples
            obs_state = obs_state[:actual_batch_size]
            actions = actions[:actual_batch_size]
            batch_size_actual = actual_batch_size

        # Process entire batch at once (batch-wise processing)
        # Extract current arm states for all samples in batch
        current_arm_states = get_current_arm_state(obs_state, arm_dim=arm_dim)  # (B, arm_dim)

        # Convert entire batch of observation states to relative representation
        relative_obs_batch = convert_observation_state_to_relative(
            obs_state,  # (B, obs_horizon, state_dim)
            arm_dim=arm_dim,
        )  # (B, obs_horizon, state_dim)

        # Convert entire batch of actions to relative representation
        relative_actions_batch = convert_actions_to_relative(
            actions,  # (B, horizon, action_dim)
            current_arm_states,  # (B, arm_dim)
            obs_horizon=obs_horizon,
            arm_dim=arm_dim,
        )  # (B, horizon, action_dim)

        # Flatten temporal dimensions and store
        # For obs: (B, obs_horizon, state_dim) -> (B*obs_horizon, state_dim)
        relative_obs_batch_flat = relative_obs_batch.reshape(-1, relative_obs_batch.shape[-1])
        all_relative_obs_states.append(relative_obs_batch_flat.cpu().numpy())

        # For actions: (B, horizon, action_dim) -> (B*horizon, action_dim)
        relative_actions_batch_flat = relative_actions_batch.reshape(-1, relative_actions_batch.shape[-1])
        all_relative_actions.append(relative_actions_batch_flat.cpu().numpy())

        samples_processed += batch_size_actual

    logging.info(f"Processed {samples_processed} samples")

    # Concatenate all collected data
    if all_relative_actions:
        relative_actions = np.concatenate(all_relative_actions, axis=0)  # (N*horizon, action_dim)
        relative_obs_states = np.concatenate(all_relative_obs_states, axis=0)  # (N*obs_horizon, state_dim)

        logging.info(f"Collected relative actions shape: {relative_actions.shape}")
        logging.info(f"Collected relative obs states shape: {relative_obs_states.shape}")
    else:
        # Handle edge case where no data was collected
        logging.warning("No relative data was collected!")
        relative_actions = np.array([])
        relative_obs_states = np.array([])

    return {
        'action': relative_actions,
        'observation.state': relative_obs_states,
    }


def main():
    """Main function to compute relative action statistics."""
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load configuration
    logging.info(f"Loading configuration from {args.config_path}")
    cfg = load_config(args.config_path)
    
    # Get obs_horizon from config
    obs_horizon = cfg.policy.n_obs_steps
    logging.info(f"Using obs_horizon={obs_horizon} from config")
    
    # Create dataset (absolute representation)
    logging.info("Creating dataset...")
    dataset = make_dataset(cfg)
    
    logging.info("="*80)
    logging.info(f"Dataset info:")
    logging.info(f"  - Total frames: {dataset.num_frames}")
    logging.info(f"  - Total episodes: {dataset.num_episodes}")
    logging.info(f"  - Total samples: {len(dataset)}")
    logging.info(f"  - Arm dimension: {args.arm_dim}")
    logging.info(f"  - Observation horizon: {obs_horizon}")
    logging.info("="*80)
    
    # Collect relative data
    relative_data = collect_relative_data(
        dataset=dataset,
        arm_dim=args.arm_dim,
        obs_horizon=obs_horizon,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    
    # Compute statistics for each modality
    logging.info("Computing statistics from relative representations...")
    stats = {}
    
    for key, data in relative_data.items():
        logging.info(f"Computing stats for {key} (shape: {data.shape})")
        stats[key] = compute_stats_from_data(data)
        
        # Log statistics summary
        logging.info(f"  {key} statistics:")
        logging.info(f"    - min: {stats[key]['min'][:5]}... (showing first 5)")
        logging.info(f"    - max: {stats[key]['max'][:5]}...")
        logging.info(f"    - mean: {stats[key]['mean'][:5]}...")
        logging.info(f"    - std: {stats[key]['std'][:5]}...")
    
    # Convert numpy arrays to lists for JSON serialization
    stats_serializable = {}
    for key, stat_dict in stats.items():
        stats_serializable[key] = {
            stat_name: stat_value.tolist()
            for stat_name, stat_value in stat_dict.items()
        }
    
    # Add metadata
    output_data = {
        "metadata": {
            "config_path": args.config_path,
            "arm_dim": args.arm_dim,
            "obs_horizon": obs_horizon,
            "num_samples_processed": len(relative_data['action']),
            "dataset_info": {
                "num_frames": dataset.num_frames,
                "num_episodes": dataset.num_episodes,
            },
        },
        "stats": stats_serializable,
    }
    
    # Save to JSON file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info("="*80)
    logging.info(f"Statistics saved to: {output_path}")
    logging.info("="*80)
    logging.info("Summary of computed statistics:")
    logging.info("  - These statistics should be used for normalizing relative actions/states")
    logging.info("  - Statistics computed AFTER converting to relative representation")
    logging.info("  - Following UMI's approach (Option 1)")
    logging.info("="*80)


if __name__ == "__main__":
    main()