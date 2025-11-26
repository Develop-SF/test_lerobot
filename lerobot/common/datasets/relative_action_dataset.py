"""
Dataset wrapper for relative action training.

Wraps LeRobotDataset to convert actions and observations to relative
representations on-the-fly during training, following UMI and ManiSkill principles.

IMPORTANT: This wrapper converts to relative BEFORE normalization happens in the policy.
The policy can use either:
1. Identity normalization (skip_normalization=True) - simpler, values already small
2. Computed statistics from relative representations (stats_path) - more principled

Following UMI's approach, you can compute proper statistics using:
    python lerobot/scripts/compute_relative_action_stats.py
"""

import torch
import json
import copy
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.relative_actions import (
    convert_actions_to_relative,
    convert_observation_state_to_relative,
    get_current_arm_state,
)


class RelativeActionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that converts absolute actions and observations to relative on-the-fly.
    
    This implements:
    - PD2.1: Relative trajectory as action representation
    - PD2.2: Relative trajectory as proprioception (joint state observations)
    
    CRITICAL: Conversion happens BEFORE normalization. The policy should either:
    1. Use identity normalization for states/actions (recommended)
    2. OR have statistics recomputed from relative representations
    
    Usage:
        dataset = LeRobotDataset(...)
        relative_dataset = RelativeActionDataset(
            dataset, 
            arm_dim=6, 
            obs_horizon=2,
            skip_normalization=True  # Recommended for relative mode
        )
    """
    
    def __init__(
        self,
        base_dataset: LeRobotDataset,
        arm_dim: int = 6,
        obs_horizon: int = 2,
        skip_normalization: bool = True,
        stats_path: Optional[str] = None,
    ):
        """
        Initialize the relative action dataset wrapper.
        
        Args:
            base_dataset: The underlying LeRobotDataset with absolute actions
            arm_dim: Number of arm joints (excluding gripper)
            obs_horizon: Number of observation steps
            skip_normalization: If True, modifies stats to use identity normalization
                for state and action. This is simpler but less principled.
                Ignored if stats_path is provided.
            stats_path: Path to JSON file with pre-computed statistics from relative
                representations. If provided, uses these statistics instead of
                identity normalization. Compute using:
                    python lerobot/scripts/compute_relative_action_stats.py
        """
        self.base_dataset = base_dataset
        self.arm_dim = arm_dim
        self.obs_horizon = obs_horizon
        self.skip_normalization = skip_normalization
        self.stats_path = stats_path
        
        # Expose important attributes from base dataset
        self.meta = base_dataset.meta
        self.hf_dataset = base_dataset.hf_dataset
        self.repo_id = base_dataset.repo_id
        
        # Setup normalization statistics
        if stats_path is not None:
            self._setup_computed_normalization(stats_path)
        elif skip_normalization:
            self._setup_identity_normalization()
    
    def _setup_computed_normalization(self, stats_path: str):
        """
        Load and apply pre-computed normalization statistics from relative representations.
        
        This is the most principled approach (Option 1): statistics computed AFTER
        converting to relative representation, following UMI's methodology.
        
        Args:
            stats_path: Path to JSON file with computed statistics
        """
        print(f"[RelativeActionDataset] Loading computed statistics from {stats_path}")
        
        # Load statistics from file
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
        
        # Verify metadata matches
        metadata = stats_data.get("metadata", {})
        if metadata.get("arm_dim") != self.arm_dim:
            print(f"WARNING: Loaded stats have arm_dim={metadata.get('arm_dim')}, "
                  f"but current arm_dim={self.arm_dim}")
        if metadata.get("obs_horizon") != self.obs_horizon:
            print(f"WARNING: Loaded stats have obs_horizon={metadata.get('obs_horizon')}, "
                  f"but current obs_horizon={self.obs_horizon}")
        
        stats = stats_data["stats"]
        
        # Convert loaded statistics to torch tensors
        for key in stats:
            if key == "action" and key in self.meta.stats:
                self.meta.stats["action"] = {
                    stat_name: torch.tensor(stat_value, dtype=torch.float32)
                    for stat_name, stat_value in stats[key].items()
                }
            elif key == "observation.state" and key in self.meta.stats:
                self.meta.stats["observation.state"] = {
                    stat_name: torch.tensor(stat_value, dtype=torch.float32)
                    for stat_name, stat_value in stats[key].items()
                }
        
        print(f"  ✓ Loaded statistics computed from {metadata.get('num_samples_processed', 'N/A')} relative samples")
        print(f"  ✓ Using statistics computed AFTER relative conversion (UMI Option 1)")
        print(f"  ✓ This is more principled than identity normalization")
    
    def _setup_identity_normalization(self):
        """
        Replace normalization statistics with identity (scale=1, offset=0).
        
        This is simpler but less principled than computing stats from relative
        representations. It works because:
        - Relative actions/states are already small (deltas around 0)
        - Identity normalization doesn't distort the distribution
        - UMI uses this approach for rotations
        
        For the most principled approach, use stats_path instead.
        """
        # Modify action statistics
        if "action" in self.meta.stats:
            action_shape = self.meta.stats["action"]["max"].shape
            self.meta.stats["action"] = {
                "max": torch.ones(action_shape),
                "min": -torch.ones(action_shape),
                "mean": torch.zeros(action_shape),
                "std": torch.ones(action_shape),
            }
        
        # Modify state statistics if present
        if "observation.state" in self.meta.stats:
            state_shape = self.meta.stats["observation.state"]["max"].shape
            self.meta.stats["observation.state"] = {
                "max": torch.ones(state_shape),
                "min": -torch.ones(state_shape),
                "mean": torch.zeros(state_shape),
                "std": torch.ones(state_shape),
            }
        
        print(f"[RelativeActionDataset] Using identity normalization for relative mode")
        print(f"  - Actions: scale=1, offset=0")
        print(f"  - States: scale=1, offset=0")
        print(f"  - Reason: Simpler approach, relative values already small")
        print(f"  - For most principled approach, compute stats with:")
        print(f"    python lerobot/scripts/compute_relative_action_stats.py")
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample and convert to relative representation.
        
        Conversion order (following UMI):
        1. Load absolute data from base dataset
        2. Convert to relative (this method)
        3. Normalization happens later in the policy (with identity stats if skip_normalization=True)
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with relative actions and observations
        """
        # Get the original sample (absolute values)
        sample = self.base_dataset[idx]
        
        # Check if observation.state exists
        if "observation.state" not in sample:
            # No state observations, cannot do relative conversion
            return sample
        
        # Extract current arm state (absolute) before any conversion
        # This is used as reference for action conversion
        obs_state = sample["observation.state"]
        current_arm_state = get_current_arm_state(obs_state, arm_dim=self.arm_dim)
        
        # Convert observations to relative (PD2.2)
        # Historical positions relative to current position
        sample["observation.state"] = convert_observation_state_to_relative(
            obs_state, arm_dim=self.arm_dim
        )
        
        # Convert actions to relative (PD2.1)
        # Actions as deltas from current position
        sample["action"] = convert_actions_to_relative(
            sample["action"],
            current_arm_state,
            obs_horizon=self.obs_horizon,
            arm_dim=self.arm_dim,
        )
        
        return sample
    
    def compute_normalizer_on_the_fly(self, batch_size: int = 64, num_workers: int = 4):
        """
        Compute normalization statistics on-the-fly by iterating through the dataset.
        
        This follows UMI's approach (get_normalizer method):
        1. Create a DataLoader to iterate through the dataset
        2. For each batch, __getitem__() is called which converts to relative
        3. Collect the already-relative data
        4. Compute statistics from the collected relative data
        
        Args:
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            
        Returns:
            Dictionary with computed statistics for 'action' and 'observation.state'
        """
        print(f"[RelativeActionDataset] Computing normalization statistics on-the-fly...")
        print(f"  - Iterating through dataset to collect relative representations")
        print(f"  - Following UMI's approach: stats computed AFTER conversion")
        
        # Cache for collecting data (similar to UMI's data_cache)
        data_cache = {
            'action': [],
            'observation.state': []
        }
        
        # Create dataloader (similar to UMI lines 199-203)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        
        # Iterate through dataset and collect relative data (UMI lines 204-207)
        for batch in tqdm(dataloader, desc='Iterating dataset to get normalization'):
            # At this point, batch contains data that has already been converted to relative
            # by __getitem__(), so we just collect it
            if 'observation.state' in batch:
                data_cache['observation.state'].append(copy.deepcopy(batch['observation.state']))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        
        # Concatenate collected data
        stats = {}
        for key in data_cache.keys():
            if len(data_cache[key]) == 0:
                continue
                
            # Concatenate all batches
            data_array = torch.cat(data_cache[key], dim=0)  # (N, obs_horizon or action_horizon, dim)
            
            # Convert to numpy for statistics computation
            data_np = data_array.cpu().numpy()
            
            # Flatten temporal dimension (N, T, D) -> (N*T, D)
            # This follows UMI's approach (line 216)
            N, T, D = data_np.shape
            data_flat = data_np.reshape(N * T, D)
            
            # Compute statistics (following UMI's array_to_stats approach)
            stats[key] = {
                'min': np.min(data_flat, axis=0),
                'max': np.max(data_flat, axis=0),
                'mean': np.mean(data_flat, axis=0),
                'std': np.std(data_flat, axis=0),
            }
            
            print(f"  ✓ Computed stats for {key} from {data_flat.shape[0]} samples")
        
        return stats
    
    def apply_computed_stats(self, stats: Dict[str, Dict[str, np.ndarray]]):
        """
        Apply computed statistics to the dataset's meta.stats.
        
        Args:
            stats: Dictionary with statistics for each key
        """
        # Convert numpy arrays to torch tensors and update meta.stats
        for key in stats:
            if key == "action" and key in self.meta.stats:
                self.meta.stats["action"] = {
                    stat_name: torch.tensor(stat_value, dtype=torch.float32)
                    for stat_name, stat_value in stats[key].items()
                }
            elif key == "observation.state" and key in self.meta.stats:
                self.meta.stats["observation.state"] = {
                    stat_name: torch.tensor(stat_value, dtype=torch.float32)
                    for stat_name, stat_value in stats[key].items()
                }
        
        print(f"[RelativeActionDataset] Applied computed statistics to dataset")
        print(f"  ✓ Statistics computed from relative representations (UMI Option 1)")
    
    # Expose other useful methods from base dataset
    @property
    def fps(self) -> int:
        return self.base_dataset.fps
    
    @property
    def features(self) -> dict:
        return self.base_dataset.features
    
    @property
    def stats(self) -> dict:
        """Return modified stats (identity normalization for relative mode)."""
        return self.base_dataset.meta.stats
    
    @property
    def num_frames(self) -> int:
        return self.base_dataset.num_frames
    
    @property
    def num_episodes(self) -> int:
        return self.base_dataset.num_episodes
    
    @property
    def episode_data_index(self) -> dict:
        """Return episode data index from base dataset (needed for EpisodeAwareSampler)."""
        return self.base_dataset.episode_data_index