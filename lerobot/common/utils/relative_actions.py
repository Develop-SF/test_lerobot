"""
Utilities for relative action representation in lerobot.

Implements relative action conversion following UMI and ManiSkill design principles:
- PD2.1: Relative trajectory as action representation
- PD2.2: Relative trajectory as proprioception (state observations)

For joint space control:
- Actions and observations are relative to current joint position
- Only arm joints are made relative; gripper joints stay absolute (if present)
"""

import torch
import numpy as np
from typing import Union, Dict


def convert_actions_to_relative(
    actions: Union[torch.Tensor, np.ndarray],
    current_state: Union[torch.Tensor, np.ndarray],
    obs_horizon: int = 2,
    arm_dim: int = 6,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert absolute actions to relative actions (PD2.1).
    
    The reference point is the action at timestep (obs_horizon - 1), which corresponds
    to the "current" action position that aligns with the last observation.
    
    Args:
        actions: Absolute actions of shape (B, horizon, action_dim) or (horizon, action_dim)
        current_state: Current arm joint state of shape (B, arm_dim) or (arm_dim,)
        obs_horizon: Number of observation steps
        arm_dim: Number of arm joints (excluding gripper)
        
    Returns:
        Relative actions of the same shape as input, where only arm joints are relative
    """
    is_numpy = isinstance(actions, np.ndarray)
    if is_numpy:
        actions = torch.from_numpy(actions)
        current_state = torch.from_numpy(current_state)
    
    # Make a copy to avoid modifying the input
    relative_actions = actions.clone()
    
    # Handle both batched and unbatched inputs
    if actions.ndim == 3:  # (B, horizon, action_dim)
        # Reference is the current state, expanded to match action dimensions
        reference = current_state.unsqueeze(1)  # (B, 1, arm_dim)
        # Convert only arm joints to relative
        relative_actions[:, :, :arm_dim] = actions[:, :, :arm_dim] - reference
    else:  # (horizon, action_dim)
        # Reference is the current state
        reference = current_state.unsqueeze(0)  # (1, arm_dim)
        # Convert only arm joints to relative
        relative_actions[:, :arm_dim] = actions[:, :arm_dim] - reference
    
    if is_numpy:
        return relative_actions.numpy()
    return relative_actions


def convert_actions_to_absolute(
    relative_actions: Union[torch.Tensor, np.ndarray],
    current_state: Union[torch.Tensor, np.ndarray],
    obs_horizon: int = 2,
    arm_dim: int = 6,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert relative actions back to absolute actions (inverse of PD2.1).
    
    Used during inference to convert predicted relative actions to executable absolute actions.
    
    Args:
        relative_actions: Relative actions of shape (B, horizon, action_dim) or (horizon, action_dim)
        current_state: Current arm joint state of shape (B, arm_dim) or (arm_dim,)
        obs_horizon: Number of observation steps
        arm_dim: Number of arm joints (excluding gripper)
        
    Returns:
        Absolute actions of the same shape as input, where only arm joints are converted
    """
    is_numpy = isinstance(relative_actions, np.ndarray)
    if is_numpy:
        relative_actions = torch.from_numpy(relative_actions)
        current_state = torch.from_numpy(current_state)
    
    # Make a copy to avoid modifying the input
    absolute_actions = relative_actions.clone()
    
    # Handle both batched and unbatched inputs
    if relative_actions.ndim == 3:  # (B, horizon, action_dim)
        # Reference is the current state, expanded to match action dimensions
        reference = current_state.unsqueeze(1)  # (B, 1, arm_dim)
        # Convert only arm joints to absolute
        absolute_actions[:, :, :arm_dim] = relative_actions[:, :, :arm_dim] + reference
    else:  # (horizon, action_dim)
        # Reference is the current state
        reference = current_state.unsqueeze(0)  # (1, arm_dim)
        # Convert only arm joints to absolute
        absolute_actions[:, :arm_dim] = relative_actions[:, :arm_dim] + reference
    
    if is_numpy:
        return absolute_actions.numpy()
    return absolute_actions


def convert_observation_state_to_relative(
    obs_state: Union[torch.Tensor, np.ndarray],
    arm_dim: int = 6,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert absolute observation states to relative trajectory (PD2.2).
    
    Following UMI paper: "represent proprioception of history poses as relative trajectory"
    For joint state: all historical joint positions are relative to CURRENT joint position.
    
    Args:
        obs_state: Absolute states of shape (B, obs_horizon, state_dim) or (obs_horizon, state_dim)
        arm_dim: Number of arm joints (excluding gripper)
        
    Returns:
        Relative observation states of the same shape, where arm joints are relative to current
    """
    is_numpy = isinstance(obs_state, np.ndarray)
    if is_numpy:
        obs_state = torch.from_numpy(obs_state)
    
    # Make a copy to avoid modifying the input
    relative_obs = obs_state.clone()
    
    # Handle both batched and unbatched inputs
    if obs_state.ndim == 3:  # (B, obs_horizon, state_dim)
        # Reference is the current joint state (last in sequence)
        current_state = obs_state[:, -1:, :arm_dim]  # (B, 1, arm_dim)
        # Convert arm joint positions to relative
        relative_obs[:, :, :arm_dim] = obs_state[:, :, :arm_dim] - current_state
    else:  # (obs_horizon, state_dim)
        # Reference is the current joint state (last in sequence)
        current_state = obs_state[-1:, :arm_dim]  # (1, arm_dim)
        # Convert arm joint positions to relative
        relative_obs[:, :arm_dim] = obs_state[:, :arm_dim] - current_state
    
    if is_numpy:
        return relative_obs.numpy()
    return relative_obs


def get_current_arm_state(
    obs_state: Union[torch.Tensor, np.ndarray],
    arm_dim: int = 6,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract the current arm state from observation state.
    
    Args:
        obs_state: Observation states of shape (B, obs_horizon, state_dim) or (obs_horizon, state_dim)
        arm_dim: Number of arm joints
        
    Returns:
        Current arm state of shape (B, arm_dim) or (arm_dim,)
    """
    is_numpy = isinstance(obs_state, np.ndarray)
    if is_numpy:
        obs_state = torch.from_numpy(obs_state)
    
    # Handle both batched and unbatched inputs
    if obs_state.ndim == 3:  # (B, obs_horizon, state_dim)
        current_state = obs_state[:, -1, :arm_dim]  # (B, arm_dim)
    else:  # (obs_horizon, state_dim)
        current_state = obs_state[-1, :arm_dim]  # (arm_dim,)
    
    if is_numpy:
        return current_state.numpy()
    return current_state


# Batch processing functions for dataset iteration
def process_batch_for_relative_training(
    batch: Dict[str, torch.Tensor],
    arm_dim: int = 6,
    obs_horizon: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Process a training batch to convert to relative actions and observations.
    
    This function is called during training data loading to convert absolute
    demonstrations to relative representations on-the-fly.
    
    Args:
        batch: Dictionary containing 'observation.state' and 'action' keys
        arm_dim: Number of arm joints
        obs_horizon: Number of observation steps
        
    Returns:
        Modified batch with relative actions and observations
    """
    batch = dict(batch)  # Shallow copy to avoid modifying the original
    
    # Convert observations to relative (PD2.2)
    if "observation.state" in batch:
        batch["observation.state"] = convert_observation_state_to_relative(
            batch["observation.state"], arm_dim=arm_dim
        )
    
    # Get current arm state for action conversion (before observation conversion)
    # We need the absolute current state as reference
    current_state = get_current_arm_state(batch["observation.state"], arm_dim=arm_dim)
    
    # Convert actions to relative (PD2.1)
    batch["action"] = convert_actions_to_relative(
        batch["action"], 
        current_state, 
        obs_horizon=obs_horizon,
        arm_dim=arm_dim
    )
    
    return batch


def process_observation_for_relative_inference(
    observation: Dict[str, Union[torch.Tensor, np.ndarray]],
    arm_dim: int = 6,
) -> tuple[Dict[str, Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
    """
    Process an observation during inference to convert to relative representation.
    
    Args:
        observation: Dictionary containing observation keys (e.g., 'observation.state')
        arm_dim: Number of arm joints
        
    Returns:
        Tuple of (modified_observation, current_arm_state) where:
        - modified_observation: Observation with relative states
        - current_arm_state: Absolute current arm state for action conversion
    """
    observation = dict(observation)  # Shallow copy
    
    # Extract current arm state before conversion
    if "observation.state" in observation:
        current_state = get_current_arm_state(observation["observation.state"], arm_dim=arm_dim)
        
        # Convert observations to relative (PD2.2)
        observation["observation.state"] = convert_observation_state_to_relative(
            observation["observation.state"], arm_dim=arm_dim
        )
    else:
        # No state observation, cannot do relative actions
        current_state = None
    
    return observation, current_state

