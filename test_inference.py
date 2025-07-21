import h5py
import numpy as np
import torch
import torchvision
import imageio
import numpy
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


dataset_path = "/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/Pick_Right_Arm_and_Hand_Sim_2025-06-12_trimmed.hdf5"

device = "cuda"





pretrained_policy_path = "/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/lerobot/approach_DP/output/checkpoints/last/pretrained_model"


policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)


policy.reset()


with h5py.File(dataset_path, "r") as f:
    demo_key = list(f["data"].keys())[0]
    demo_group = f["data"][demo_key]
    obs_group = demo_group["obs"]
    obs_dict = {"qpos": obs_group["qpos"][0], "qvel": obs_group["qvel"][0]}
    obs_dict["images"] = obs_group["images"][0]


state = torch.from_numpy(np.concatenate([obs_dict['qpos'], obs_dict['qvel']]))
image = torch.from_numpy(obs_dict["images"])

# Convert to float32 with image from channel first in [0,255]
# to channel last in [0,1]
state = state.to(torch.float32)
image = image.to(torch.float32) / 255
image = image.permute(2, 0, 1)

# resize image to 350x350 with torchvision
image = torchvision.transforms.Resize(350)(image)
# # Send data tensors from CPU to GPU
state = state.to(device, non_blocking=True)
image = image.to(device, non_blocking=True)

state = state.unsqueeze(0)
image = image.unsqueeze(0)

# Create the policy input dictionary
observation = {
    "observation.state": state,
    "observation.image": image,
}

# Predict the next action with respect to the current observation
with torch.inference_mode():
    action = policy.select_action(observation)