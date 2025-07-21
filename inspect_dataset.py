

# %%
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# The path to your generated dataset
# Using an absolute path is a good practice when loading local datasets
repo_id = "/home/shinfang-2f/jay_pick_two_arms_and_hands_sim_20250523_085738/IL/emily_right_sim_approach"
dataset_path = Path(repo_id)

print(f"Loading dataset from: {dataset_path.absolute()}")

# Load the entire dataset
# When loading a local dataset, the repo_id should be the path to the dataset directory.
dataset = LeRobotDataset(repo_id=str(dataset_path))

print("\n--- Custom Metadata ---")

# Access the metadata via the `meta.info` attribute
# This is a dictionary containing all the data from `meta/info.json`
metadata_info = dataset.meta.info

joint_names = metadata_info.get("joint_names")
action_min = metadata_info.get("action_min")
action_max = metadata_info.get("action_max")

if joint_names:
    print(f"joint_names: {joint_names}")
else:
    print("joint_names not found in metadata.")

if action_min:
    print(f"action_min: {action_min}")
else:
    print("action_min not found in metadata.")

if action_max:
    print(f"action_max: {action_max}")
else:
    print("action_max not found in metadata.") 

# %%
dataset.meta

# %%
# show metatdata

# Show dataset metadata
print("Dataset metadata:")
print(f"  Number of episodes: {dataset.num_episodes}")
print(f"  Number of frames: {dataset.num_frames}")
print(f"  Frames per second: {dataset.meta.fps}")
print(f"  Robot type: {dataset.meta.robot_type}")
print(f"  Camera keys: {dataset.meta.camera_keys}")


# %%
print(f"Number of episodes: {dataset.num_episodes}")
print(f"Number of frames: {dataset.num_frames}")

# You can access metadata like this:
print(f"Frames per second: {dataset.meta.fps}")
print(f"Robot type: {dataset.meta.robot_type}")
print(f"Camera keys: {dataset.meta.camera_keys}")

# %%
sample = dataset[0]

print("\nSample from the dataset:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: tensor of shape {value.shape} and dtype {value.dtype}")
    else:
        print(f"  {key}: {value}")


# %%




# Get the first frame of the dataset

# You can also use it with a PyTorch DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

print("\nTesting with DataLoader:")
for batch in dataloader:
    print(f"Batch observation.image shape: {batch['observation.image'].shape}")
    print(f"Batch observation.state shape: {batch['observation.state'].shape}")
    print(f"Batch action shape: {batch['action'].shape}")
    break 


