import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# The path to your generated dataset
repo_id = "emily_right_sim_approach"
dataset_path = Path(repo_id)

print(f"Loading dataset from: {dataset_path.absolute()}")

# Load the entire dataset
dataset = LeRobotDataset(repo_id=str(dataset_path))

print(f"Number of episodes: {dataset.num_episodes}")
print(f"Number of frames: {dataset.num_frames}")

# You can access metadata like this:
print(f"Frames per second: {dataset.meta.fps}")
print(f"Robot type: {dataset.meta.robot_type}")
print(f"Camera keys: {dataset.meta.camera_keys}")

# Get the first frame of the dataset
sample = dataset[0]

print("\nSample from the dataset:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: tensor of shape {value.shape} and dtype {value.dtype}")
    else:
        print(f"  {key}: {value}")

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