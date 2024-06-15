from typing import Tuple

import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int], device: torch.device):
        self.device = device
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=capacity, device=device)
        )

    def add(self, state, action, reward, next_state, done):
        experience = TensorDict({
            "state": torch.tensor(state, dtype=torch.uint8, device=self.device),
            "action": torch.tensor([action], device=self.device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "next_state": torch.tensor(next_state, dtype=torch.float32, device=self.device),
            "done": torch.tensor([done], dtype=torch.float32, device=self.device),
        })
        self.buffer.extend(experience)

    def sample(self, batch_size: int):
        sample = self.buffer.sample(batch_size)
        return (
            sample["state"],
            sample["action"].squeeze(-1),  # Remove the extra dimension
            sample["reward"].squeeze(-1),
            sample["next_state"],
            sample["done"].squeeze(-1)
        )

    def __len__(self):
        return len(self.buffer)

# Testing the ReplayBuffer
if __name__ == "__main__":
    state_shape = (128, 128, 4)  # Example state shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer = SimpleReplayBuffer(capacity=10000, state_shape=state_shape, device=device)

    # Add a dummy experience to test
    state = np.random.randint(0, 255, state_shape, dtype=np.uint8)
    action = 2
    reward = 1.0
    next_state = np.random.randint(0, 255, state_shape, dtype=np.uint8)
    done = False

    buffer.add(state, action, reward, next_state, done)

    # Sample a batch
    batch_size = 4
    batch = buffer.sample(batch_size)

    print(f"States batch shape: {batch[0].shape}")
    print(f"Actions batch shape: {batch[1].shape}")
    print(f"Rewards batch shape: {batch[2].shape}")
    print(f"Next states batch shape: {batch[3].shape}")
    print(f"Dones batch shape: {batch[4].shape}")