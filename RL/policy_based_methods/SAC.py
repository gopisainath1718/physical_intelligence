import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        img_shape: tuple[int, int, int] = (3, 84, 84),
        state_dim: int = 12,
        action_dim: int = 7,
    ):
        self.capacity = capacity
        self.ptr: int = 0
        self.size: int = 0

        self.state      = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.img        = np.zeros((self.capacity, *img_shape), dtype=np.uint8) 
        self.action     = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward     = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_img   = np.zeros((self.capacity, *img_shape), dtype=np.uint8)
        self.done      = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state: np.array, img: np.array, action: np.array, reward: float, next_state: np.array, next_img: np.array, done: bool):
        self.state[self.ptr]      = state
        self.img[self.ptr]        = img
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.next_img[self.ptr]   = next_img
        self.done[self.ptr]       = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size, self.capacity)
        
    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[idx]).to(device),
            torch.FloatTensor(self.img[idx]).to(device) / 255.0,
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.next_state[idx]).to(device),
            torch.FloatTensor(self.next_img[idx]).to(device) / 255.0,
            torch.FloatTensor(self.done[idx]).to(device),
        )

    def __len__(self):
        return len(self.size)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_features=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),     
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)


class Policy(nn.Module):
    pass


class SAC:
    pass


if __name__ == '__main__':
    pass