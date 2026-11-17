import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import robosuite as suite
from config import Config
from data_collection import get_action
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

class DaggerDataset(Dataset):
    def __init__(self, path):
        self.states, self.images, self.actions = [], [], []
        with h5py.File(path, "r") as f:
            num_demos = f.attrs["num_demos"]
            for i in range(num_demos):
                s = np.array(f[f"demo_{i}/states"])
                a = np.array(f[f"demo_{i}/actions"])
                image = np.array(f[f"demo_{i}/images"])
                self.states.append(s)
                self.actions.append(a)
                self.images.append(image)

        print(f"Loaded states, actions and images from {num_demos} demos.")

    def add(self, states, images, actions) -> None:
        self.states.append(states)
        self.actions.append(actions)
        self.images.append(images)

    def finalize(self) -> None:
        self.all_states = np.concatenate(self.states, axis=0).astype(np.float32)
        self.all_actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        self.all_images = np.concatenate(self.images, axis=0)   

    def __len__(self) -> int:
        return len(self.all_states)
    
    def __getitem__(self, idx):
        image = torch.Tensor(self.all_images[idx]).permute(2, 0, 1) / 255.0
        return self.all_states[idx], image, self.all_actions[idx]
    

class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, cnn_out_dim),
            nn.ReLU()
        )

    def forward(self, img):
        return self.encoder(img)


class BCPolicy(nn.Module):
    def __init__(self, state_dim, cnn_out_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.encoder = CNNEncoder(cnn_out_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.joint_pos_head = nn.Linear(hidden_dim, action_dim - 1)
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, img):
        img_features = self.encoder(img)
        input = torch.concatenate([state, img_features], dim=1)
        x = self.trunk(input)
        joint_pos = self.joint_pos_head(x)
        gripper_pos = self.gripper_head(x)
        return joint_pos, gripper_pos


def dagger_rollout(env, bc_policy, beta, num_episodes) -> tuple[list, list, list]:
    all_states, all_images, all_actions = [], [], []

    for _ in range(num_episodes):
        obs = env.reset()
        phase = "approach"
        states, images, actions = [], [], []
        for _ in range(cfg.max_steps):
            state = get_state(obs)
            image = np.array(obs[f"{cfg.camera}_image"])

            expert_action, phase = get_action(obs, phase)

            if np.random.rand() < beta:
                action = expert_action
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    img = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                    
                    pos, grip = bc_policy(s, img)
                    grip = 1.0 if grip.item() > 0 else -1.0
                    action = np.concatenate([pos.cpu().numpy().squeeze(), [grip]])

            states.append(state)
            images.append(image)
            actions.append(expert_action)

            obs, reward, done, info = env.step(action)
        all_states.append(np.array(states))
        all_images.append(np.array(images))
        all_actions.append(np.array(actions))

    return all_states, all_images, all_actions


def train_one_epoch(dataloader: DataLoader, bc_policy: BCPolicy, joint_loss_criterion, gripper_loss_criterion, optimizer, train_size) -> float:
    bc_policy.train()

    train_loss: float = 0.0

    for states, images, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        images = images.to(device)

        optimizer.zero_grad()

        pred_pos_logits, pred_gripper_logits = bc_policy(states, images)
        grip_target = (actions[:, 6] + 1) / 2   # shifting to 0-1
        joint_pos_loss = joint_loss_criterion(pred_pos_logits, actions[:, :6])
        gripper_loss = gripper_loss_criterion(pred_gripper_logits, grip_target.unsqueeze(1))
        loss = joint_pos_loss + 1.5 * gripper_loss
        loss.backward()

        optimizer.step()
        train_loss += loss.item() * len(states)
    
    train_loss = train_loss / train_size
    return train_loss


@torch.no_grad()
def env_eval(env, num_episodes, bc_model, policy_path):

    bc_model.load_state_dict(torch.load(policy_path, map_location=device))
    bc_model.eval()

    plt.ion()
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8))
    ax.axis("off")
    ax.set_title(f"{cfg.camera} view")

    for _ in range(num_episodes):
        obs = env.reset()
        for _ in range(cfg.max_steps):
            state = get_state(obs)
            image = obs[f"{cfg.camera}_image"]
            image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

            joint_pos, gripper = bc_model(state, image)
            gripper = torch.tensor([[1.0 if gripper.item() > 0 else -1.0]], device=device)
            action = torch.cat([joint_pos, gripper], dim=1).cpu().numpy().squeeze()
            obs, reward, done, info = env.step(action)
            env.render()

            img_display.set_data(obs[f"{cfg.camera}_image"])
            fig.canvas.flush_events()

    plt.ioff()
    plt.close(fig)
    env.close()


OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cube_pos"]
def get_state(obs):
    return np.concatenate([np.array(obs[k], dtype=np.float64).flatten() for k in OBS_KEYS])


#data loader
cfg = Config()
dataset = DaggerDataset(cfg.demo_data_path)
dataset.finalize()
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = dataset.all_states.shape[1]
action_dim = dataset.all_actions.shape[1]

bc_policy = BCPolicy(state_dim=state_dim, cnn_out_dim=cfg.cnn_out_dim, action_dim=action_dim).to(device)
joint_loss_criterion = nn.MSELoss()
gripper_loss_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(bc_policy.parameters(), lr=cfg.lr)


def main():
    os.makedirs(os.path.dirname(cfg.dagger_policy_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.dagger_plot_path), exist_ok=True)

    env = suite.make(cfg.task,
        robots=cfg.robot,
        has_renderer=cfg.renderer,
        has_offscreen_renderer=cfg.offscreen_renderer,
        use_camera_obs=cfg.camera_obs,
        camera_names = [cfg.camera],
        camera_heights=cfg.img_size, camera_widths=cfg.img_size,
        reward_shaping=True,
        control_freq=cfg.control_freq
    )

    train_losses = []
    best_loss = float("inf")

    for round in range(cfg.dagger_rounds):
        beta = cfg.beta_start - (cfg.beta_start - cfg.beta_end) * round/(cfg.dagger_rounds-1)
        states, images, actions = dagger_rollout(env, bc_policy, beta, cfg.dagger_ep_per_round)
        
        for s, i, a in zip(states, images, actions):
            dataset.add(s, i, a)
        dataset.finalize()
        
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        train_size = len(dataset)

        for epoch in range(cfg.dagger_num_epochs):
            train_loss = train_one_epoch(dataloader, bc_policy, joint_loss_criterion, gripper_loss_criterion, optimizer, train_size)

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(bc_policy.state_dict(), cfg.dagger_policy_path)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.dagger_num_epochs} | Train: {train_loss:.6f}")
            
            train_losses.append(train_loss)
        
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Dagger Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(cfg.dagger_plot_path, dpi=150)
        plt.close()
        print("Saved: dagger_policy_best.pth, dagger_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        main()
    else:
        env = suite.make(cfg.task,
            robots=cfg.robot,
            has_renderer=cfg.renderer,
            has_offscreen_renderer=cfg.offscreen_renderer,
            use_camera_obs=cfg.camera_obs,
            camera_names = [cfg.camera],
            camera_heights=cfg.img_size, camera_widths=cfg.img_size,
            reward_shaping=True,
            control_freq=cfg.control_freq
        )
        env_eval(env, 10, bc_policy, cfg.dagger_policy_path)