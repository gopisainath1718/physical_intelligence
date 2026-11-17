import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import robosuite as suite
import h5py
import matplotlib.pyplot as plt
from config import Config


class DemoDataset(Dataset):
    def __init__(self, path):
        self.states = []
        self.actions = []
        self.images = []
        with h5py.File(path, "r") as f:
            num_demos = f.attrs["num_demos"]
            for i in range(num_demos):
                s = np.array(f[f"demo_{i}/states"])
                a = np.array(f[f"demo_{i}/actions"])
                image = np.array(f[f"demo_{i}/images"])
                self.states.append(s)
                self.actions.append(a)
                self.images.append(image)
        
        self.states = np.concatenate(self.states, axis=0).astype(np.float32)
        self.actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        self.images = np.concatenate(self.images, axis=0)
        print(f"Loaded states, actions and images from {num_demos} demos.")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        img = torch.Tensor(self.images[idx]).permute(2, 0, 1) / 255.0   #reorder and normalize
        return self.states[idx], img, self.actions[idx]


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
    

def train_one_epoch(bc_model, train_dataloader, joint_loss_criterion, gripper_loss_criterion, optimizer, device):
    bc_model.train()

    total_loss = 0

    for states, images, actions in train_dataloader:
        states = states.to(device)
        actions = actions.to(device)
        images = images.to(device)

        optimizer.zero_grad()

        pred_pos_logits, pred_gripper_logits = bc_model(states, images)
        grip_target = (actions[:, 6] + 1) / 2   # shifting to 0-1
        joint_pos_loss = joint_loss_criterion(pred_pos_logits, actions[:, :6])
        gripper_loss = gripper_loss_criterion(pred_gripper_logits, grip_target.unsqueeze(1))
        loss = joint_pos_loss + 1.5 * gripper_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item() * len(states)
    
    train_loss = total_loss / train_size
    return train_loss



@torch.no_grad()
def eval(bc_model, val_dataloader, joint_loss_criterion, gripper_loss_criterion, device):
    bc_model.eval()

    total_loss = 0

    for states, images, actions in val_dataloader:
        states = states.to(device)
        actions = actions.to(device)
        images = images.to(device)
        
        pred_pos_logits, pred_gripper_logits = bc_model(states, images)
        grip_target = (actions[:, 6] + 1) / 2   # shifting to 0-1
        joint_pos_loss = joint_loss_criterion(pred_pos_logits, actions[:, :6])
        gripper_loss = gripper_loss_criterion(pred_gripper_logits, grip_target.unsqueeze(1))
        loss = joint_pos_loss + 1.5 * gripper_loss

        total_loss += loss.item() * len(states)
    
    val_loss = total_loss / val_size
    return val_loss

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

val_split = 0.1

dataset = DemoDataset(cfg.demo_data_path)
train_size = int(len(dataset) * (1 - val_split))
val_size = int(len(dataset) * val_split)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = dataset.states.shape[1]
action_dim = dataset.actions.shape[1]

bc_model = BCPolicy(state_dim=state_dim, cnn_out_dim=cfg.cnn_out_dim, action_dim=action_dim).to(device)
joint_loss_criterion = nn.MSELoss()
gripper_loss_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(bc_model.parameters(), lr=cfg.lr)

#Main Loop
def main():
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(bc_model, train_dataloader, joint_loss_criterion, gripper_loss_criterion, optimizer, device)
        val_loss = eval(bc_model, val_dataloader, joint_loss_criterion, gripper_loss_criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(bc_model.state_dict(), cfg.bc_policy_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("BC Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(cfg.bc_plot_path, dpi=150)
    plt.close()
    print(f"Best val loss: {best_val_loss:.6f}")
    print("Saved: bc_policy_best.pth, bc_loss.png")


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
        env_eval(env, 10, bc_model, cfg.bc_policy_path)