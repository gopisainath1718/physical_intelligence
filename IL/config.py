from dataclasses import dataclass

@dataclass
class Config:
    #env
    task: str = "Lift"
    robot: str = "Panda"
    camera: str = "agentview"
    renderer: bool = True
    offscreen_renderer: bool = True
    camera_obs: bool = True
    img_size: int = 84
    control_freq: int = 20

    #data
    num_demos: int = 50
    max_steps: int = 120
    demo_data_path: str = "results/demo_data/demos_50_camera.hdf5"

    #model
    state_dim: int = 12
    action_dim: int = 7
    cnn_out_dim: int = 64

    #training
    batch_size: int = 256
    lr: float = 3e-4
    num_epochs: int = 100

    #bc
    bc_policy_path: str = "results/bc/models/policy_best_300_camera.pth"
    bc_plot_path: str = "results/bc/plots/loss_300_camera.png"

    #dagger
    dagger_rounds: int = 10
    dagger_ep_per_round: int = 25
    beta_start: float = 1.0
    beta_end: float = 0.0
    dagger_num_epochs: int = 50
    dagger_policy_path: str = "results/dagger/models/policy_best_300_camera.pth"
    dagger_plot_path: str = "results/dagger/plots/loss_300_camera.png"
