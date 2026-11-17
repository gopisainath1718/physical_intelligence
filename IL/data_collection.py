import robosuite as suite
import numpy as np
import h5py
from config import Config


def get_action(obs, phase: str) -> tuple[np.array, str]:
	ee_pos = obs["robot0_eef_pos"]
	cube_pos = obs["cube_pos"]
	gripper_action = -1.0

	if phase == "approach":
		target = cube_pos + np.array([0, 0, 0.1])  # above cube
		if np.linalg.norm(ee_pos - target) < 0.02:
			phase = "descend"
	elif phase == "descend":
		target = cube_pos + np.array([0, 0, 0.005])
		if np.linalg.norm(ee_pos - target) < 0.01:
			phase = "grasp"
	elif phase == "grasp":
		target = cube_pos + np.array([0, 0, 0.005])
		gripper_action = 1.0  # close
		if obs["robot0_gripper_qpos"][0] <= 0.025:
			phase = "lift"
	elif phase == "lift":
		target = cube_pos + np.array([0, 0, 0.2])
		gripper_action = 1.0  # keep closed

	action = np.zeros(7)
	action[:3] = (target - ee_pos) * 5	# kp
	action[:3] = np.clip(action[:3], -1, 1)
	action[6] = gripper_action
	
	return action, phase	
	


def collect_demos() -> None:
	cfg = Config()
	env = suite.make(
		cfg.task,
		robots=cfg.robot,
		has_renderer=cfg.renderer,
		has_offscreen_renderer=cfg.offscreen_renderer,
		use_camera_obs=cfg.camera_obs,
		camera_names=[cfg.camera],
		camera_heights=cfg.img_size, camera_widths=cfg.img_size,
		reward_shaping=True,
		control_freq=cfg.control_freq
	)

	all_demos = []
	success_count = 0

	for ep in range(cfg.num_demos*2):
		actions, states, images, rewards = [], [], [], []
		obs = env.reset()
		phase = "approach"
		
		for _ in range(cfg.max_steps):
			action, phase = get_action(obs, phase)
	
			states.append(np.concatenate([
				np.array(obs[k], dtype=np.float64).flatten() for k in
				["robot0_eef_pos", "robot0_eef_quat",
				"robot0_gripper_qpos", "cube_pos"]
			]))
			actions.append(action)
			images.append(obs[f"{cfg.camera}_image"])
	
			obs, reward, done, info = env.step(action)
			rewards.append(reward)
		
		if env._check_success():
			all_demos.append({
				"states": np.array(states),
				"actions": np.array(actions),
				"rewards": np.array(rewards),
				"images": np.array(images, dtype=np.uint8)
			})	
			success_count += 1
			print(f"Demo {success_count}/{cfg.num_demos} collected (ep {ep})")
			
		if success_count >= cfg.num_demos:
			break
	
	env.close()

	if not all_demos:
		print("No successfull demos collected.")
		return

	with h5py.File(cfg.demo_data_path, "w") as f:
		for i, demo in enumerate(all_demos):
			grp = f.create_group(f"demo_{i}")
			grp.create_dataset("states", data=demo["states"])
			grp.create_dataset("actions", data=demo["actions"])
			grp.create_dataset("rewards", data=demo["rewards"])
			grp.create_dataset("images", data=demo["images"], compression="gzip", compression_opts=4)

		f.attrs["num_demos"] = len(all_demos)
		f.attrs["state_dim"] = all_demos[0]["states"].shape[1]
		f.attrs["action_dim"] = all_demos[0]["actions"].shape[1]

	print(f"\nSaved {len(all_demos)} demos to {cfg.demo_data_path}")
	print(f"State dim: {all_demos[0]['states'].shape[1]}")
	print(f"Action dim: {all_demos[0]['actions'].shape[1]}")


if __name__ == '__main__':
	collect_demos()
