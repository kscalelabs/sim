"""Trains a humanoid to stand up."""

import argparse
import isaacgym
import os
import cv2
import torch
from sim.envs import task_registry
from sim.utils.helpers import get_args
from sim.tdmpc.src import logger
from sim.tdmpc.src.algorithm.helper import Episode, ReplayBuffer
from sim.tdmpc.src.algorithm.tdmpc import TDMPC
from dataclasses import dataclass, field
from isaacgym import gymapi
from typing import List
import time
import numpy as np
from pathlib import Path
import random
from datetime import datetime
torch.backends.cudnn.benchmark = True
__LOGS__ = "logs"

@dataclass
class TDMPC_DoraConfigs:
    seed: int = 42
    task : str = "walk"
    exp_name : str = "dora"
    device : str = "cuda:0"
    num_envs : int = 10

    lr : float = 1e-3
    modality : str = "state"
    enc_dim: int = 512 # 256
    mlp_dim = [512, 256] # [256, 256]
    latent_dim: int = 100

    iterations : int = 12
    num_samples : int = 512
    num_elites : int = 50
    mixture_coef : float = 0.05
    min_std : float = 0.05
    temperature : float = 0.5
    momentum : float = 0.1
    horizon : int = 5
    std_schedule: str = f"linear(0.5, {min_std}, 3000)"
    horizon_schedule: str = f"linear(1, {horizon}, 2500)"

    batch_size: int = 1024
    max_buffer_size : int = 1000000
    reward_coef : float = 1
    value_coef : float = 0.5
    consistency_coef : float = 2
    rho : float = 0.5
    kappa : float = 0.1
    per_alpha: float = 0.6
    per_beta : float =  0.4
    grad_clip_norm : float =  10
    seed_steps: int = 750
    update_freq: int = 2
    tau: int = 0.01

    discount : float = 0.99
    buffer_device : str = "cpu"
    train_steps : int = int(1e6)
    num_q : int = 3

    action_repeat : int = 2
    eval_freq: int = 15000
    eval_freq_episode : int = 10
    eval_episodes : int = 1

    save_model : bool = True
    save_video : bool = False

    use_wandb  : bool = False
    wandb_entity : str = "crajagopalan"
    wandb_project : str = "xbot"
	


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class VideoRecorder:
    """Utility class for logging evaluation videos."""
    def __init__(self, root_dir, render_size=384, fps=25):
        self.save_dir = (root_dir / 'eval_video') if root_dir else None
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # type: ignore[attr-defined]
        logger.make_dir(self.save_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Creates a directory to store videos.
        dir = os.path.join(self.save_dir, now + ".mp4")
        self.video = cv2.VideoWriter(dir, fourcc, float(fps), (1920, 1080))

    def init(self, env, h1, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and enabled
        self.record(env, h1)

    def record(self, env, h1):
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        self.video.write(img)

    def save(self,):
          self.video.release()
			
def evaluate(test_env, agent, h1, step, video, action_repeat=1):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    obs, privileged_obs = test_env.reset()
    critic_obs = privileged_obs if privileged_obs is not None else obs	
    state = torch.cat([obs, critic_obs], dim=-1) if privileged_obs is not None else obs
    dones, ep_reward, t = torch.tensor([False] * test_env.num_envs), torch.tensor([0.] * test_env.num_envs), 0
    if video: video.init(test_env, h1, enabled=True)
    for i in range(int(1000 // action_repeat)):
        actions = agent.plan(state, eval_mode=True, step=step, t0=t==0)
        for _ in range(action_repeat):
            obs, privileged_obs, rewards, dones, infos = test_env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            ep_reward += rewards.cpu()
            t += 1
            if video: video.record(test_env, h1)
        state = torch.cat([obs, critic_obs], dim=-1) if privileged_obs is not None else obs
    episode_rewards.append(ep_reward)
    if video: video.save()
    print(f"Timestep : {t} Episode Rewards - {torch.cat(episode_rewards).mean().item()}")
    return torch.nanmean(torch.cat(episode_rewards)).item()

def play(args: argparse.Namespace) -> None:
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()	
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args)	

    fp = "/home/guest/sim/logs/2024-09-11_08-08-01_walk_state_dora/models/tdmpc_policy_2350.pt"
    config = torch.load(fp)["config"]
    tdmpc_cfg = TDMPC_DoraConfigs(**config)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(3, -3, 1)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
    )

    set_seed(tdmpc_cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / f"{tdmpc_cfg.task}_{tdmpc_cfg.modality}_{tdmpc_cfg.exp_name}_{str(tdmpc_cfg.seed)}"

    obs, privileged_obs = env.reset()
    critic_obs = privileged_obs if privileged_obs is not None else obs	
    state = torch.cat([obs, critic_obs], dim=-1)[0] if privileged_obs is not None else obs[0]

    tdmpc_cfg.obs_shape = [state.shape[0]]
    tdmpc_cfg.action_shape = (env.num_actions)
    tdmpc_cfg.action_dim = env.num_actions
    tdmpc_cfg.episode_length = 100 # int(env.max_episode_length // tdmpc_cfg.action_repeat)
    tdmpc_cfg.num_envs = env.num_envs

    L = logger.Logger(work_dir, tdmpc_cfg)
    log_dir = logger.make_dir(work_dir)	
    video = VideoRecorder(log_dir)

    agent = TDMPC(tdmpc_cfg)
    
    agent.load(fp)
    step = 0
    episode_idx, start_time = 0, time.time()
    if fp is not None:
        episode_idx = int(fp.split(".")[0].split("_")[-1])
        step = episode_idx * tdmpc_cfg.episode_length

    # Log training episode
    evaluate(env, agent, h1, step, video, tdmpc_cfg.action_repeat)
    print('Testing completed successfully')

if __name__ == "__main__":
    play(get_args())