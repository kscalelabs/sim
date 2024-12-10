import argparse
import logging

import numpy as np
from tqdm import tqdm

from sim.env_helpers import debug_robot_state
from sim.envs.base.mujoco_env import MujocoCfg, MujocoEnv
from sim.utils.args_parsing import parse_args_with_extras
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg, StompyMicroCfgPPO
from sim.envs.humanoids.stompymicro_env import StompyMicroEnv
from sim.utils.helpers import class_to_dict, parse_sim_params
from sim.algo.ppo.on_policy_runner import OnPolicyRunner
import torch # isort: skip

logger = logging.getLogger(__name__)

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3)

DEVICE = "cuda:0"  # "cpu"
LOAD_MODEL_PATH = "examples/experiments/standing/robustv1/policy_1.pt"


def coplay(args: argparse.Namespace) -> None:
    """Run both Isaac Gym and Mujoco environments simultaneously."""
    cfg = MujocoCfg()
    cfg.env.num_envs = 1
    
    cfg.gains.kp_scale = 1.
    cfg.gains.kd_scale = 1.
    cfg.gains.tau_factor = 4.0
    
    # Remove randomness
    cfg.domain_rand.start_pos_noise = 0.0
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.action_noise = 0.0
    cfg.domain_rand.action_delay = 0.0
    
    cfg.sim.physx.max_gpu_contact_pairs = 2**10
    
    train_cfg = StompyMicroCfgPPO()
    train_cfg.runner.load_run = -1
    train_cfg.seed = 0

    # Create and setup environment
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    RENDER_MUJOCO = True
    isaac_env = StompyMicroEnv(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=RENDER_MUJOCO,
    )
    mujoco_env = MujocoEnv(
        cfg,
        render=RENDER_MUJOCO,
    )

    # Setup and load policy
    all_cfg = {**class_to_dict(train_cfg), **class_to_dict(cfg)}
    ppo_runner = OnPolicyRunner(isaac_env, all_cfg, device=DEVICE)
    resume_path = "logs/StompyMicro/Dec04_20-29-12_StandingRobust/model_12001.pt"
    ppo_runner.load(resume_path, load_optimizer=False)
    ppo_runner.alg.actor_critic.eval()
    ppo_runner.alg.actor_critic.to(DEVICE)
    policy = ppo_runner.alg.actor_critic.act_inference

    isaac_obs, _ = isaac_env.reset()
    mujoco_obs_np = mujoco_env.reset()
    mujoco_obs = torch.from_numpy(mujoco_obs_np).float().to(DEVICE)
    
    ## DEBUG ##
    print(f"Policy device: {next(ppo_runner.alg.actor_critic.parameters()).device}")
    print(f"Isaac obs device: {isaac_obs.device}")
    print(f"Mujoco obs device: {mujoco_obs.device}")
    
    print(f"Isaac obs shape: {isaac_obs.shape}")
    print(f"Mujoco obs shape: {mujoco_obs.shape}")
    
    print(isaac_obs[0][-59:].detach().cpu().numpy())
    print(mujoco_obs[0][-59:].detach().cpu().numpy())
    ## DEBUG ##
    
    done1 = done2 = False
    
    ISAAC = False

    steps = int(cfg.env.episode_length_s / (cfg.sim.dt * cfg.control.decimation))
    for t in tqdm(range(steps)):
        if ISAAC:
            isaac_actions = policy(isaac_obs.detach())
            isaac_env.commands[:] = torch.zeros(4)
            isaac_obs, _, _, done1, _ = isaac_env.step(isaac_actions.detach())
        
        mujoco_actions = policy(mujoco_obs.detach())
        mujoco_env.commands[:] = torch.zeros(4)
        mujoco_obs_np, _, done2, _ = mujoco_env.step(mujoco_actions.detach().cpu().numpy())
        mujoco_obs = torch.from_numpy(mujoco_obs_np).float().to(DEVICE)

        if t % 17 == 0:
            print(f"Time: {t * cfg.sim.dt:.2f}s")
            if ISAAC:
                debug_robot_state("Isaac", isaac_obs[0].detach().cpu().numpy(), isaac_actions[0].detach().cpu().numpy())
            debug_robot_state("Mujoco", mujoco_obs[0].detach().cpu().numpy(), mujoco_actions[0].detach().cpu().numpy())

        if done1 or done2:
            break


if __name__ == "__main__":
    args = parse_args_with_extras(lambda x: x)
    print("Arguments:", vars(args))
    coplay(args)
