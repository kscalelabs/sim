import argparse

from sim.env_helpers import debug_robot_state
from sim.utils.args_parsing import parse_args_with_extras
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg, StompyMicroCfgPPO
from sim.envs.humanoids.stompymicro_env import StompyMicroEnv
from sim.utils.helpers import (
    class_to_dict,
    parse_sim_params,
)
from sim.algo.ppo.on_policy_runner import OnPolicyRunner
import torch # isort: skip


def simple_play(args: argparse.Namespace) -> None:
    DEVICE = "cpu"
    RAND = True
    # Setup environment and training configs
    env_cfg, train_cfg = StompyMicroCfg(), StompyMicroCfgPPO()
    env_cfg.env.num_envs = 1
    env_cfg.domain_rand.randomize_friction = RAND
    env_cfg.domain_rand.randomize_base_mass = RAND
    env_cfg.domain_rand.push_robots = RAND
    if not RAND:
        env_cfg.domain_rand.action_noise = 0
        env_cfg.domain_rand.action_delay = 0
        env_cfg.domain_rand.push_interval_s = 0
        env_cfg.domain_rand.max_push_vel_xy = 0
    else:
        env_cfg.domain_rand.push_interval_s = 0.1 #1.7
        env_cfg.domain_rand.max_push_vel_xy = 0.85
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    train_cfg.runner.load_run = -10
    train_cfg.seed = 0

    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = StompyMicroEnv(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=False,
    )
    command = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(DEVICE)
    env.commands[:] = command

    # Setup and load policy
    all_cfg = {**class_to_dict(train_cfg), **class_to_dict(env_cfg)}
    ppo_runner = OnPolicyRunner(env, all_cfg, device=DEVICE)
    resume_path = "logs/StompyMicro/Dec04_20-29-12_StandingRobust/model_12001.pt"
    ppo_runner.load(resume_path, load_optimizer=False)
    ppo_runner.alg.actor_critic.eval()
    ppo_runner.alg.actor_critic.to(DEVICE)
    policy = ppo_runner.alg.actor_critic.act_inference

    # Run simulation
    obs, _ = env.reset()
    steps = int(env_cfg.env.episode_length_s / (env_cfg.sim.dt * env_cfg.control.decimation))
    for t in range(steps):
        actions = policy(obs.detach())
        env.commands[:] = command
        obs, _, _, _, _ = env.step(actions.detach())
        if t % 17 == 0:
            debug_robot_state("Isaac", obs[0], actions[0])


if __name__ == "__main__":
    args = parse_args_with_extras(lambda x: x)
    print("Arguments:", vars(args))
    simple_play(args)
