"""
E.g. usage:
python sim/simple_play.py --task=stompymicro --sim_device=cpu --num_envs=1 --max_iterations=300 --load_run=-8
"""

import argparse
import logging
import os

import numpy as np
from datetime import datetime

from sim import ROOT_DIR
from sim.env_helpers import debug_robot_state
from sim.utils.args_parsing import parse_args_with_extras
from sim.envs.humanoids.stompymicro_config import StompyMicroCfgPPO, StompyMicroCfg
from sim.envs.humanoids.stompymicro_env import StompyMicroEnv
from sim.utils.helpers import (
    class_to_dict,
    get_load_path,
    parse_sim_params,
)
from sim.algo.ppo.on_policy_runner import OnPolicyRunner
from kinfer.export.pytorch import export_to_onnx  # isort: skip
from kinfer.inference.python import ONNXModel  # isort: skip
from isaacgym import gymapi  # isort: skip
import torch  # isort: skip

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


def play(args: argparse.Namespace) -> None:
    LOAD_MODEL_PATH = "examples/experiments/standing/robustv1/policy_1.pt"
    DEVICE = "cpu"

    ## ISAAC ##
    args.load_model = LOAD_MODEL_PATH
    env_cfg, train_cfg = StompyMicroCfg(), StompyMicroCfgPPO()
    env_cfg.env.num_envs = 1
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    train_cfg.runner.load_run = -8
    train_cfg.seed = 0
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = StompyMicroEnv(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    train_cfg_dict = class_to_dict(train_cfg)
    env_cfg_dict = class_to_dict(env_cfg)
    all_cfg = {**train_cfg_dict, **env_cfg_dict}
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)
    ppo_runner = OnPolicyRunner(env, all_cfg, log_dir=log_dir, device=DEVICE)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    ppo_runner.load(resume_path, load_optimizer=False)
    ppo_runner.alg.actor_critic.eval()
    ppo_runner.alg.actor_critic.to(DEVICE)
    policy = ppo_runner.alg.actor_critic.act_inference
    # -> env, policy
    ## ISAAC ##

    obs, _ = env.reset()
    steps = int(env_cfg.env.episode_length_s / (env_cfg.sim.dt * env_cfg.control.decimation))
    for t in range(steps):
        actions = policy(obs.detach())
        env.commands[:] = torch.zeros(4)
        obs, _, _, _, _ = env.step(actions.detach())
        if t % 17 == 0:
            debug_robot_state("Isaac", obs[0], actions[0])


if __name__ == "__main__":
    args = parse_args_with_extras(lambda x: x)
    print("Arguments:", vars(args))
    play(args)
