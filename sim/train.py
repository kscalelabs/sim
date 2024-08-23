"""Trains a humanoid to stand up."""

import argparse

from sim.utils.helpers import get_args
from sim.utils.task_registry import TaskRegistry


def train(args: argparse.Namespace) -> None:
    # pfb30
    # from sim.envs import register_tasks
    from sim.envs.humanoids.stompymini_config import MiniCfg, MiniCfgPPO
    from sim.envs.humanoids.stompymini_env import MiniFreeEnv

    task_registry = TaskRegistry()
    task_registry.register("stompymini", MiniFreeEnv, MiniCfg(), MiniCfgPPO())

    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


# Puts this import down here so that the environments are registered

if __name__ == "__main__":
    # python -m sim.humanoid_gym.train
    train(get_args())
