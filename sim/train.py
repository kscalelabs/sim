"""Trains a humanoid to stand up."""

# ruff: noqa
# mypy: ignore-errors

import argparse

from sim.envs import task_registry  # noqa: E402
from sim.utils.helpers import get_args  # noqa: E402


def train(args: argparse.Namespace) -> None:
    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


# Puts this import down here so that the environments are registered

if __name__ == "__main__":
    # python -m sim.humanoid_gym.train
    train(get_args())
