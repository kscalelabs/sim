"""Trains a humanoid to stand up."""

# ruff: noqa
# mypy: ignore-errors

import argparse

from sim.envs import task_registry  # noqa: E402
from sim.utils.args_parsing import parse_args_with_extras  # noqa: E402


def train(args: argparse.Namespace) -> None:
    env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


def add_train_arguments(parser):
    """Add training-specific arguments."""
    # Training
    parser.add_argument("--horovod", action="store_true", default=False, help="Use horovod for multi-gpu training")
    parser.add_argument("--trimesh", action="store_true", default=False, help="Use trimesh terrain")


if __name__ == "__main__":
    args = parse_args_with_extras(add_train_arguments)
    print("Arguments:", vars(args))
    train(args)
