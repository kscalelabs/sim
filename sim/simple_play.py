import argparse
import logging

import numpy as np
from tqdm import tqdm

from sim.envs import task_registry  # noqa: E402
from sim.utils.args_parsing import parse_args_with_extras
from sim.utils.cmd_manager import CommandManager  # noqa: E402

from isaacgym import gymapi  # isort: skip

np.set_printoptions(precision=4, suppress=True)  # Customize precision

logger = logging.getLogger(__name__)


def play(args: argparse.Namespace) -> None:
    logger.info("Configuring environment and training settings...")
    print(f"task: {args.task}")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    logger.info("train_cfg.runner_class_name: %s", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    cmd_manager = CommandManager(
        num_envs=env_cfg.env.num_envs, mode=args.command_mode, device=env.device, env_cfg=env_cfg
    )

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(3, -3, 1)
    camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    logger.info("body_handle: %s", body_handle)
    logger.info("actor_handle: %s", actor_handle)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
    )

    for _ in tqdm(range(train_cfg.runner.max_iterations)):
        actions = policy(obs.detach())
        commands = cmd_manager.update(env.dt)
        env.commands[:] = commands

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

    cmd_manager.close()


def add_play_arguments(parser):
    """Add play-specific arguments."""
    parser.add_argument(
        "--command_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oscillating", "random", "keyboard"],
        help="Control mode for the robot",
    )


if __name__ == "__main__":
    args = parse_args_with_extras(add_play_arguments)
    print("Arguments:", vars(args))
    play(args)

