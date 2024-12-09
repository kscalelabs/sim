"""
E.g. usage:
python sim/simple_play.py --task=stompymicro --sim_device=cpu --num_envs=1 --max_iterations=300 --load_run=-8
"""

import argparse
import logging

import numpy as np
from tqdm import tqdm

from sim.env_helpers import debug_robot_state
from sim.envs import task_registry  # noqa: E402
from sim.utils.args_parsing import parse_args_with_extras
from sim.utils.cmd_manager import CommandManager  # noqa: E402

from isaacgym import gymapi  # isort: skip
import torch  # isort: skip

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


def play(args: argparse.Namespace) -> None:
    logger.info("Configuring environment and training settings...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 1.5
    env_cfg.domain_rand.max_push_vel_xy = 0.6
    env_cfg.domain_rand.max_push_ang_vel = 1.2
    env_cfg.domain_rand.joint_angle_noise = 0.0
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

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

    for t in tqdm(range(train_cfg.runner.max_iterations)):
        actions = policy(obs.detach())
        env.commands[:] = torch.zeros(4)

        obs, _, _, _, _ = env.step(actions.detach())
        if t % 17 == 0:
            debug_robot_state("Isaac", obs[0], actions[0])
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)


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
