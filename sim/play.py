# mypy: ignore-errors
"""Play a trained policy in the environment.

Run:
    python sim/play.py --task gpr --log_h5
"""
import argparse
import copy
import logging
import math
import os
import time
import uuid
from datetime import datetime
from typing import Any, Union

import cv2
import h5py
import numpy as np
from isaacgym import gymapi
from kinfer.export.pytorch import export_to_onnx
from tqdm import tqdm

# Local imports third
from sim.env import run_dir  # noqa: E402
from sim.envs import task_registry  # noqa: E402
from sim.model_export import (  # noqa: E402
    ActorCfg,
    convert_model_to_onnx,
    get_actor_policy,
)
from sim.utils.helpers import get_args  # noqa: E402
from sim.utils.logger import Logger  # noqa: E402

import torch  # special case with isort: skip comment

logger = logging.getLogger(__name__)


def export_policy_as_jit(actor_critic: Any, path: Union[str, os.PathLike]) -> None:
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def play(args: argparse.Namespace) -> None:
    logger.info("Configuring environment and training settings...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    num_parallel_envs = 2
    env_cfg.env.num_envs = num_parallel_envs
    env_cfg.sim.max_gpu_contact_pairs = 2**10 * num_parallel_envs

    if args.trimesh:
        env_cfg.terrain.mesh_type = "trimesh"
    else:
        env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.0
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    logger.info("train_cfg.runner_class_name: %s", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy if needed
    if args.export_policy:
        path = os.path.join(".")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # export policy as a onnx module (used to run it on web)
    if args.export_onnx:
        path = ppo_runner.load_path
        embodiment = ppo_runner.cfg["experiment_name"].lower()
        policy_cfg = ActorCfg(embodiment=embodiment)

        if embodiment == "gpr":
            policy_cfg.cycle_time = 0.4
        elif embodiment == "zeroth":
            policy_cfg.cycle_time = 0.2
        else:
            print(f"Specific policy cfg for {embodiment} not implemented")

        actor_model, sim2sim_info, input_tensors = get_actor_policy(path, policy_cfg)

        # Merge policy_cfg and sim2sim_info into a single config object
        export_config = {**vars(policy_cfg), **sim2sim_info}

        export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_policy.onnx")
        print("Exported policy as kinfer-compatible onnx to: ", path)

    # Prepare for logging
    env_logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    env_steps_to_run = 1000

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.log_h5:
        from sim.h5_logger import HDF5Logger
        # Create directory for HDF5 files
        h5_dir = run_dir() / "h5_out" / args.task / now
        h5_dir.mkdir(parents=True, exist_ok=True)

        # Get observation dimensions
        num_actions = env.num_dof
        obs_buffer = env.obs_buf.shape[1]
        prev_actions = np.zeros((num_actions), dtype=np.double)

        h5_loggers = []
        for env_idx in range(env_cfg.env.num_envs):
            h5_dir = run_dir() / "h5_out" / args.task / now / f"env_{env_idx}"
            h5_dir.mkdir(parents=True, exist_ok=True)

            h5_loggers.append(
                HDF5Logger(
                    data_name=f"{args.task}_env_{env_idx}",
                    num_actions=num_actions,
                    max_timesteps=env_steps_to_run,
                    num_observations=obs_buffer,
                    h5_out_dir=str(h5_dir),
                )
            )

    if args.render:
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

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]

        # Creates a directory to store videos.
        video_dir = run_dir() / "videos"
        experiment_dir = video_dir / train_cfg.runner.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        dir = os.path.join(experiment_dir, now + str(args.run_name) + ".mp4")
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for t in tqdm(range(env_steps_to_run)):
        actions = policy(obs.detach())

        if args.fix_command:
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if args.render:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            video.write(img[..., :3])

        # Log states
        dof_pos_target = actions[robot_index, joint_index].item() * env.cfg.control.action_scale
        dof_pos = env.dof_pos[robot_index, joint_index].item()
        dof_vel = env.dof_vel[robot_index, joint_index].item()
        dof_torque = env.torques[robot_index, joint_index].item()
        command_x = env.commands[robot_index, 0].item()
        command_y = env.commands[robot_index, 1].item()
        command_yaw = env.commands[robot_index, 2].item()
        base_vel_x = env.base_lin_vel[robot_index, 0].item()
        base_vel_y = env.base_lin_vel[robot_index, 1].item()
        base_vel_z = env.base_lin_vel[robot_index, 2].item()
        base_vel_yaw = env.base_ang_vel[robot_index, 2].item()
        contact_forces_z = env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()

        env_logger.log_states(
            {
                "dof_pos_target": dof_pos_target,
                "dof_pos": dof_pos,
                "dof_vel": dof_vel,
                "dof_torque": dof_torque,
                "command_x": command_x,
                "command_y": command_y,
                "command_yaw": command_yaw,
                "base_vel_x": base_vel_x,
                "base_vel_y": base_vel_y,
                "base_vel_z": base_vel_z,
                "base_vel_yaw": base_vel_yaw,
                "contact_forces_z": contact_forces_z,
            }
        )
        actions = actions.detach().cpu().numpy()
        if args.log_h5:
            # Extract the current observation
            for env_idx in range(env_cfg.env.num_envs):
                h5_loggers[env_idx].log_data(
                    {
                        "t": np.array([t * env.dt], dtype=np.float32),
                        "2D_command": np.array(
                            [
                                np.sin(2 * math.pi * t * env.dt / env.cfg.rewards.cycle_time),
                                np.cos(2 * math.pi * t * env.dt / env.cfg.rewards.cycle_time),
                            ],
                            dtype=np.float32,
                        ),
                        "3D_command": np.array(env.commands[env_idx, :3].cpu().numpy(), dtype=np.float32),
                        "joint_pos": np.array(env.dof_pos[env_idx].cpu().numpy(), dtype=np.float32),
                        "joint_vel": np.array(env.dof_vel[env_idx].cpu().numpy(), dtype=np.float32),
                        "prev_actions": prev_actions[env_idx].astype(np.float32),
                        "curr_actions": actions[env_idx].astype(np.float32),
                        "ang_vel": env.base_ang_vel[env_idx].cpu().numpy().astype(np.float32),
                        "euler_rotation": env.base_euler_xyz[env_idx].cpu().numpy().astype(np.float32),
                        "buffer": env.obs_buf[env_idx].cpu().numpy().astype(np.float32),
                    }
                )

            prev_actions = actions

        if infos["episode"]:
            num_episodes = env.reset_buf.sum().item()
            if num_episodes > 0:
                env_logger.log_rewards(infos["episode"], num_episodes)

    env_logger.print_rewards()

    if args.render:
        video.release()

    if args.log_h5:
        # print(f"Saving HDF5 file to {h5_logger.h5_file_path}") # TODO use code from kdatagen
        for h5_logger in h5_loggers:
            h5_logger.close()
        print(f"HDF5 file(s) saved!")


if __name__ == "__main__":
    base_args = get_args()
    parser = argparse.ArgumentParser(description="Extend base arguments with log_h5")
    parser.add_argument("--log_h5", action="store_true", help="Enable HDF5 logging")
    parser.add_argument("--render", action="store_true", help="Enable rendering", default=True)
    parser.add_argument("--fix_command", action="store_true", help="Fix command", default=True)
    parser.add_argument("--export_onnx", action="store_true", help="Export policy as ONNX", default=True)
    parser.add_argument("--export_policy", action="store_true", help="Export policy as JIT", default=True)
    args, unknown = parser.parse_known_args(namespace=base_args)

    play(args)
