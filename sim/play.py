# mypy: ignore-errors
"""Play a trained policy in the environment.

Run:
    python sim/play2.py --task zbot2
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
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Local imports third
from sim.env import run_dir
from sim.envs import task_registry
from sim.h5_logger import HDF5Logger

import torch  # special case with isort: skip comment
from sim.env import run_dir  # noqa: E402
from sim.envs import task_registry  # noqa: E402
from sim.model_export import ActorCfg, get_actor_policy  # noqa: E402
from sim.utils.helpers import get_args  # noqa: E402
from sim.utils.logger import Logger  # noqa: E402
from kinfer.export.pytorch import export_to_onnx

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

    num_parallel_envs = 5
    env_cfg.env.num_envs = num_parallel_envs
    env_cfg.sim.max_gpu_contact_pairs = 2**10 * num_parallel_envs

    if args.trimesh:
        env_cfg.terrain.mesh_type = "trimesh"
    else:
        env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 10
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = True
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
        embodiment = ppo_runner.cfg['experiment_name'].lower()
        policy_cfg = ActorCfg(
            embodiment=embodiment,
            cycle_time=env_cfg.rewards.cycle_time,
            sim_dt=env_cfg.sim.dt,
            sim_decimation=env_cfg.control.decimation,
            tau_factor=env_cfg.safety.torque_limit,
            action_scale=env_cfg.control.action_scale,
            lin_vel_scale=env_cfg.normalization.obs_scales.lin_vel,
            ang_vel_scale=env_cfg.normalization.obs_scales.ang_vel,
            quat_scale=env_cfg.normalization.obs_scales.quat,
            dof_pos_scale=env_cfg.normalization.obs_scales.dof_pos,
            dof_vel_scale=env_cfg.normalization.obs_scales.dof_vel,
            num_single_obs=env_cfg.env.num_single_obs,
            num_actions=env_cfg.env.num_actions,
            num_joints=env_cfg.env.num_joints,
            frame_stack=env_cfg.env.frame_stack,
            clip_observations=env_cfg.normalization.clip_observations,
            clip_actions=env_cfg.normalization.clip_actions,
            use_projected_gravity=env_cfg.sim.use_projected_gravity,
        )

        actor_model, sim2sim_info, input_tensors = get_actor_policy(path, policy_cfg)

        # Merge policy_cfg and sim2sim_info into a single config object
        export_config = {**vars(policy_cfg), **sim2sim_info}

        export_to_onnx(
            actor_model,
            input_tensors=input_tensors,
            config=export_config,
            save_path="kinfer_policy.onnx"
        )
        print("Exported policy as kinfer-compatible onnx to: ", path)

    # Prepare for logging
    env_logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    env_steps_to_run = 1000

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.log_h5:
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
            
            h5_loggers.append(HDF5Logger(
                data_name=f"{args.task}_env_{env_idx}",
                num_actions=num_actions,
                max_timesteps=env_steps_to_run,
                num_observations=obs_buffer,
                h5_out_dir=str(h5_dir)
            ))

    if args.render:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        # camera_offset = gymapi.Vec3(3, -3, 1)
        camera_offset = gymapi.Vec3(1, -2, 0.5)
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.1, 0.1, 1), np.deg2rad(135))
        camera_rotation = gymapi.Quat.from_euler_zyx(np.deg2rad(0), np.deg2rad(30), np.deg2rad(115))
        # breakpoint()
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

    # Input data logging
    input_data = {
        'timestamp': [],
        # 2D command
        'sin_command': [],
        'cos_command': [],
        # 3D command
        'command_x': [],
        'command_y': [],
        'command_yaw': [],
        # Base state
        'ang_vel_x': [],
        'ang_vel_y': [],
        'ang_vel_z': [],
        'roll': [],
        'pitch': [],
        'yaw': [],
    }
    
    # Add joint state columns
    for i in range(env.num_dof):
        input_data[f'joint_pos_{i}'] = []
        input_data[f'joint_vel_{i}'] = []

    # Target data logging (original)
    target_data = {
        'timestamp': [],
    }
    for i in range(env.num_dof):
        target_data[f'dof_pos_target_{i}'] = []
        target_data[f'dof_vel_target_{i}'] = []

    prev_actions = np.zeros((num_parallel_envs, env.num_dof * 2), dtype=np.double)

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

        # Log input data
        input_data['timestamp'].append(t * env.dt)
        # 2D command
        input_data['sin_command'].append(np.sin(2 * math.pi * t * env.dt / env.cfg.rewards.cycle_time))
        input_data['cos_command'].append(np.cos(2 * math.pi * t * env.dt / env.cfg.rewards.cycle_time))
        # 3D command
        input_data['command_x'].append(env.commands[robot_index, 0].item())
        input_data['command_y'].append(env.commands[robot_index, 1].item())
        input_data['command_yaw'].append(env.commands[robot_index, 2].item())
        
        # Joint states
        for i in range(env.num_dof):
            input_data[f'joint_pos_{i}'].append(env.dof_pos[robot_index, i].item())
            input_data[f'joint_vel_{i}'].append(env.dof_vel[robot_index, i].item())
            
        # Base state
        input_data['ang_vel_x'].append(env.base_ang_vel[robot_index, 0].item())
        input_data['ang_vel_y'].append(env.base_ang_vel[robot_index, 1].item())
        input_data['ang_vel_z'].append(env.base_ang_vel[robot_index, 2].item())
        input_data['roll'].append(env.base_euler_xyz[robot_index, 0].item())
        input_data['pitch'].append(env.base_euler_xyz[robot_index, 1].item())
        input_data['yaw'].append(env.base_euler_xyz[robot_index, 2].item())

        # Log target data
        target_data['timestamp'].append(t * env.dt)
        actions_np = actions.detach().cpu().numpy()

        # Clip actions to match environment limits
        actions_np = np.clip(actions_np, -env.cfg.normalization.clip_actions, env.cfg.normalization.clip_actions)
        actions_np = actions_np * env.cfg.control.action_scale

        for i in range(env.num_dof):
            target_data[f'dof_pos_target_{i}'].append(actions_np[robot_index, i])
            target_data[f'dof_vel_target_{i}'].append(actions_np[robot_index, i + env.num_dof])

        prev_actions = actions_np
        
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

    # Save to CSV
    csv_dir = run_dir() / "csv_out" / args.task / now
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Save inputs
    df_inputs = pd.DataFrame(input_data)
    input_path = csv_dir / "inputs.csv"
    df_inputs.to_csv(input_path, index=False)
    print(f"Input CSV file saved to: {input_path}")

    # Save targets
    df_targets = pd.DataFrame(target_data)
    target_path = csv_dir / "targets.csv"
    df_targets.to_csv(target_path, index=False)
    print(f"Target CSV file saved to: {target_path}")


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
