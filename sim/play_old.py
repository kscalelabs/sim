# mypy: ignore-errors
"""Play a trained policy in the environment.

Run:
    python sim/play.py --task g1 --log_h5
    python sim/play.py --task stompymini --log_h5
"""

import argparse
import copy
import logging
import os
from datetime import datetime
from typing import Any, Union

import cv2
import h5py
import numpy as np
from isaacgym import gymapi
from tqdm import tqdm

logger = logging.getLogger(__name__)

from sim.env import run_dir  # noqa: E402
from sim.envs import task_registry  # noqa: E402
from sim.model_export import convert_model_to_onnx  # noqa: E402
from sim.utils.helpers import get_args  # noqa: E402
from sim.utils.logger import Logger  # noqa: E402

import torch  # isort: skip


def export_policy_as_jit(actor_critic: Any, path: Union[str, os.PathLike]) -> None:
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def play(args: argparse.Namespace) -> None:
    logger.info("Configuring environment and training settings...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
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
    env_cfg.domain_rand.joint_angle_noise = 0.01
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

    # Prepare for logging
    env_logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 1000
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.log_h5:
        h5_file = h5py.File(f"data{now}.h5", "w")

        # Create dataset for actions
        max_timesteps = stop_state_log
        num_dof = env.num_dof
        dset_actions = h5_file.create_dataset("actions", (max_timesteps, num_dof), dtype=np.float32)

        # Create dataset of observations
        buf_len = len(env.obs_history)  # length of observation buffer
        dset_2D_command = h5_file.create_dataset(
            "observations/2D_command", (max_timesteps, buf_len, 2), dtype=np.float32
        )  # sin and cos commands
        dset_3D_command = h5_file.create_dataset(
            "observations/3D_command", (max_timesteps, buf_len, 3), dtype=np.float32
        )  # x, y, yaw commands
        dset_q = h5_file.create_dataset(
            "observations/q", (max_timesteps, buf_len, num_dof), dtype=np.float32
        )  # joint positions
        dset_dq = h5_file.create_dataset(
            "observations/dq", (max_timesteps, buf_len, num_dof), dtype=np.float32
        )  # joint velocities
        dset_obs_actions = h5_file.create_dataset(
            "observations/actions", (max_timesteps, buf_len, num_dof), dtype=np.float32
        )  # actions
        dset_ang_vel = h5_file.create_dataset(
            "observations/ang_vel", (max_timesteps, buf_len, 3), dtype=np.float32
        )  # root angular velocity
        dset_euler = h5_file.create_dataset(
            "observations/euler", (max_timesteps, buf_len, 3), dtype=np.float32
        )  # root orientation

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

    for t in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())
        if args.log_h5:
            dset_actions[t] = actions.detach().numpy()
        if args.fix_command:
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        print(f"IMU: {obs[0, (3 * env.num_actions + 5) + 3 : (3 * env.num_actions + 5) + 2 * 3]}")

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

        if args.log_h5:
            for i in range(buf_len):
                cur_obs = env.obs_history[i].tolist()[0]
                dset_2D_command[t, i] = cur_obs[0:2]  # sin and cos commands
                dset_3D_command[t, i] = cur_obs[2:5]  # x, y, yaw commands
                dset_q[t, i] = cur_obs[5 : 5 + num_dof]  # joint positions
                dset_dq[t, i] = cur_obs[5 + num_dof : 5 + 2 * num_dof]  # joint velocities
                dset_obs_actions[t, i] = cur_obs[5 + 2 * num_dof : 5 + 3 * num_dof]  # actions
                dset_ang_vel[t, i] = cur_obs[5 + 3 * num_dof : 8 + 3 * num_dof]  # root angular velocity
                dset_euler[t, i] = cur_obs[8 + 3 * num_dof : 11 + 3 * num_dof]  # root orientation

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
        if infos["episode"]:
            num_episodes = env.reset_buf.sum().item()
            if num_episodes > 0:
                env_logger.log_rewards(infos["episode"], num_episodes)

    env_logger.print_rewards()
    env_logger.plot_states()

    if args.render:
        video.release()

    if args.log_h5:
        print("Saving data to " + os.path.abspath(f"data{now}.h5"))
        h5_file.close()


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
