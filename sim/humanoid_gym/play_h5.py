import argparse
import logging
import os
from datetime import datetime

import cv2
import h5py
import numpy as np
from isaacgym import gymapi
from tqdm import tqdm

from sim.logging import configure_logging

logger = logging.getLogger(__name__)
import copy

from sim.env import run_dir
from sim.humanoid_gym.envs import *  # noqa: F403


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def play(args: argparse.Namespace) -> None:
    configure_logging()

    logger.info("Configuring environment and training settings...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
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
    EXPORT_POLICY = True
    if EXPORT_POLICY:
        path = os.path.join(".")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    # Prepare for logging
    env_logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 1000

    # Initialize HDF5 file
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    h5_file = h5py.File("data" + now + ".h5", "w")
    max_timesteps = stop_state_log
    dset_obs = h5_file.create_dataset("observations", (max_timesteps,) + obs.shape, dtype=np.float32)
    sample_action = policy(obs.detach())
    dset_actions = h5_file.create_dataset("actions", (max_timesteps,) + sample_action.shape, dtype=np.float32)

    # Create datasets for additional logged parameters
    dset_dof_pos_target = h5_file.create_dataset("dof_pos_target", (max_timesteps,), dtype=np.float32)
    dset_dof_pos = h5_file.create_dataset("dof_pos", (max_timesteps,), dtype=np.float32)
    dset_dof_vel = h5_file.create_dataset("dof_vel", (max_timesteps,), dtype=np.float32)
    dset_dof_torque = h5_file.create_dataset("dof_torque", (max_timesteps,), dtype=np.float32)
    dset_command_x = h5_file.create_dataset("command_x", (max_timesteps,), dtype=np.float32)
    dset_command_y = h5_file.create_dataset("command_y", (max_timesteps,), dtype=np.float32)
    dset_command_yaw = h5_file.create_dataset("command_yaw", (max_timesteps,), dtype=np.float32)
    dset_base_vel_x = h5_file.create_dataset("base_vel_x", (max_timesteps,), dtype=np.float32)
    dset_base_vel_y = h5_file.create_dataset("base_vel_y", (max_timesteps,), dtype=np.float32)
    dset_base_vel_z = h5_file.create_dataset("base_vel_z", (max_timesteps,), dtype=np.float32)
    dset_base_vel_yaw = h5_file.create_dataset("base_vel_yaw", (max_timesteps,), dtype=np.float32)
    dset_contact_forces_z = h5_file.create_dataset(
        "contact_forces_z", (max_timesteps, 2), dtype=np.float32
    )

    if RENDER:
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

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # Creates a directory to store videos.
        video_dir = run_dir() / "videos"
        experiment_dir = video_dir / train_cfg.runner.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        dir = os.path.join(experiment_dir, datetime.now().strftime("%b%d_%H-%M-%S") + str(args.run_name) + ".mp4")
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for t in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())

        # Store observations and actions
        dset_obs[t] = obs.cpu().numpy()
        dset_actions[t] = actions.detach().numpy()

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
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

        # Store logged data
        dset_dof_pos_target[t] = dof_pos_target
        dset_dof_pos[t] = dof_pos
        dset_dof_vel[t] = dof_vel
        dset_dof_torque[t] = dof_torque
        dset_command_x[t] = command_x
        dset_command_y[t] = command_y
        dset_command_yaw[t] = command_yaw
        dset_base_vel_x[t] = base_vel_x
        dset_base_vel_y[t] = base_vel_y
        dset_base_vel_z[t] = base_vel_z
        dset_base_vel_yaw[t] = base_vel_yaw
        dset_contact_forces_z[t] = contact_forces_z

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

    if RENDER:
        video.release()

    # Close HDF5 file
    print("Saving data to" + os.path.abspath("data" + now + ".h5"))
    h5_file.close()


# Puts this import down here so that the environments are registered
# before we try to use them.
from humanoid.utils import Logger, get_args, task_registry  # noqa: E402

if __name__ == "__main__":
    RENDER = True
    FIX_COMMAND = True

    play(get_args())
