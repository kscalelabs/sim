import argparse
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from isaacgym import gymapi
from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel
from tqdm import tqdm

from sim.env_helpers import debug_robot_state
from sim.envs import task_registry
from sim.envs.base.mujoco_env import MujocoCfg, MujocoEnv
from sim.model_export import ActorCfg
from sim.sim2sim.helpers import get_actor_policy
from sim.utils.args_parsing import parse_args_with_extras
from sim.utils.cmd_manager import CommandManager
import torch # isort: skip

logger = logging.getLogger(__name__)


@dataclass
class ObservationBuffer:
    """Store observations from both environments for comparison."""

    isaac_obs: List[np.ndarray]
    mujoco_obs: List[np.ndarray]
    isaac_actions: List[np.ndarray]
    mujoco_actions: List[np.ndarray]
    timestamps: List[float]


def setup_isaac_env(args):
    """Setup Isaac Gym environment following original implementation."""
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

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    cmd_manager = CommandManager(
        num_envs=env_cfg.env.num_envs, mode=args.command_mode, device=env.device, env_cfg=env_cfg
    )

    # camera_properties = gymapi.CameraProperties()
    # camera_properties.width = 1920
    # camera_properties.height = 1080
    # h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    # camera_offset = gymapi.Vec3(3, -3, 1)
    # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    # actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    # body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    # env.gym.attach_camera_to_body(
    #     h1, env.envs[0], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION
    # )

    return env, policy, cmd_manager, train_cfg


def coplay(args: argparse.Namespace) -> None:
    """Run both Isaac Gym and Mujoco environments simultaneously."""
    # Setup Isaac Gym
    isaac_env, isaac_policy, _, train_cfg = setup_isaac_env(args)
    isaac_obs = isaac_env.reset()[0]

    # Setup Mujoco
    mujoco_cfg = MujocoCfg()
    mujoco_cfg.gains.kp_scale = 100.0
    mujoco_cfg.gains.kd_scale = 0.2
    mujoco_cfg.gains.tau_factor = 6.350
    mujoco_cfg.domain_rand.start_pos_noise = 0.0
    mujoco_cfg.domain_rand.randomize_friction = False
    mujoco_cfg.domain_rand.randomize_base_mass = False
    mujoco_cfg.domain_rand.push_robots = False
    # mujoco_cfg.domain_rand.action_delay = 0.5
    mujoco_cfg.domain_rand.action_noise = 0.0
    mujoco_cfg.env.num_envs = 1
    mujoco_env = MujocoEnv(mujoco_cfg, render=False)

    # Load and export policy
    LOAD_MODEL_PATH = "policy_1.pt"
    policy_cfg = ActorCfg(embodiment=mujoco_cfg.asset.name)
    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print("Export config:", export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    mujoco_policy = ONNXModel("kinfer_test.onnx")

    # Initialize Mujoco state
    mujoco_obs = mujoco_env.reset()
    target_q = np.zeros(mujoco_env.num_joints)
    prev_actions = np.zeros(mujoco_env.num_joints)
    hist_obs = np.zeros(mujoco_policy.get_metadata()["num_observations"])

    # Initialize observation buffer
    obs_buffer = ObservationBuffer([], [], [], [], [])

    for t in tqdm(range(train_cfg.runner.max_iterations)):
        # Isaac Gym step
        isaac_actions = isaac_policy(isaac_obs.detach())
        isaac_env.commands[:] = torch.zeros(4)
        isaac_obs, _, _, done1, info1 = isaac_env.step(isaac_actions.detach())

        # Mujoco step
        q, dq, quat, v, omega, euler = mujoco_obs
        x_vel, y_vel, rot = 0.0, 0.0, 0.0
        outputs = mujoco_policy(
            {
                "x_vel.1": np.array([x_vel], dtype=np.float32),
                "y_vel.1": np.array([y_vel], dtype=np.float32),
                "rot.1": np.array([rot], dtype=np.float32),
                "t.1": np.array([t * mujoco_cfg.sim.dt], dtype=np.float32),
                "dof_pos.1": (q - mujoco_env.default_joint_pos).astype(np.float32),
                "dof_vel.1": dq.astype(np.float32),
                "prev_actions.1": prev_actions.astype(np.float32),
                "imu_ang_vel.1": omega.astype(np.float32),
                "imu_euler_xyz.1": euler.astype(np.float32),
                "buffer.1": hist_obs.astype(np.float32),
            }
        )
        target_q = outputs["actions_scaled"]
        prev_actions = outputs["actions"]
        hist_obs = outputs["x.3"]

        mujoco_obs, _, done2, info2 = mujoco_env.step(target_q)

        # Store observations
        obs_buffer.isaac_obs.append(isaac_obs[0].detach().cpu().numpy())
        obs_buffer.mujoco_obs.append(mujoco_obs)
        obs_buffer.isaac_actions.append(isaac_actions[0].detach().cpu().numpy())
        obs_buffer.mujoco_actions.append(target_q)
        obs_buffer.timestamps.append(t * mujoco_cfg.sim.dt)

        # Debug output
        if t % 17 == 0:
            command_input = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            mujoco_obs_buf = np.concatenate([command_input, q, dq, prev_actions, omega, euler])

            print(f"Time: {t * mujoco_cfg.sim.dt:.2f}s")
            debug_robot_state("Isaac", isaac_obs[0].detach().cpu().numpy(), isaac_actions[0].detach().cpu().numpy())
            debug_robot_state("Mujoco", mujoco_obs_buf, actions=target_q)

        # Render Isaac Gym
        isaac_env.gym.fetch_results(isaac_env.sim, True)
        isaac_env.gym.step_graphics(isaac_env.sim)
        isaac_env.gym.render_all_camera_sensors(isaac_env.sim)

        if done1 and done2:
            break

    # Save observations
    if args.save_observations:
        np.savez(
            "coplay_observations.npz",
            isaac_obs=np.array(obs_buffer.isaac_obs),
            mujoco_obs=np.array(obs_buffer.mujoco_obs),
            isaac_actions=np.array(obs_buffer.isaac_actions),
            mujoco_actions=np.array(obs_buffer.mujoco_actions),
            timestamps=np.array(obs_buffer.timestamps),
        )


def add_coplay_arguments(parser):
    """Add coplay-specific arguments."""
    parser.add_argument(
        "--command_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oscillating", "random", "keyboard"],
        help="Control mode for the robot",
    )
    parser.add_argument(
        "--save_observations",
        action="store_true",
        help="Save observation buffers to file",
    )


if __name__ == "__main__":
    args = parse_args_with_extras(add_coplay_arguments)
    print("Arguments:", vars(args))
    coplay(args)
