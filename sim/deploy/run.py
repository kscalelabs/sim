"""Basic sim2sim and sim2real deployment script.
          <joint name="lfemurrx" pos="0 0 0" axis="1 0 0" range="-0.349066 2.79253" class="stiff_medium"/>

Run example:
    mjpython sim/deploy/run.py --load_model sim/deploy/tests/walking_policy.pt --world MUJOCO
"""

import argparse
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import mujoco
import mujoco.viewer
import numpy as np
from humanoid.envs import *
from humanoid.utils import task_registry
from isaacgym import gymapi
from isaacgym.torch_utils import *
from policy import SimPolicy
from tqdm import tqdm

from sim.deploy.config import RobotConfig
from sim.env import stompy_mjcf_path
from sim.stompy_legs.joints import Stompy as StompyFixed


class Worlds(Enum):
    MUJOCO = "SIM"
    REAL = "REAL"
    ISAAC = "ISAAC"


class World(ABC):
    @abstractmethod
    def step(self, obs: np.ndarray) -> None:
        """Performs a simulation step in the world."""
        pass

    @abstractmethod
    def get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts an observation from the world state.

        Returns:
            A tuple containing the following:
            - dof_pos: The joint positions.
            - dof_vel: The joint velocities.
            - orientation: The orientation of the robot.
            - ang_vel: The angular velocity of the robot.
        """
        pass


class MujocoWorld(World):
    """Simulated world using MuJoCo.

    Attributes:
        cfg: The robot configuration.
        model: The MuJoCo model of the robot.
        data: The MuJoCo data structure for the simulation.
    """

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.robot_model_path))

        self.model.opt.timestep = self.cfg.dt

        # First step
        self.data = mujoco.MjData(self.model)
        self.data.qpos = self.model.keyframe("default").qpos

        mujoco.mj_step(self.model, self.data)

        # Set self.data initial state to the default standing position
        # self.data.qpos = StompyFixed.default_standing()
        # TODO: This line should produce same results soon

    def step(
        self,
        tau: np.ndarray = None,
    ) -> None:
        """Performs a simulation step in the MuJoCo world.

        Args:
            target_dof_pos: The target joint positions.
            q: The current joint positions.
            dq: The current joint velocities.
            target_dq: The target joint velocities (optional).
        """
        tau = np.clip(tau, -self.cfg.tau_limit, self.cfg.tau_limit)
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts an observation from the MuJoCo world state.

        # TODO: That should be on the policy side

        Returns:
            A tuple containing the following:
            - dof_pos: The joint positions.
            - dof_vel: The joint velocities.
            - orientation: The orientation of the robot.
            - ang_vel: The angular velocity of the robot.
        """
        # test this
        dof_pos = self.data.qpos.astype(np.double)  # dofx1
        dof_vel = self.data.qvel.astype(np.double)  # dofx1
        # change to quaternion with w at the end
        orientation = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)  # 4x1
        ang_vel = self.data.sensor("angular-velocity").data.astype(np.double)  # 3x1

        # Extract only dof
        dof_pos = dof_pos[-self.cfg.num_actions :]
        dof_vel = dof_vel[-self.cfg.num_actions :]

        return dof_pos, dof_vel, orientation, ang_vel

    def simulate(self, policy: SimPolicy) -> None:

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for step in tqdm(range(int(cfg.duration / cfg.dt)), desc="Simulating..."):
                # Get the world state
                dof_pos, dof_vel, orientation, ang_vel = self.get_observation()

                # Zero action
                # target_dof_pos = dof_pos

                # We update the policy at a lower frequency
                # The simulation runs at 1000hz, but the policy is updated at 100hz
                if step % cfg.decimation == 0:
                    action = policy.next_action(dof_pos, dof_vel, orientation, ang_vel, step)
                    target_dof_pos = action
                # set target_dof_pos to 0s
                # target_dof_pos = np.zeros_like(target_dof_pos)
                tau = policy.pd_control(target_dof_pos, dof_pos, cfg.kps, dof_vel, cfg.kds)
                # breakpoint()
                if step % 1000 == 0:
                    breakpoint()
                # set tau to zero for now
                # tau = np.zeros_like(tau)
                self.step(tau=tau)
                viewer.sync()

        viewer.close()


class IsaacWorld(World):
    """Simulated world using Isaac.

    Attributes:
        cfg: The robot configuration.
    """

    def __init__(self, cfg: RobotConfig):
        # Arguments are overwritten
        delattr(args, "world")
        delattr(args, "load_model")

        # adapt
        args.task = "only_legs_ppo"
        args.device = "cpu"
        args.physics_engine = gymapi.SIM_PHYSX
        args.num_envs = 1
        args.subscenes = 0
        args.use_gpu = False
        args.use_gpu_pipeline = False
        args.num_threads = 1
        args.sim_device = "cpu"
        args.headless = False
        args.seed = None
        args.resume = False
        args.max_iterations = None
        args.experiment_name = None
        args.run_name = "v1"
        args.load_run = None
        args.checkpoint = None  # "300"
        args.rl_device = "cpu"

        args.load_run = "Apr04_17-35-43_height_right3"
        args.run_name = "height_right3"

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        # override some parameters for testing
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.sim.max_gpu_contact_pairs = 2**10
        # env_cfg.terrain.mesh_type = 'trimesh'
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 5
        env_cfg.noise.add_noise = True  # True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.joint_angle_noise = 0.0
        env_cfg.noise.curriculum = False
        env_cfg.noise.noise_level = 0.5

        train_cfg.seed = 123145
        print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

        env, _ = task_registry.make_env(name=args.task, env_cfg=env_cfg, args=args)
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
        self.env = env

        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            train_cfg=train_cfg,
            args=args,
        )
        # Load it from the jit
        self.policy = ppo_runner.get_inference_policy(device=args.device)

        # Load it up
        env.commands[:, 0] = 0.0  # y axis + down - up
        env.commands[:, 1] = 0.5  # x axis - left + right
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

    def step(
        self,
        obs: np.ndarray,
        policy: SimPolicy,
    ) -> None:
        """Performs a simulation step in the Isaac world.

        Args:
            obs: The observation.
            policy: The policy to use.
        """
        actions = policy(obs.detach())  # * 0.
        return actions

    def get_observation(self, actions) -> np.ndarray:
        obs, _, _, _, _ = self.env.step(actions.detach())
        return obs

    def simulate(self, policy=None) -> None:
        for step in tqdm(range(int(cfg.duration / cfg.dt)), desc="Simulating..."):
            actions = self.step(obs)
            obs = self.get_observation(actions)


def main(args: argparse.Namespace, cfg: RobotConfig) -> None:
    """
    Run the policy simulation using the provided policy and configuration.

    Args:
        args: The arguments containing the path to the model.
    """
    if args.world == Worlds.MUJOCO.name:
        world = MujocoWorld(cfg)
        policy = SimPolicy(args.load_model, cfg)
    elif args.world == Worlds.ISAAC.name:
        world = IsaacWorld(cfg)
        policy = None
    else:
        raise ValueError("Invalid world type.")

    world.simulate(policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment policy.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--world", type=str, default="MUJOCO", help="Type of deployment.")
    args = parser.parse_args()

    dof = len(StompyFixed.default_standing())
    robot_path = stompy_mjcf_path(legs_only=True)
    num_single_obs = dof * 3 + 11

    # is2ac, lets scale kps and kds to be the same as our stiffness and dmaping
    # kps = np.ones((dof), dtype=np.double)
    # kds = np.ones((dof), dtype=np.double) * 1
    # tau_limit = np.ones((dof), dtype=np.double)
    kps = np.ones((dof), dtype=np.double) * 200
    kds = np.ones((dof), dtype=np.double) * 10
    tau_limit = np.ones((dof), dtype=np.double) * 200

    cfg = RobotConfig(
        robot_model_path=robot_path, dof=dof, kps=kps, kds=kds, tau_limit=tau_limit, num_single_obs=num_single_obs
    )
    main(args, cfg)
