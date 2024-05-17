"""Basic sim2sim and sim2real deployment script.

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
from tqdm import tqdm
from sim.env import stompy_mjcf_path

# from humanoid.envs import *
# from humanoid.utils import Logger, task_registry
# from isaacgym.torch_utils import *
# from isaacgym import gymapi

from policy import SimPolicy

from sim.deploy.config import RobotConfig


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
        mujoco.mj_step(self.model, self.data)

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
        dof_pos = self.data.qpos.astype(np.double) # dofx1
        dof_vel = self.data.qvel.astype(np.double) # dofx1
        # change to quaternion with w at the end
        orientation = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double) # 4x1
        ang_vel = self.data.sensor("angular-velocity").data.astype(np.double) # 3x1

        # Extract only dof
        dof_pos = dof_pos[-self.cfg.num_actions:]
        dof_vel = dof_vel[-self.cfg.num_actions:]

        return dof_pos, dof_vel, orientation, ang_vel

    def simulate(self, policy: SimPolicy) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for step in tqdm(range(int(cfg.duration / cfg.dt)), desc="Simulating..."):
                # Get the world state
                dof_pos, dof_vel, orientation, ang_vel = self.get_observation()
    
                # We update the policy at a lower frequency
                # The simulation runs at 1000hz, but the policy is updated at 100hz
                if step % cfg.decimation == 0:
                    action = policy.next_action(dof_pos, dof_vel, orientation, ang_vel, step)
                    target_dof_pos = action * cfg.action_scale
            
                tau = policy.pd_control(target_dof_pos, dof_pos, cfg.kps, dof_vel, cfg.kds)

                self.step(tau=tau)
                viewer.sync()
        
        viewer.close()


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
        policy = SimPolicy(args.load_model, cfg)
    else:
        raise ValueError("Invalid world type.")

    world.simulate(policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment policy.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--world", type=str, default="MUJOCO", help="Type of deployment.")
    args = parser.parse_args()

    if "xbot" in args.load_model:
        robot_path = f"/Users/pfb30/sim/third_party/humanoid-gym/resources/robots/XBot/mjcf/XBot-L.xml"
        dof: int = 12
        kps: np.ndarray = np.array(
            [200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
        kds: np.ndarray = np.array(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
        tau_limit: np.ndarray = 200. * np.ones(dof, dtype=np.double)
        num_single_obs = dof * 3 + 11
    else:
        dof=17
        robot_path = stompy_mjcf_path()
        num_single_obs = dof * 3 + 11

        kps = np.ones((dof), dtype=np.double) * 200
        kds = np.ones((dof), dtype=np.double) * 10
        tau_limit = np.ones((dof), dtype=np.double) * 200

    # # TODO - this should be read from the config
    # robot_path = f"/Users/pfb30/sim/stompy/robot_new.xml"
    # num_single_obs = dof * 3 + 11
    # dof = 
    # kps = np.ones((dof), dtype=np.double) * 200
    # kds = np.ones((dof), dtype=np.double) * 10
    # tau_limit = np.ones((dof), dtype=np.double) * 200
    cfg = RobotConfig(
        robot_model_path=robot_path,
        dof=dof, kps=kps, kds=kds, tau_limit=tau_limit, 
        num_single_obs=num_single_obs
    )
    main(args, cfg)
