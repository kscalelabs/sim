
import numpy as np
import onnxruntime as ort

from sim.env_helpers import debug_robot_state
from sim.utils.cmd_manager import CommandManager
from sim.envs.base.mujoco_env import MujocoCfg, MujocoEnv

np.set_printoptions(precision=2, suppress=True)


def run_simulation(env: MujocoEnv, policy: ort.InferenceSession, cfg: MujocoCfg, cmd_manager: CommandManager) -> None:
    """
    Run a policy in the Mujoco environment.
    
    Args:
        env: MujocoEnv instance
        policy: ONNX policy for controlled simulation
        cfg: Simulation configuration
    """
    obs = env.reset()
    count = 0
    fall_count = 0
    
    # Initialize policy state
    target_q = np.zeros(env.num_joints)
    prev_actions = np.zeros(env.num_joints)
    hist_obs = np.zeros(policy.get_metadata()["num_observations"])
    
    commands = cmd_manager.update(cfg.sim.dt)
    
    while count * cfg.sim.dt < cfg.env.episode_length_s:
        q, dq, quat, v, omega, euler = obs
        phase = count * cfg.sim.dt / cfg.rewards.cycle_time
        command_input = np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase), 0, 0, 0]
        )
        obs_buf = np.concatenate([command_input, q, dq, prev_actions, omega, euler])
        policy_output = policy({
            "x_vel.1": np.array([command_input[2]], dtype=np.float32),
            "y_vel.1": np.array([command_input[3]], dtype=np.float32),
            "rot.1": np.array([command_input[4]], dtype=np.float32),
            "t.1": np.array([count * cfg.sim.dt], dtype=np.float32),
            "dof_pos.1": (q - env.default_joint_pos).astype(np.float32),
            "dof_vel.1": dq.astype(np.float32),
            "prev_actions.1": prev_actions.astype(np.float32),
            "imu_ang_vel.1": omega.astype(np.float32),
            "imu_euler_xyz.1": euler.astype(np.float32),
            "buffer.1": hist_obs.astype(np.float32),
        })
        
        target_q = policy_output["actions_scaled"]
        prev_actions = policy_output["actions"]
        hist_obs = policy_output["x.3"]

        commands = cmd_manager.update(cfg.sim.dt)
        
        obs, _, done, info = env.step(target_q)
        count += 1
        
        if info.get('fall', False):
            print("Robot fell, resetting...")
            fall_count += 1
            obs = env.reset()
            target_q = np.zeros(env.num_joints)
            prev_actions = np.zeros(env.num_joints)
            hist_obs = np.zeros(policy.get_metadata()["num_observations"])
        
        if count % 17 == 0:
            print(obs_buf)
            debug_robot_state(obs_buf, policy_output["actions"])
    
    print(f"\nSimulation complete:")
    print(f"Duration: {count * cfg.sim.dt:.1f}s")
    print(f"Falls: {fall_count}")


if __name__ == "__main__":
    import numpy as np
    from kinfer.export.pytorch import export_to_onnx
    from kinfer.inference.python import ONNXModel
    from scipy.spatial.transform import Rotation as R

    from sim.model_export import ActorCfg
    from sim.sim2sim.helpers import get_actor_policy

    cfg = MujocoCfg()
    cfg.env.num_envs = 1
    env = MujocoEnv(cfg, render=True)
    cmd_manager = CommandManager(num_envs=cfg.env.num_envs, mode="fixed", default_cmd=[0.0, 0.0, 0.0, 0.0])
    LOAD_MODEL_PATH = "policy_1.pt" 

    policy_cfg = ActorCfg(embodiment=cfg.asset.name)

    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")

    run_simulation(env, policy, cfg, cmd_manager)
