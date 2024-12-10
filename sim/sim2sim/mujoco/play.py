import numpy as np
import onnxruntime as ort
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from sim.env_helpers import debug_robot_state
from sim.envs.base.mujoco_env import MujocoCfg, MujocoEnv
from sim.utils.cmd_manager import CommandManager

np.set_printoptions(precision=2, suppress=True)


def run_simulation(env: MujocoEnv, policy: ort.InferenceSession, cfg: MujocoCfg, cmd_manager: CommandManager, num_episodes: int = 1) -> float:
    """Run a policy in the Mujoco environment."""
    total_reward = 0
    rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        count = 0
        episode_reward = 0

        target_q = np.zeros(env.num_joints)
        prev_actions = np.zeros(env.num_joints)
        hist_obs = np.zeros(policy.get_metadata()["num_observations"])

        while count * cfg.sim.dt * cfg.control.decimation < cfg.env.episode_length_s:
            q, dq, quat, v, omega, euler = obs
            phase = count * cfg.sim.dt / cfg.rewards.cycle_time
            command_input = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase), 0.0, 0.0, 0.0])
            obs_buf = np.concatenate([command_input, q, dq, prev_actions, omega, euler])
            policy_output = policy(
                {
                    "x_vel.1": np.array([command_input[2]], dtype=np.float32),
                    "y_vel.1": np.array([command_input[3]], dtype=np.float32),
                    "rot.1": np.array([command_input[4]], dtype=np.float32),
                    "t.1": np.array([count * cfg.sim.dt], dtype=np.float32),
                    "dof_pos.1": q.astype(np.float32),
                    "dof_vel.1": dq.astype(np.float32),
                    "prev_actions.1": prev_actions.astype(np.float32),
                    "imu_ang_vel.1": omega.astype(np.float32),
                    "imu_euler_xyz.1": euler.astype(np.float32),
                    "buffer.1": hist_obs.astype(np.float32),
                }
            )

            target_q = policy_output["actions_scaled"]
            prev_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]

            obs, reward, done, info = env.step(target_q)
            episode_reward += reward
            count += 1

            if count % 17 == 0:
                pass  # debug_robot_state("Mujoco", obs_buf, actions=target_q)

            if done:
                # print(f"Episode {episode + 1} finished with reward: {episode_reward}")
                total_reward += episode_reward
                rewards.append(episode_reward)
                break
    
    env.close()

    average_reward = total_reward / num_episodes
    # print(f"Average reward over {num_episodes} episodes: {average_reward}")
    return rewards


def run_experiment(env: MujocoEnv, policy: ort.InferenceSession, cfg: MujocoCfg, cmd_manager: CommandManager):
    kp_scale_range = np.linspace(0.1, 5.0, 40)
    kd_scale_range = np.linspace(0.1, 5.0, 40)
    kp_mesh, kd_mesh = np.meshgrid(kp_scale_range, kd_scale_range)
    gains_combinations = np.column_stack((kp_mesh.flatten(), kd_mesh.flatten()))

    results = []
    for i, (kp, kd) in enumerate(gains_combinations):
        cfg.gains.kp_scale = kp
        cfg.gains.kd_scale = kd
        rewards = run_simulation(env, policy, cfg, cmd_manager, num_episodes=10)
        mean_reward = np.mean(rewards)
        results.append(mean_reward)
        print(f"Experiment {i}/{len(gains_combinations)} -> (kds: {kd}, kps: {kp}, reward: {mean_reward:.2f})")

    results = np.array(results)
    kp_interp = np.linspace(kp_scale_range.min(), kp_scale_range.max(), 100)
    kd_interp = np.linspace(kd_scale_range.min(), kd_scale_range.max(), 100)
    kp_mesh_fine, kd_mesh_fine = np.meshgrid(kp_interp, kd_interp)

    rewards_smooth = griddata(
        gains_combinations, 
        results, 
        (kp_mesh_fine, kd_mesh_fine), 
        method='cubic'
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(
        rewards_smooth,
        extent=[kp_scale_range.min(), kp_scale_range.max(), 
                kd_scale_range.min(), kd_scale_range.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    plt.colorbar(label='Mean Reward')
    plt.xlabel('KP Scale')
    plt.ylabel('KD Scale')
    plt.title('Reward Heatmap for PD Gains')

    plt.contour(
        kp_mesh_fine, kd_mesh_fine, rewards_smooth,
        levels=10,
        colors='white',
        alpha=0.3,
        linestyles='solid'
    )

    best_idx = np.argmax(results)
    best_kp = gains_combinations[best_idx][0]
    best_kd = gains_combinations[best_idx][1]
    plt.plot(best_kp, best_kd, 'r*', markersize=15, label='Best Gains')

    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nBest configuration found:")
    print(f"KP Scale: {best_kp:.2f}")
    print(f"KD Scale: {best_kd:.2f}")
    print(f"Mean Reward: {results[best_idx]:.2f}")


if __name__ == "__main__":
    import numpy as np
    from kinfer.export.pytorch import export_to_onnx
    from kinfer.inference.python import ONNXModel
    from scipy.spatial.transform import Rotation as R

    from sim.model_export import ActorCfg
    from sim.sim2sim.helpers import get_actor_policy

    """
    Current parameters:
    gains.kp_scale: 1.445
    gains.kd_scale: 19.975
    gains.tau_factor: 6.350
    """

    cfg = MujocoCfg()
    cfg.gains.kp_scale = 4.6  # 1.643  # 1.445
    cfg.gains.kd_scale = 7.5  # 29.027  # 19.975
    cfg.gains.tau_factor = 4
    cfg.env.num_envs = 1
    render = False
    env = MujocoEnv(cfg, render=render)
    cmd_manager = CommandManager(num_envs=cfg.env.num_envs, mode="fixed", default_cmd=[0.0, 0.0, 0.0, 0.0])
    LOAD_MODEL_PATH = "examples/experiments/standing/robustv1/policy_1.pt"

    policy_cfg = ActorCfg(embodiment=cfg.asset.name)

    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")
    
    rewards = run_simulation(MujocoEnv(cfg, render=True), policy, cfg, cmd_manager, num_episodes=1)
    print(f"Mean reward: {np.mean(rewards)}")
    run_experiment(env, policy, cfg, cmd_manager)

    # kp_scale_range = np.linspace(0.5, 10.0, 100)
    # kd_scale_range = np.linspace(0.2, 10.0, 100)
    # np.random.shuffle(kp_scale_range)
    # np.random.shuffle(kd_scale_range)
    # for i in range(100):
    #     cfg.gains.kp_scale = kp_scale_range[i]
    #     cfg.gains.kd_scale = kd_scale_range[i]
    #     rewards = run_simulation(env, policy, cfg, cmd_manager, num_episodes=1)
    #     print(f"{i} -> (kds: {cfg.gains.kd_scale}, kps: {cfg.gains.kp_scale})")

    # 9 -> (kds: 0.8929292929292931, kps: 8.656565656565656)
