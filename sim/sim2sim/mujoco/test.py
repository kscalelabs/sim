import numpy as np
from sim.utils.args_parsing import parse_args_with_extras
from sim.envs import task_registry
from kinfer.inference.python import ONNXModel
from kinfer.export.pytorch import export_to_onnx
from sim.sim2sim.helpers import get_actor_policy
from sim.model_export import ActorCfg
import torch # isort: skip


def setup_isaac_policy(args):
    """Setup Isaac Gym policy following original implementation."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    return policy, env


def test_policies(args):
    """Compare outputs from Isaac and Mujoco policies given identical inputs."""
    # Load Isaac policy exactly as in coplay.py
    isaac_policy, env = setup_isaac_policy(args)
    
    LOAD_MODEL_PATH = "examples/experiments/standing/robustv1/policy_1.pt"
    policy_cfg = ActorCfg(embodiment="stompymicro")
    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    print(input_tensors)
    
    # Export and load ONNX model
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print("Export config:", export_config)
    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    mujoco_policy = ONNXModel("kinfer_test.onnx")
    
    # Create zeroed input state
    x_vel = y_vel = rot = 0.0
    t = 0.0
    dof_pos = np.zeros(16)
    dof_vel = np.zeros(16)
    prev_actions = np.zeros(16)
    imu_ang_vel = np.zeros(3)
    imu_euler_xyz = np.zeros(3)
    hist_obs = np.zeros(mujoco_policy.get_metadata()["num_observations"])

    # Prepare input for Mujoco policy
    mujoco_inputs = {
        "x_vel.1": np.array([x_vel], dtype=np.float32),
        "y_vel.1": np.array([y_vel], dtype=np.float32),
        "rot.1": np.array([rot], dtype=np.float32),
        "t.1": np.array([t], dtype=np.float32),
        "dof_pos.1": dof_pos.astype(np.float32),
        "dof_vel.1": dof_vel.astype(np.float32),
        "prev_actions.1": prev_actions.astype(np.float32),
        "imu_ang_vel.1": imu_ang_vel.astype(np.float32),
        "imu_euler_xyz.1": imu_euler_xyz.astype(np.float32),
        "buffer.1": hist_obs.astype(np.float32),
    }

    # Get Mujoco policy output
    mujoco_outputs = mujoco_policy(mujoco_inputs)
    mujoco_actions = mujoco_outputs["actions"]
    mujoco_actions_scaled = mujoco_outputs["actions_scaled"]

    # Prepare input for Isaac policy
    command = torch.zeros(1, 4, device=env.device)  # [lin_vel_x, lin_vel_y, ang_vel_yaw, jumping_signal]
    obs_buf = torch.zeros(1, env.num_obs, device=env.device)
    
    # Get Isaac policy output
    with torch.no_grad():
        isaac_actions = isaac_policy(obs_buf)
        isaac_actions = isaac_actions.cpu().numpy()

    # Print comparison
    print("\nPolicy Output Comparison:")
    print("-" * 50)
    print("Mujoco raw actions:")
    print(mujoco_actions)
    print("\nMujoco scaled actions:")
    print(mujoco_actions_scaled)
    print("\nIsaac actions:")
    print(isaac_actions)
    
    # Compare differences
    print("\nDifferences:")
    print("-" * 50)
    print("Max difference (Mujoco raw vs Isaac):", np.max(np.abs(mujoco_actions - isaac_actions)))
    print("Mean difference (Mujoco raw vs Isaac):", np.mean(np.abs(mujoco_actions - isaac_actions)))
    print("Max difference (Mujoco scaled vs Isaac):", np.max(np.abs(mujoco_actions_scaled - isaac_actions)))
    print("Mean difference (Mujoco scaled vs Isaac):", np.mean(np.abs(mujoco_actions_scaled - isaac_actions)))


def add_arguments(parser):
    """Add test-specific arguments."""
    parser.add_argument(
        "--command_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oscillating", "random", "keyboard"],
        help="Control mode for the robot",
    )


if __name__ == "__main__":
    args = parse_args_with_extras(add_arguments)
    print("Arguments:", vars(args))
    test_policies(args)
