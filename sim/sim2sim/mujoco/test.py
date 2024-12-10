import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sim import ROOT_DIR
from sim.utils.args_parsing import parse_args_with_extras
from sim.model_export import ActorCfg
from sim.sim2sim.helpers import get_actor_policy

from sim.envs.humanoids.stompymicro_config import StompyMicroCfgPPO, StompyMicroCfg
from sim.envs.humanoids.stompymicro_env import StompyMicroEnv
from sim.utils.helpers import (
    class_to_dict,
    get_load_path,
    parse_sim_params,
)
from sim.algo.ppo.on_policy_runner import OnPolicyRunner
from kinfer.export.pytorch import export_to_onnx  # isort: skip
from kinfer.inference.python import ONNXModel  # isort: skip
import torch  # isort: skip

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3)


def test_policies(args):
    LOAD_MODEL_PATH = "examples/experiments/standing/robustv1/policy_1.pt"
    DEVICE = "cuda:0"
    
    ## ISAAC ##
    args.load_model = LOAD_MODEL_PATH
    env_cfg, train_cfg = StompyMicroCfg(), StompyMicroCfgPPO()
    env_cfg.env.num_envs = 1
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = -8
    train_cfg.seed = 0
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = StompyMicroEnv(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    train_cfg_dict = class_to_dict(train_cfg)
    env_cfg_dict = class_to_dict(env_cfg)
    all_cfg = {**train_cfg_dict, **env_cfg_dict}
    ppo_runner = OnPolicyRunner(env, all_cfg, log_dir=None, device=DEVICE)
    resume_path = get_load_path(
        root=os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name), load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint
    )
    ppo_runner.load(resume_path, load_optimizer=False)
    ppo_runner.alg.actor_critic.eval()
    ppo_runner.alg.actor_critic.to(DEVICE)
    policy = ppo_runner.alg.actor_critic.act_inference

    obs_buf = torch.zeros((1, 885)).to(DEVICE)
    with torch.no_grad():
        isaac_actions = policy(obs_buf)
        print(isaac_actions)
        isaac_actions = isaac_actions.cpu().numpy()[0]
    ## ISAAC ##

    ## MUJOCO ##
    policy_cfg = ActorCfg(embodiment="stompymicro")
    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    mujoco_policy = ONNXModel("kinfer_test.onnx")
    num_actions = sim2sim_info["num_actions"]
    mujoco_inputs = {
        "x_vel.1": np.zeros(1, dtype=np.float32),
        "y_vel.1": np.zeros(1, dtype=np.float32),
        "rot.1": np.zeros(1, dtype=np.float32),
        "t.1": np.zeros(1, dtype=np.float32),
        "dof_pos.1": np.zeros(num_actions, dtype=np.float32),
        "dof_vel.1": np.zeros(num_actions, dtype=np.float32),
        "prev_actions.1": np.zeros(num_actions, dtype=np.float32),
        "imu_ang_vel.1": np.zeros(3, dtype=np.float32),
        "imu_euler_xyz.1": np.zeros(3, dtype=np.float32),
        "buffer.1": np.zeros(sim2sim_info["num_observations"], dtype=np.float32),
    }
    mujoco_outputs = mujoco_policy(mujoco_inputs)
    ## MUJOCO ##

    ## COMPARE ##
    print("\nPolicy Output Comparison:")
    print("-" * 50)
    print("\nMujoco actions:")
    print(mujoco_outputs["actions_scaled"])
    print("\nIsaac actions:")
    print(isaac_actions)
    print("\nDifference metrics:", np.abs(mujoco_outputs["actions_scaled"] - isaac_actions).mean())
    plt.plot(mujoco_outputs["actions_scaled"], label="Mujoco")
    plt.plot(isaac_actions, label="Isaac")
    plt.legend()
    plt.show()
    ## COMPARE ##

    return mujoco_outputs, isaac_actions


if __name__ == "__main__":
    args = parse_args_with_extras(lambda x: x)
    print("Arguments:", vars(args))
    test_policies(args)
