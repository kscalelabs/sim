import time
import multiprocessing as mp
from tqdm import tqdm

from sim.sim2sim.optimizers.bayesian import BayesianOptimizer, OptimizeParam
from sim.sim2sim.mujoco.play import run_simulation
from sim.envs.base.mujoco_env import MujocoCfg, MujocoEnv


# def run_parallel_sims(num_threads: int) -> None:
#     """Run multiple simulation instances in parallel with a delay between each start."""
#     processes = []
#     delay_between_starts = 0.1

#     for idx in range(num_threads):
#         p = mp.Process(target=run_simulation, args=(idx, args))
#         p.start()
#         processes.append(p)
#         time.sleep(delay_between_starts)  # Introduce a delay before starting the next process

#     # Wait for all processes to finish with a progress bar
#     for p in tqdm(processes, desc="Running parallel simulations"):
#         p.join()


if __name__ == "__main__":
    import numpy as np
    import onnxruntime as ort
    from kinfer.export.pytorch import export_to_onnx
    from kinfer.inference.python import ONNXModel
    from scipy.spatial.transform import Rotation as R

    from sim.model_export import ActorCfg
    from sim.sim2sim.helpers import get_actor_policy
    from sim.utils.cmd_manager import CommandManager

    cfg = MujocoCfg()
    cfg.env.num_envs = 1
    env = MujocoEnv(cfg, render=True)
    cmd_manager = CommandManager(
        num_envs=1, # cfg.env.num_envs,
        mode="fixed", default_cmd=[0.0, 0.0, 0.0, 0.0]
    )
    LOAD_MODEL_PATH = "policy_1.pt" 

    policy_cfg = ActorCfg(embodiment=cfg.asset.name)

    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")
    
    # gains.kp_scale: 1.566
    # gains.kd_scale: 9.312
    # 940 -> {'gains.kp_scale': 2.7692284239249987, 'gains.kd_scale': 17.73252701327175, 'gains.tau_factor': 3.1723978203087935}
    
    """
    Best parameters found:
    gains.kp_scale: 1.445
    gains.kd_scale: 19.975
    gains.tau_factor: 6.350
    """
    parameters = [
        OptimizeParam(
            name='gains.kp_scale',
            min_val=0.5,
            max_val=10.0,
        ),
        OptimizeParam(
            name='gains.kd_scale',
            min_val=4.0,
            max_val=30.0,
        ),
        OptimizeParam(
            name='gains.tau_factor',
            min_val=1.0,
            max_val=10.0,
        ),
    ]
    optimizer = BayesianOptimizer(
        base_cfg=cfg,
        policy=policy,
        parameters=parameters,
        cmd_manager=cmd_manager,
        n_initial_points=200,
        n_iterations=1000,
        exploration_weight=0.25
    )
    
    best_params, history = optimizer.optimize()
    
    print("\nOptimization complete!")
    print("Best parameters found:")
    for name, value in best_params.items():
        print(f"{name}: {value:.3f}")
    
    # Optional: Save results
    import json
    with open("optimization_results.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "history": history
        }, f, indent=2)
