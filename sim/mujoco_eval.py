"""Sim2sim deployment evaluation.

Completely self-contained.

Run:
    python sim/mujoco_eval.py --load_model examples/gpr_walking.kinfer --embodiment gpr
    python sim/mujoco_eval.py --load_model kinfer_policy.onnx --embodiment zbot2
"""

import argparse
import dataclasses
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple, TypedDict

import gpytorch
import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
import pygame
import torch
from kinfer.inference.python import ONNXModel
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


@dataclass
class ModelInfo(TypedDict):
    num_actions: int
    num_observations: int
    robot_effort: List[float]
    robot_stiffness: List[float]
    robot_damping: List[float]
    sim_dt: float
    sim_decimation: int
    pd_decimation: int
    tau_factor: float


def handle_keyboard_input() -> None:
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd

    keys = pygame.key.get_pressed()

    # Update movement commands based on arrow keys
    if keys[pygame.K_UP]:
        x_vel_cmd += 0.0005
    if keys[pygame.K_DOWN]:
        x_vel_cmd -= 0.0005
    if keys[pygame.K_LEFT]:
        y_vel_cmd += 0.0005
    if keys[pygame.K_RIGHT]:
        y_vel_cmd -= 0.0005

    # Yaw control
    if keys[pygame.K_a]:
        yaw_vel_cmd += 0.001
    if keys[pygame.K_z]:
        yaw_vel_cmd -= 0.001


def quaternion_to_euler_array(quat: np.ndarray) -> np.ndarray:
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Get the gravity orientation from the quaternion.

    Args:
        quaternion: np.ndarray[float, float, float, float]

    Returns:
        gravity_orientation: np.ndarray[float, float, float]
    """
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


@dataclass
class RealWorldParams:
    """Parameters to simulate real-world conditions."""

    sensor_latency: int = 2  # frames of delay for sensor readings
    motor_latency: int = 1  # frames of delay for motor commands
    sensor_noise_std: float = 0.01  # standard deviation of sensor noise
    motor_noise_std: float = 0.02  # standard deviation of motor noise
    init_pos_noise_std: float = 0.03  # standard deviation of initial position noise


class SensorBuffer:
    """Ring buffer to simulate sensor latency."""

    def __init__(self, size: int, shape: tuple):
        self.buffer = np.zeros((size,) + shape)
        self.size = size
        self.idx = 0

    def push(self, data: np.ndarray) -> None:
        self.buffer[self.idx] = data
        self.idx = (self.idx + 1) % self.size

    def get(self) -> np.ndarray:
        return self.buffer[self.idx]


def get_obs(data: mujoco.MjData, sensor_buffer: SensorBuffer, params: RealWorldParams) -> Tuple[np.ndarray, ...]:
    """Extracts an observation from the mujoco data structure with added noise and latency."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    base_vel = data.qvel[:3].astype(np.double)  # Store base velocity before adding noise

    # Add sensor noise
    q += np.random.normal(0, params.sensor_noise_std, q.shape)
    dq += np.random.normal(0, params.sensor_noise_std, dq.shape)
    quat += np.random.normal(0, params.sensor_noise_std, quat.shape)
    omega += np.random.normal(0, params.sensor_noise_std, omega.shape)
    base_vel += np.random.normal(0, params.sensor_noise_std, base_vel.shape)

    # Normalize quaternion after adding noise
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-6:  # If quaternion is too close to zero
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default to identity rotation
    else:
        quat = quat / quat_norm

    # Store in buffer to simulate latency
    sensor_data = np.concatenate([q, dq, quat, omega, base_vel])
    sensor_buffer.push(sensor_data)
    delayed_data = sensor_buffer.get()

    # Unpack delayed data
    q = delayed_data[: len(q)]
    dq = delayed_data[len(q) : len(q) + len(dq)]
    quat = delayed_data[len(q) + len(dq) : len(q) + len(dq) + 4]
    omega = delayed_data[len(q) + len(dq) + 4 : len(q) + len(dq) + 4 + 3]
    base_vel = delayed_data[len(q) + len(dq) + 4 + 3 :]

    # Ensure quaternion is normalized after delay and handle zero case
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-6:  # If quaternion is too close to zero
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default to identity rotation
    else:
        quat = quat / quat_norm

    try:
        r = R.from_quat(quat)
    except ValueError:
        print(f"Warning: Invalid quaternion detected: {quat}, norm: {np.linalg.norm(quat)}")
        # Fall back to identity rotation
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        r = R.from_quat(quat)

    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    v = r.apply(base_vel, inverse=True).astype(np.double)

    return (q, dq, quat, v, omega, gvec)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands."""
    return kp * (target_q + default - q) - kd * dq


def run_mujoco(
    embodiment: str,
    policy: ort.InferenceSession,
    model_info: ModelInfo,
    real_world_params: RealWorldParams,
    keyboard_use: bool = False,
    render: bool = True,
    sim_duration: float = 60.0,
    terrain: bool = False,
    random_pushes: bool = False,
    push_magnitude: float = 100.0,  # Maximum force magnitude in Newtons
    push_duration: float = 0.2,  # Duration of push in seconds
    min_push_interval: float = 1.0,  # Minimum time between pushes
    max_push_interval: float = 3.0,  # Maximum time between pushes
) -> bool:
    """Run the Mujoco simulation using the provided policy and configuration."""
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    if terrain:
        mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed_terrain.xml"
    else:
        mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = model_info["sim_dt"]
    data = mujoco.MjData(model)

    assert isinstance(model_info["num_actions"], int)
    assert isinstance(model_info["num_observations"], int)
    assert isinstance(model_info["robot_effort"], list)
    assert isinstance(model_info["robot_stiffness"], list)
    assert isinstance(model_info["robot_damping"], list)

    # tau_limit = np.array(list(model_info["robot_effort"]) + list(model_info["robot_effort"])) * model_info["tau_factor"]
    # kps = np.array(list(model_info["robot_stiffness"]) + list(model_info["robot_stiffness"]))
    # kds = np.array(list(model_info["robot_damping"]) + list(model_info["robot_damping"]))
    # HACKY FOR HEADLESS
    efforts = model_info["robot_effort"]
    stiffnesses = model_info["robot_stiffness"]
    dampings = model_info["robot_damping"]
    leg_lims = [efforts[0], efforts[1], efforts[1], efforts[0], efforts[2]]
    tau_limit = np.array(leg_lims + leg_lims) * model_info["tau_factor"]

    leg_kps = [stiffnesses[0], stiffnesses[1], stiffnesses[1], stiffnesses[0], stiffnesses[2]]
    kps = np.array(leg_kps + leg_kps)

    leg_kds = [dampings[0], dampings[1], dampings[1], dampings[0], dampings[2]]
    kds = np.array(leg_kds + leg_kds)
    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-model_info["num_actions"] :]
        # Add noise to initial position
        default += np.random.normal(0, real_world_params.init_pos_noise_std, default.shape)
        print("Default position (with noise):", default)
    except Exception as e:
        print(f"No default position found, using noisy zero initialization: {e}")
        default = np.random.normal(0, real_world_params.init_pos_noise_std, model_info["num_actions"])

    # Initialize sensor buffer for latency simulation
    sensor_shape = (len(data.qpos) + len(data.qvel) + 4 + 3 + 3,)  # qpos + qvel + quat + omega + base_vel
    print(f"Sensor buffer shape: {sensor_shape}, Total size: {sum(sensor_shape)}")
    sensor_buffer = SensorBuffer(real_world_params.sensor_latency, sensor_shape)

    # Get initial valid sensor data
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-6:  # If quaternion is too close to zero
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Default to identity rotation
    else:
        quat = quat / quat_norm
    omega = data.sensor("angular-velocity").data.astype(np.double)
    base_vel = data.qvel[:3].astype(np.double)

    # Create initial sensor data with valid quaternion
    initial_sensor_data = np.concatenate([q, dq, quat, omega, base_vel])

    # Initialize buffer with valid data
    for _ in range(real_world_params.sensor_latency):
        sensor_buffer.push(initial_sensor_data)

    # Buffer for motor command latency
    motor_buffer = SensorBuffer(real_world_params.motor_latency, (model_info["num_actions"],))
    # Initialize motor buffer with zeros
    for _ in range(real_world_params.motor_latency):
        motor_buffer.push(np.zeros(model_info["num_actions"]))

    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((model_info["num_actions"]), dtype=np.double)
    prev_actions = np.zeros((model_info["num_actions"]), dtype=np.double)
    hist_obs = np.zeros((model_info["num_observations"]), dtype=np.double)

    count_lowlevel = 0

    input_data = {
        "x_vel.1": np.zeros(1).astype(np.float32),
        "y_vel.1": np.zeros(1).astype(np.float32),
        "rot.1": np.zeros(1).astype(np.float32),
        "t.1": np.zeros(1).astype(np.float32),
        "dof_pos.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "dof_vel.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "prev_actions.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        # "imu_euler_xyz.1": np.zeros(3).astype(np.float32),
        "projected_gravity.1": np.zeros(3).astype(np.float32),
        # "imu_ang_vel.1": np.zeros(3).astype(np.float32),
        "buffer.1": np.zeros(model_info["num_observations"]).astype(np.float32),
    }

    # Initialize variables for tracking upright steps and average speed
    upright_steps = 0
    total_speed = 0.0
    step_count = 0

    # Initialize push-related variables if random pushes are enabled
    if random_pushes:
        next_push_time = np.random.uniform(min_push_interval, max_push_interval)  # Random time for first push
        current_push_end = 0.0
        current_push_force = np.zeros(3)

    for _ in tqdm(range(int(sim_duration / model_info["sim_dt"])), desc="Simulating..."):
        current_time = _ * model_info["sim_dt"]

        if random_pushes:
            # Check if it's time for a new push
            if current_time >= next_push_time and current_time >= current_push_end:
                # Generate random force direction (in horizontal plane)
                angle = np.random.uniform(0, 2 * np.pi)
                force_magnitude = np.random.uniform(0.3 * push_magnitude, push_magnitude)
                current_push_force = np.array(
                    [force_magnitude * np.cos(angle), force_magnitude * np.sin(angle), 0.0]  # No vertical force
                )

                # Set push duration and next push time
                current_push_end = current_time + push_duration
                next_push_interval = np.random.uniform(min_push_interval, max_push_interval)
                next_push_time = current_time + push_duration + next_push_interval

                # print(
                #     f"\nApplying push at t={current_time:.2f}s: magnitude={force_magnitude:.1f}N, "
                #     f"angle={angle:.1f}rad, next push in {next_push_interval:.1f}s"
                # )

            # Apply force if within push duration
            if current_time < current_push_end:
                data.xfrc_applied[1] = np.concatenate([current_push_force, np.zeros(3)])
            else:
                data.xfrc_applied[1] = np.zeros(6)

        if keyboard_use:
            handle_keyboard_input()

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data, sensor_buffer, real_world_params)
        q = q[-model_info["num_actions"] :]
        dq = dq[-model_info["num_actions"] :]

        # eu_ang = quaternion_to_euler_array(quat)
        # eu_ang[eu_ang > math.pi] -= 2 * math.pi

        # eu_ang = np.array([0.0, 0.0, 0.0])
        # omega = np.array([0.0, 0.0, 0.0])
        # gvec = np.array([0.0, 0.0, -1.0])

        is_upright = gvec[2] < -0.8
        if is_upright:
            upright_steps += 1

        # Calculate speed and accumulate for average speed calculation
        speed = np.linalg.norm(v[:2])  # Speed in the x-y plane
        total_speed += speed.item()
        step_count += 1

        # If robot falls, print statistics and break
        if not is_upright and upright_steps > 0:
            break

        # 1000hz -> 50hz
        if count_lowlevel % model_info["sim_decimation"] == 0:
            # Convert sim coordinates to policy coordinates
            cur_pos_obs = q - default
            cur_vel_obs = dq

            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)

            input_data["t.1"] = np.array([count_lowlevel * model_info["sim_dt"]], dtype=np.float32)

            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)

            input_data["prev_actions.1"] = prev_actions.astype(np.float32)

            input_data["projected_gravity.1"] = gvec.astype(np.float32)
            input_data["imu_ang_vel.1"] = omega.astype(np.float32)
            # input_data["imu_euler_xyz.1"] = eu_ang.astype(np.float32)

            input_data["buffer.1"] = hist_obs.astype(np.float32)

            policy_output = policy(input_data)
            positions = policy_output["actions_scaled"]
            curr_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]

            target_q = positions

            prev_actions = curr_actions

        if count_lowlevel % model_info["pd_decimation"] == 0:
            # Generate PD control with motor noise
            tau = pd_control(target_q, q, kps, dq, kds, default)
            tau = np.clip(tau, -tau_limit, tau_limit)
            tau += np.random.normal(0, real_world_params.motor_noise_std, tau.shape)

            # Simulate motor command latency
            motor_buffer.push(tau)
            tau = motor_buffer.get()

        data.ctrl = tau
        mujoco.mj_step(model, data)

        if render:
            viewer.render()
        count_lowlevel += 1

    if render:
        viewer.close()

    # Calculate average speed
    if step_count > 0:
        average_speed = total_speed / step_count
    else:
        average_speed = 0.0

    # Save or print the statistics at the end of the episode
    print("=" * 100)
    print("PERFORMANCE STATISTICS")
    print("-" * 100)
    print(f"Number of upright steps: {upright_steps} ({upright_steps * model_info['sim_dt']:.2f} seconds)")
    print(f"Average speed: {average_speed:.4f} m/s")
    print("=" * 100)

    return upright_steps == int(sim_duration / model_info["sim_dt"])


def test_robustness(
    embodiment: str,
    policy: ort.InferenceSession,
    model_info: ModelInfo,
    params: RealWorldParams,
    test_duration: float = 10.0,
    push: bool = False,
) -> Tuple[RealWorldParams, bool]:
    """Test if the model can handle given parameters."""
    try:
        success = run_mujoco(
            embodiment=embodiment,
            policy=policy,
            model_info=model_info,
            real_world_params=params,
            render=False,
            sim_duration=test_duration,
            random_pushes=push,
            push_magnitude=50.0,
            push_duration=0.2,
            min_push_interval=1.0,
            max_push_interval=3.0,
        )
    except Exception as e:
        print(f"Failed with error: {e}")
        success = False
    return params, success


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood
    ) -> None:
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1)))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimizer:
    def __init__(self, param_bounds: dict, objective_function: Callable) -> None:
        self.param_bounds = param_bounds
        self.objective_function = objective_function
        self.device = "cpu"

        self.train_x = torch.empty((0, len(param_bounds)), device=self.device)
        self.train_y = torch.empty(0, device=self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model: Optional[ExactGPModel] = None

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [0, 1] range."""
        normalized = torch.zeros_like(params)
        for i, (name, bounds) in enumerate(self.param_bounds.items()):
            normalized[:, i] = (params[:, i] - bounds[0]) / (bounds[1] - bounds[0])
        return normalized

    def _denormalize_params(self, normalized_params: torch.Tensor) -> torch.Tensor:
        """Convert normalized parameters back to original range."""
        denormalized = torch.zeros_like(normalized_params)
        for i, (name, bounds) in enumerate(self.param_bounds.items()):
            denormalized[:, i] = normalized_params[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        return denormalized

    def _expected_improvement(self, mean: torch.Tensor, std: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        """Calculate expected improvement."""
        z = (mean - best_f) / std
        ei = std * (z * torch.distributions.Normal(0, 1).cdf(z) + torch.distributions.Normal(0, 1).log_prob(z).exp())
        return ei

    def _get_next_points(self, n_points: int = 1) -> torch.Tensor:
        """Get next points to evaluate using random sampling and EI."""
        if len(self.train_y) == 0:
            # If no observations, return random points
            return torch.rand(n_points, len(self.param_bounds), device=self.device)

        # Generate random candidates
        n_candidates = 1000
        candidates = torch.rand(n_candidates, len(self.param_bounds), device=self.device)

        assert self.model is not None

        # Get model predictions
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(candidates))
            mean = predictions.mean
            std = predictions.stddev

        best_f = self.train_y.max()
        ei = self._expected_improvement(mean, std, best_f)

        # Return points with highest EI
        top_indices = torch.topk(ei, n_points).indices
        return candidates[top_indices]

    def optimize(self, n_iterations: int = 50) -> Tuple[dict, float]:
        """Run Bayesian optimization."""
        best_params: Optional[dict] = None
        best_value = float("-inf")

        for i in range(n_iterations):
            print(f"\nIteration {i+1}/{n_iterations}")

            # Get next points to evaluate
            next_points = self._get_next_points(n_points=1)
            params = self._denormalize_params(next_points)

            # Convert to dictionary for objective function
            param_dict = {name: params[0, i].item() for i, name in enumerate(self.param_bounds.keys())}

            # Evaluate objective function
            start_time = time.time()
            value = self.objective_function(param_dict)
            print(f"Evaluation took {time.time() - start_time:.2f}s")

            # Update best parameters if necessary
            if value > best_value:
                best_value = value
                best_params = param_dict
                print("New best parameters found!")
                print(f"Parameters: {best_params}")
                print(f"Value: {best_value}")

            self.train_x = torch.cat([self.train_x, next_points])
            self.train_y = torch.cat([self.train_y, torch.tensor([value], device=self.device)])

            if self.model is None:
                self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(self.device)
            else:
                self.model.set_train_data(self.train_x, self.train_y, strict=False)

            self.model.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for _ in range(50):
                optimizer.zero_grad()
                output = self.model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()

        if best_params is None:
            raise ValueError("No best parameters found")

        return best_params, best_value


@dataclass
class OptimizationConfig:
    """Configuration for the Bayesian optimization process."""

    # Maximum parameter bounds
    max_sensor_latency: int = 10
    max_motor_latency: int = 10
    max_sensor_noise: float = 0.2
    max_motor_noise: float = 0.2
    max_init_noise: float = 0.2

    # Parameter weights for optimization reward
    sensor_noise_weight: float = 2.0
    motor_noise_weight: float = 1.0
    init_noise_weight: float = 1.0
    sensor_latency_weight: float = 1.5
    motor_latency_weight: float = 1.0

    # Optimization settings
    n_iterations: int = 50
    n_trials: int = 3  # Number of trials per parameter set
    test_duration: float = 10.0

    # Push test settings
    push: bool = False
    push_magnitude: float = 50.0
    push_duration: float = 0.2
    min_push_interval: float = 1.0
    max_push_interval: float = 3.0


def optimize_params(
    embodiment: str,
    policy: ort.InferenceSession,
    model_info: ModelInfo,
    config: OptimizationConfig,
) -> RealWorldParams:
    """Use Bayesian optimization to find maximum tolerable parameters."""
    print("=" * 100)
    print("IDENTIFYING MAXIMUM RANDOMIZATION PARAMETERS")
    print("-" * 100)
    print("Optimization config:")
    for field, value in asdict(config).items():
        print(f"{field}: {value}")
    print("-" * 100)

    # Define parameter bounds
    param_bounds = {
        "sensor_noise": (0.0, config.max_sensor_noise),
        "motor_noise": (0.0, config.max_motor_noise),
        "init_noise": (0.0, config.max_init_noise),
        "sensor_latency": (1, config.max_sensor_latency),
        "motor_latency": (1, config.max_motor_latency),
    }

    # Define objective function
    def objective(params: dict) -> float:
        test_params = RealWorldParams(
            sensor_latency=round(params["sensor_latency"]),
            motor_latency=round(params["motor_latency"]),
            sensor_noise_std=params["sensor_noise"],
            motor_noise_std=params["motor_noise"],
            init_pos_noise_std=params["init_noise"],
        )

        print("\nTesting parameters:")
        for name, value in asdict(test_params).items():
            print(f"{name}: {value}")

        # Run multiple trials to ensure reliability
        successes = 0
        for trial in range(config.n_trials):
            print(f"\nTrial {trial + 1}/{config.n_trials}")
            success = run_mujoco(
                embodiment=embodiment,
                policy=policy,
                model_info=model_info,
                real_world_params=test_params,
                render=False,
                sim_duration=config.test_duration,
                random_pushes=config.push,
                push_magnitude=config.push_magnitude,
                push_duration=config.push_duration,
                min_push_interval=config.min_push_interval,
                max_push_interval=config.max_push_interval,
            )
            if success:
                successes += 1

        success_rate = successes / config.n_trials
        print(f"\nSuccess rate: {success_rate * 100:.1f}%")

        if success_rate == 1.0:  # Only reward if all trials succeed
            weighted_params = (
                config.sensor_noise_weight * params["sensor_noise"] / config.max_sensor_noise
                + config.motor_noise_weight * params["motor_noise"] / config.max_motor_noise
                + config.init_noise_weight * params["init_noise"] / config.max_init_noise
                + config.sensor_latency_weight * params["sensor_latency"] / config.max_sensor_latency
                + config.motor_latency_weight * params["motor_latency"] / config.max_motor_latency
            ) / sum(
                [
                    config.sensor_noise_weight,
                    config.motor_noise_weight,
                    config.init_noise_weight,
                    config.sensor_latency_weight,
                    config.motor_latency_weight,
                ]
            )
            return weighted_params
        else:
            return -50.0  # big ahh penalty

    # Run Bayesian optimization
    optimizer = BayesianOptimizer(param_bounds, objective)
    best_params, _ = optimizer.optimize(config.n_iterations)

    # Convert to RealWorldParams
    final_params = RealWorldParams(
        sensor_latency=round(best_params["sensor_latency"]),
        motor_latency=round(best_params["motor_latency"]),
        sensor_noise_std=best_params["sensor_noise"],
        motor_noise_std=best_params["motor_noise"],
        init_pos_noise_std=best_params["init_noise"],
    )

    print("\n" + "=" * 100)
    print("MAXIMUM RANDOMIZATION PARAMETERS FOUND:")
    print("-" * 100)
    for name, value in asdict(final_params).items():
        print(f"{name}: {value}")
    print("=" * 100)

    return final_params


def add_optimization_args(parser: argparse.ArgumentParser) -> None:
    """Add OptimizationConfig parameters to an argument parser."""
    # Get all fields from OptimizationConfig
    fields = dataclasses.fields(OptimizationConfig)

    # Create an argument group for optimization parameters
    opt_group = parser.add_argument_group("Optimization Parameters")

    for field in fields:
        # Convert field name from snake_case to kebab-case for args
        arg_name = f"--{field.name.replace('_', '-')}"

        # Get default value and type from field
        default = field.default
        field_type = field.type

        # Handle boolean fields differently
        if field_type is bool:
            opt_group.add_argument(
                arg_name,
                action="store_true" if not default else "store_false",
                help=f"{'Enable' if not default else 'Disable'} {field.name.replace('_', ' ')}",
            )
        else:
            opt_group.add_argument(
                arg_name,
                type=field_type,
                default=default,
                help=f"Set {field.name.replace('_', ' ')} (default: {default})",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to run to load from.")
    parser.add_argument("--keyboard_use", action="store_true", help="Enable keyboard control")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--terrain", action="store_true", help="Enable terrain")
    parser.add_argument("--identify", action="store_true", help="Identify maximum randomization parameters")
    parser.add_argument("--randomize_conditions", action="store_true", help="Randomize conditions")

    # Add optimization arguments automatically
    add_optimization_args(parser)

    args = parser.parse_args()

    if args.keyboard_use:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        pygame.init()
        pygame.display.set_caption("Simulation Control")
    else:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = -0.5, 0.0, 0.0

    policy = ONNXModel(args.load_model)
    metadata = policy.get_metadata()

    # HACKY: (NOT YET ADDED TO METADATA)
    metadata["pd_decimation"] = 10

    # randomize conditions

    if args.randomize_conditions:
        # PD constants
        damping = metadata["robot_damping"]
        damping = list(damping * np.random.uniform(1, 1.005, size=len(damping)))
        stiffness = metadata["robot_stiffness"]
        stiffness = list(stiffness * np.random.uniform(1, 1.005, size=len(stiffness)))

        # pd decimation
        pd_decimation = metadata["sim_decimation"] + np.random.randint(-4, 5)

        # sim_decimation
        sim_decimation = metadata["pd_decimation"] + np.random.randint(-4, 5)
    else:
        damping = metadata["robot_damping"]
        stiffness = metadata["robot_stiffness"]
        sim_decimation = metadata["sim_decimation"]
        pd_decimation = metadata["pd_decimation"]

    print("=" * 100)
    print("Testing with parameters:")
    print(f"stiffness: {stiffness}")
    print(f"damping: {damping}")
    print(f"sim_decimation: {sim_decimation}")
    print(f"pd_decimation: {pd_decimation}")
    print("=" * 100)

    model_info: ModelInfo = {
        "num_actions": metadata["num_actions"],
        "num_observations": metadata["num_observations"],
        "robot_effort": metadata["robot_effort"],
        "robot_stiffness": stiffness,
        "robot_damping": damping,
        "sim_dt": metadata["sim_dt"],
        "sim_decimation": sim_decimation,
        "pd_decimation": pd_decimation,
        "tau_factor": metadata["tau_factor"],
    }

    # Create real-world parameters
    real_world_params = RealWorldParams(
        sensor_latency=4, motor_latency=1, sensor_noise_std=0.04, motor_noise_std=0.04, init_pos_noise_std=0.04
    )

    # real_world_params = RealWorldParams(
    #     sensor_latency=1, motor_latency=1, sensor_noise_std=0.0, motor_noise_std=0.0, init_pos_noise_std=0.0
    # )

    if args.identify:
        # Create optimization config from args
        opt_config = OptimizationConfig(
            **{
                field.name: getattr(args, field.name.replace("-", "_"))
                for field in dataclasses.fields(OptimizationConfig)
            }
        )

        best_params = optimize_params(
            embodiment=args.embodiment,
            policy=policy,
            model_info=model_info,
            config=opt_config,
        )
        real_world_params = best_params

    run_mujoco(
        embodiment=args.embodiment,
        policy=policy,
        model_info=model_info,
        real_world_params=real_world_params,
        keyboard_use=args.keyboard_use,
        render=args.render,
        terrain=args.terrain,
        random_pushes=args.push,
        push_magnitude=args.push_magnitude,
        push_duration=args.push_duration,
        min_push_interval=args.min_push_interval,
        max_push_interval=args.max_push_interval,
    )
