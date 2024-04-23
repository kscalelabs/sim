"""Defines the environment configuration for getting up task.

Command to run:

    python sim/humanoid_gym/play.py --task getup_ppo --load_run Apr08_00-02-41_ --run_name v1
"""

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from sim.env import stompy_urdf_path
from sim.stompy.joints import Stompy

NUM_JOINTS = len(Stompy.all_joints())  # 37


class GetupCfg(LeggedRobotCfg):
    """Configuration class for the Legs humanoid robot."""

    class env(LeggedRobotCfg.env):  # noqa: N801
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 11 + NUM_JOINTS * 3
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 25 + NUM_JOINTS * 4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = NUM_JOINTS
        num_envs = 4096
        episode_length_s = 10  # episode length in seconds
        use_ref_actions = False

    class safety:  # noqa: N801
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):  # noqa: N801
        file = str(stompy_urdf_path())

        name = "stompy"
        foot_name = "_leg_1_x4_1_outer_1"  # "foot"
        knee_name = "belt_knee"  # "knee"
        terminate_after_contacts_on: list[str] = []  # "link_torso_1_top_torso_1"]
        penalize_contacts_on: list[str] = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):  # noqa: N801
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.0

    class noise:  # noqa: N801
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:  # noqa: N801
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):  # noqa: N801
        pos = [0.0, 0.0, 0.2]

        # Face up
        # rot = [-0.717107, 0.0, 0.0, 0.717107]
        # Face down
        rot = [-0.717107, 0.0, 0.0, -0.717107]

        default_joint_angles = {k: 0.0 for k in Stompy.all_joints()}

        default_positions = Stompy.default_standing()
        for joint in default_positions:
            default_joint_angles[joint] = default_positions[joint]

    class control(LeggedRobotCfg.control):  # noqa: N801
        # PD Drive parameters:
        stiffness = {
            "x10": 200,
            "x8": 200,
            "x6": 200,
            "x4": 200,
            "foot": 200,
            "forward": 200,
            "knee": 200,
            "ankle": 200,
        }
        damping = {
            "x10": 10,
            "x8": 10,
            "x6": 10,
            "x4": 10,
            "foot": 10,
            "forward": 10,
            "knee": 10,
            "ankle": 10,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):  # noqa: N801
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):  # noqa: N801
            num_threads = 12
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = -0.02  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:  # noqa: N801
        randomize_friction = True
        friction_range = [0.1, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):  # noqa: N801
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:  # noqa: N801
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:  # noqa: N801
        # quite important to keep it right
        base_height_target = 0.97
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17  # rad
        target_feet_height = 0.06  # m
        cycle_time = 0.64  # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # forces above this value are penalized

        class scales:  # noqa: N801
            # force the model to learn specific position?
            # joint_pos = 2.
            # height reward
            base_height = 1
            # base_acc = 0.2
            # energy
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7
            # collision = -1.0

    class normalization:  # noqa: N801
        class obs_scales:  # noqa: N801
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class viewer:  # noqa: N801
        ref_env = 0
        pos = [4, -4, 2]
        lookat = [0, -2, 0]


class GetupCfgPPO(LeggedRobotCfgPPO):
    """inherited from LeggedRobotCfgPPO."""

    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:  # noqa: N801
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):  # noqa: N801
        entropy_coef = 0.01
        learning_rate = 1e-5

    class runner:  # noqa: N801
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "Getup"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
