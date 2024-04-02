"""Defines the environment configuration for the Stompy humanoid robot."""

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from sim.env import stompy_urdf_path
from sim.stompy.joints import Stompy

NUM_JOINTS = len(Stompy.all_joints())  # 37


class StompyCfg(LeggedRobotCfg):
    """Configuration class for the Stompy humanoid robot."""

    class env(LeggedRobotCfg.env):  # noqa: N801
        # Change the observation dim
        frame_stack = 15  # Number of frames in an observation
        c_frame_stack = 3  # Number of frames in a critic observation
        num_single_obs = 121  # Size of a single observation (for the actor policy)
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 137  # Size of a single privileged observation (for the critic)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = NUM_JOINTS  # Torque command for each joint.
        num_envs = 4096
        episode_length_s = 8  # Maximum episode length
        use_ref_actions = False

    class safety:  # noqa: N801
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):  # noqa: N801
        file = str(stompy_urdf_path())

        name = "stompy"
        foot_name = "ankle"  # Rubber / Foot get merged into ankle
        knee_name = "belt_knee"  # Something like ..._belt_knee_left...""

        # Terminate the episode upon contacting a delicate part.
        terminate_after_contacts_on = ["head", "gripper"]

        # Penalize non-foot contacts.
        penalize_contacts_on = ["torso"]

        self_collisions = 0  # 1 to disable, 0 to enable
        collapse_fixed_joints = True
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):  # noqa: N801
        mesh_type = "plane"  # plane; trimesh
        curriculum = False
        # For rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # Number of terrain rows (levels)
        num_cols = 20  # Number of terrain cols (types)
        max_init_terrain_level = 10  # Starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 1.0

    class noise:  # noqa: N801
        add_noise = True
        noise_level = 0.1  # Scales other values

        class noise_scales:  # noqa: N801
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):  # noqa: N801
        pos = [0.0, 0.0, 0.95]
        default_joint_angles = {k: 0.0 for k in Stompy.all_joints()}

    class sim(LeggedRobotCfg.sim):  # noqa: N801
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):  # noqa: N801
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = -0.01
            rest_offset = -0.015
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2  # gymapi.CC_ALL_SUBSTEPS

    class domain_rand:  # noqa: N801
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):  # noqa: N801
        num_commands = 8  # Command space is the joint angles for the shoulder and elbow.
        resampling_time = 8.0

    class rewards:  # noqa: N801
        # base_height_target = 1.1
        min_dist = 0.2
        max_dist = 0.5

        # Put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17  # rad
        target_feet_height = 0.06  # m
        cycle_time = 0.64  # sec

        # If true, negative total rewards are clipped at zero.
        only_positive_rewards = False

        # Max contact force should be a bit above the weight of the robot. In
        # our case the robot weighs 62 Kg, so we set it to 700.
        # max_contact_force = 700

        class scales:  # noqa: N801
            # Reference motion tracking
            # joint_pos = 1.6
            # feet_clearance = 1.0
            # feet_contact_number = 1.2

            # Gait
            # feet_air_time = 1.0
            # foot_slip = -0.05
            # feet_distance = 0.2
            # knee_distance = 0.2

            # Contact
            # feet_contact_forces = -0.01

            # Velocity tracking
            # tracking_lin_vel = 1.2
            # tracking_ang_vel = 1.1
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5

            # Base position
            base_height = 1.0
            # base_acc = 0.1
            base_acc = 0.0

            # Energy
            # torques = -1e-6
            # dof_vel = -5e-5
            # dof_acc = -1e-8
            # collision = -1e-2
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            collision = 0.0

    class normalization:  # noqa: N801
        class obs_scales:  # noqa: N801
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0


class StompyPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"

    class policy:  # noqa: N801
        init_noise_std = 1.0
        actor_hidden_dims = [512, 512, 256]
        critic_hidden_dims = [1024, 512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):  # noqa: N801
        entropy_coef = 0.01
        learning_rate = 3e-4
        num_learning_epochs = 2
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        max_grad_norm = 0.5

    class runner:  # noqa: N801
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60
        max_iterations = 3001

        # logging
        save_interval = 100
        experiment_name = "stompy_ppo"
        run_name = ""

        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # Updated from load_run and checkpoint.
