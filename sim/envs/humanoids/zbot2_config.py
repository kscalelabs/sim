"""Defines the environment configuration for the Getting up task"""



from sim.env import robot_urdf_path
from sim.envs.base.legged_robot_config import (  # type: ignore
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from sim.resources.zbot2.joints import Robot

NUM_JOINTS = len(Robot.all_joints())


class ZBot2Cfg(LeggedRobotCfg):
    """Configuration class for the Legs humanoid robot."""

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        # num_single_obs = 11 + NUM_JOINTS * 3
        num_single_obs = 8 + NUM_JOINTS * 3 # pfb30
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 25 + NUM_JOINTS * 4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = NUM_JOINTS
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

        # from kinfer import proto as P
        # input_schema = P.IOSchema(
        #     values=[
        #         P.ValueSchema(
        #             value_name="vector_command",
        #             vector_command=P.VectorCommandSchema(
        #                 dimensions=3,  # x_vel, y_vel, rot
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="timestamp",
        #             timestamp=P.TimestampSchema(
        #                 start_seconds=0,
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="dof_pos",
        #             joint_positions=P.JointPositionsSchema(
        #                 joint_names=Robot.joint_names(),
        #                 unit=P.JointPositionUnit.RADIANS,
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="dof_vel",
        #             joint_velocities=P.JointVelocitiesSchema(
        #                 joint_names=Robot.joint_names(),
        #                 unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="prev_actions",
        #             joint_positions=P.JointPositionsSchema(
        #                 joint_names=Robot.joint_names(), unit=P.JointPositionUnit.RADIANS
        #             ),
        #         ),
        #         # Abusing the IMU schema to pass in euler and angular velocity instead of raw sensor data
        #         P.ValueSchema(
        #             value_name="imu_ang_vel",
        #             imu=P.ImuSchema(
        #                 use_accelerometer=False,
        #                 use_gyroscope=True,
        #                 use_magnetometer=False,
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="imu_euler_xyz",
        #             imu=P.ImuSchema(
        #                 use_accelerometer=True,
        #                 use_gyroscope=False,
        #                 use_magnetometer=False,
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="hist_obs",
        #             state_tensor=P.StateTensorSchema(
        #                 # 11 is the number of single observation features - 6 from IMU, 5 from command input
        #                 # 3 comes from the number of times num_actions is repeated in the observation (dof_pos, dof_vel, prev_actions)
        #                 shape=[frame_stack * (11 + NUM_JOINTS * 3)],
        #                 dtype=P.DType.FP32,
        #             ),
        #         ),
        #     ]
        # )

        # output_schema = P.IOSchema(
        #     values=[
        #         P.ValueSchema(
        #             value_name="actions",
        #             joint_positions=P.JointPositionsSchema(
        #                 joint_names=Robot.joint_names(), unit=P.JointPositionUnit.RADIANS
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="actions_raw",
        #             joint_positions=P.JointPositionsSchema(
        #                 joint_names=Robot.joint_names(), unit=P.JointPositionUnit.RADIANS
        #             ),
        #         ),
        #         P.ValueSchema(
        #             value_name="new_x",
        #             state_tensor=P.StateTensorSchema(
        #                 shape=[frame_stack * (11 + NUM_JOINTS * 3)],
        #                 dtype=P.DType.FP32,
        #             ),
        #         ),
        #     ]
        # )

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
        terminate_after_contacts_on = []

    class asset(LeggedRobotCfg.asset):
        name = "zbot2"
        file = str(robot_urdf_path(name))

        foot_name = ["FOOT", "FOOT_2"]
        knee_name = ["3215_BothFlange_6", "3215_BothFlange_5"]

        termination_height = 0.2
        default_feet_height = 0.01

        penalize_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

        # pfb30
        friction = 0.013
        armature = 0.008

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "plane"
        mesh_type = "trimesh"
        curriculum = True
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 9  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.3, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]
        restitution = 0.0

    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, Robot.height]
        rot = Robot.rotation

        default_joint_angles = {k: 0.0 for k in Robot.all_joints()}

        default_positions = Robot.default_walking()
        for joint in default_positions:
            default_joint_angles[joint] = default_positions[joint]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = Robot.stiffness()
        damping = Robot.damping()
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
        use_projected_gravity = False

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        start_pos_noise = 0.1
        skip_joints = ["L_Hip_Roll", "R_Hip_Roll", "L_Hip_Yaw", "R_Hip_Yaw"]
        randomize_friction = True
        friction_range = [0.1, 1.5]
        randomize_base_mass = True
        added_mass_range = [-0.1, 0.2]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.1
        max_push_ang_vel = 0.2
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02
        randomize_pd_gains = False

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/swdwdwd]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = Robot.height
        min_dist = 0.07
        max_dist = 0.4

        # put some settings here for LLM parameter tuning
        # pfb30
        target_joint_pos_scale = 0.28  # rad
        target_feet_height = 0.027  # m
        cycle_time = 0.25
        # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5.0
        max_contact_force = 400  # forces above this value are penalized

        class scales:
            joint_pos = 2.2
            feet_clearance = 1.6
            feet_contact_number = 1.4
            # gait
            feet_air_time = 1.5
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            # contact 
            feet_contact_forces = -0.02
            # vel tracking
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # stand_still = 5
            # base pos
            default_joint_pos = 0.8
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.003
            torques = -1e-10
            dof_vel = -1e-5
            dof_acc = -5e-9
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class viewer:
        ref_env = 0
        pos = [4, -4, 2]
        lookat = [0, -2, 0]


class ZBot2StandingCfg(ZBot2Cfg):
    """Standing configuration for the ZBot2 humanoid robot."""

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, Robot.standing_height]
        rot = Robot.rotation

        default_joint_angles = {k: 0.0 for k in Robot.all_joints()}

        default_positions = Robot.default_standing()
        for joint in default_positions:
            default_joint_angles[joint] = default_positions[joint]

    class rewards:
        base_height_target = Robot.height
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.17  # rad
        target_feet_height = 0.05  # m
        cycle_time = 0.5  # sec
        only_positive_rewards = False
        tracking_sigma = 5
        max_contact_force = 200

        class scales:
            default_joint_pos = 1.0
            orientation = 1
            base_height = 0.2
            base_acc = 0.2
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.0


class ZBot2CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "zbot2"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
