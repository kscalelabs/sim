"""Defines the environment configuration for the Getting up task"""

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from sim.env import stompy_urdf_path
from sim.humanoid_gym.envs.stompy_config import StompyCfg
from sim.stompy.joints import StompyFixed

NUM_JOINTS = len(StompyFixed.default_standing())  # 17


class LegsCfg(StompyCfg):
    """
    Configuration class for the Legs humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim

        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 11 + NUM_JOINTS * 3
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 25 + NUM_JOINTS * 4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = NUM_JOINTS
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = str(stompy_urdf_path(legs_only=True))

        name = "stompy"

        foot_name = "_leg_1_x4_1_outer_1"  # "foot"
        knee_name = "belt_knee"

        termination_height = 0.23
        default_feet_height = 0.0

        terminate_after_contacts_on = ["link_torso_1_top_torso_1"]

        penalize_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False


class LegsCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

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
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "Legs"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
