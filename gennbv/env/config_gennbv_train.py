from gennbv.env.config_legged import LeggedVisualInputConfig, LeggedVIsualInputCfgPPO
import numpy as np
from isaacgym import gymapi


class Config_GenNBV_Train(LeggedVisualInputConfig):
    position_use_polar_coordinates = False  # the position will be represented by (r, \theta, \phi) instead of (x, y, z)
    direction_use_vector = False  # (r, p, y)
    debug_save_image_tensor = False
    debug_save_path = None
    max_episode_length = 100

    class rewards:
        class scales:   # * self.dt (0.019999) / episode_length_s (20). For example, when reward_scales=1000, rew_xxx = reward * 1000 * 0.019999 / 20 = reward
            surface_coverage = 1000     # original scale (coverage ratio: [0, 1])
            short_path = 5
            termination = 50   # Terminal reward / penalty
            # collision = -10     # Penalize collisions on selected bodies

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 100.    # forces above this value are penalized

    class visual_input(LeggedVisualInputConfig.visual_input):
        camera_width = 400
        camera_height = 400
        horizontal_fov = 90.0   # Horizontal field of view in degrees. Vertical field of view is calculated from height to width ratio

        stack = 100
        supersampling_horizontal = 1
        supersampling_vertical = 1
        normalization = True    # normalize pixels to [0, 1]
        far_plane = 2000000.0   # distance in world coordinates to far-clipping plane
        near_plane = 0.0010000000474974513  # distance in world coordinate units to near-clipping plane

        type = gymapi.IMAGE_DEPTH

    class asset(LeggedVisualInputConfig.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/drone/cf2x.urdf'
        name = "cf2x"
        prop_name = "prop"
        penalize_contacts_on = ["prop", "base"]
        terminate_after_contacts_on = ["prop", "base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class env(LeggedVisualInputConfig.env):
        num_observations = 6
        episode_length_s = 20  # in second
        num_actions = 6
        env_spacing = 5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        pi = 3.14159265359
        clip_observations = 255.

        clip_pose_low = [-8., -8., 0.1, 0., -1/2*pi, 0.]
        clip_pose_idx_up = [80, 80, 50, 0, 12, 12]
        clip_pose_idx_low = [0, 0, 0, 0, 0, 0]
        
        # NOTE: dense discrete action space
        init_pose_buf = [0., 0., 10.1, 0., 90 / 180 * np.pi, 0.]
        init_action = [40, 40, 50, 0, 12, 0]
        action_unit = [0.2, 0.2, 0.2, 0., 1/12*np.pi, 1/6*np.pi]

    class termination:
        collision = True
        max_step_done = True


class DroneCfgPPO(LeggedVIsualInputCfgPPO):
    """
    NOTE: this config is only a placeholder for using task register and thus unless, since we use sb3  instead of RSL_RL
    """
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy(LeggedVIsualInputCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedVIsualInputCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedVIsualInputCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class visual_input(LeggedVisualInputConfig.visual_input):
        camera_width = 320
        camera_height = 240
        type = gymapi.IMAGE_COLOR
        # stack = 5  # consecutive frames to stack
        normalization = True
        cam_pos = (0.0, 0, 0.0)
