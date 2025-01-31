from active_reconstruction.env_gennbv.env_config_house_discrete import ReconstructionDroneConfig_House_Discrete
import numpy as np


class ReconstructionDroneConfig_House_Discrete_Large(ReconstructionDroneConfig_House_Discrete):
    position_use_polar_coordinates = False  # the position will be represented by (r, \theta, \phi) instead of (x, y, z)
    direction_use_vector = False  # (r, p, y)
    debug_save_image_tensor = False
    debug_save_path = None
    max_episode_length = 100    # max_steps_per_episode

    # init_action_buf = [0., 0., 6.1, 0., 90 / 180 * np.pi, 0.]  # 1433 only, it specifies the init position/heading of the drone
    # init_action_buf = [0., 0., 8.1, 0., 90 / 180 * np.pi, 0.]  # 20230915
    init_action_buf = [0., 0., 10.1, 0., 90 / 180 * np.pi, 0.]  # 1433 extend v1


    # init_action = [40, 35, 30, 0, 12, 0]    # 1433
    # init_action = [40, 35, 50, 0, 12, 0]    # 1433 extend
    # init_action = [50, 50, 50, 0, 12, 0]    # 20230915
    # init_action = [40, 40, 30, 0, 12, 0]    # 1433 extend v1 (but init height=6.1)
    # init_action = [40, 40, 40, 0, 12, 0]    # 1433 extend v1 (but init height=8.1)
    init_action = [40, 40, 50, 0, 12, 0]    # 1433 extend v1


    action_unit = [0.2, 0.2, 0.2, 0., 1/12*np.pi, 1/6*np.pi]  # 1433 only, position range

    class visual_input(ReconstructionDroneConfig_House_Discrete.visual_input):

        camera_width = 400
        camera_height = 400
        horizontal_fov = 90.0   # Horizontal field of view in degrees.

    class rewards:
        class scales:   # * self.dt (0.019999) / episode_length_s (20). For example, when reward_scales=1000, rew_xxx = reward * 1000 * 0.019999 / 20 = reward
            surface_coverage = 1000     # original scale (coverage ratio: [0, 1])
            short_path = 5

            # mean_AUC = 1

            termination = 50   # Terminal reward / penalty
            # collision = -1000     # Penalize collisions on selected bodies

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 100.    # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        pi = 3.14159265359
        clip_observations = 255.


        # clip_actions_low_world = [-8., -7., 0.1, 0., -1/2*pi, 0.]   # 1433
        # clip_actions_low_world = [-10., -10., 0.1, 0., -1/2*pi, 0.]   # 20230915
        clip_actions_low_world = [-8., -8., 0.1, 0., -1/2*pi, 0.]   # 1433 extend v1

        # clip_actions_up = [80, 70, 59, 0, 12, 12]       # 1433
        # clip_actions_up = [80, 70, 99, 0, 12, 12]       # 1433 extend
        # clip_actions_up = [100, 100, 99, 0, 12, 12]       # 20230915
        clip_actions_up = [80, 80, 50, 0, 12, 12]       # 1433 extend v1

        clip_actions_low = [0, 0, 0, 0, 0, 0]