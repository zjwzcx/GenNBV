from gennbv.env.config_gennbv_train import Config_GenNBV_Train
import numpy as np
from isaacgym import gymapi


class Config_GenNBV_Eval(Config_GenNBV_Train):
    max_episode_length = 30    # max_steps_per_episode during evaluation

    class rewards:
        class scales:   # * self.dt (0.019999) / episode_length_s (20). For example, when reward_scales=1000, rew_xxx = reward * 1000 * 0.019999 / 20 = reward
            surface_coverage = 50   # just easy to evaluate

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        max_contact_force = 100.    # forces above this value are penalized
