from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np
import gym
import torch


def update_mean_var_count(mean, var, count, batch_mean, batch_var, batch_count):
    """
    Imported From OpenAI Baseline
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class BestCKPTCallback(CheckpointCallback):
    def __init__(
        self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, key_list: list = None
    ):
        key_list = key_list or []
        self.rollout_count = 0
        super(BestCKPTCallback, self).__init__(save_freq, save_path, name_prefix, verbose)
        self.key_highest_value = {k: 0. for k in key_list}

    @property
    def best_save_path(self):
        return self.locals["self"].logger.dir

    @property
    def device(self):
        return self.locals["self"].device

    def _on_rollout_end(self) -> None:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        for key in self.key_highest_value.keys():
            value = self.calculate_value(key)
            if value > self.key_highest_value[key]:
                self.key_highest_value[key] = value
                path = os.path.join(self.best_save_path, "{}_best_{}".format(self.name_prefix, key))
                self.model.save(path)
                # if self.verbose > 1:
                print("Saving Best {} checkpoint to {}: {}".format(key, path, value))

    def calculate_value(self, key):
        ep_info_buffer = self.locals["self"].ep_info_buffer
        infotensor = torch.tensor([], device=self.device)
        assert key in ep_info_buffer[0], "no key named {}, can not save checkpoint".format(key)
        for ep_info in ep_info_buffer:
            # handle scalar and zero dimensional tensor infos
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor).detach().cpu().numpy()
        return value


class ReconstructionCallBack(BestCKPTCallback):
    pass


class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self._wrapped_env = env
        self.training = True

    def train(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.train()
        self.training = True

    def eval(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.eval()
        self.training = False

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def copy_state(self, source_env):
        pass



class Normalizer():
    def __init__(self, shape, clip=10., device="cuda"):
        self.shape = shape  # [buffer_size, action_size]
        self._mean = torch.zeros(shape, device=device)
        self._var = torch.ones(shape, device=device)
        self._count = 1e-4
        self.clip = clip
        self.should_estimate = True

    def stop_update_estimate(self):
        self.should_estimate = False

    def update_estimate(self, data):
        """
        data: [num_env, state_size]
        """
        if not self.should_estimate:
            return
        if len(data.shape) == self.shape:
            data = torch.unsqueeze(data, 0)
        self._mean, self._var, self._count = update_mean_var_count(
            self._mean, self._var, self._count, torch.mean(data, dim=0), torch.var(data, dim=0), data.shape[0]
        )

    def inverse(self, raw):
        return raw * torch.sqrt(self._var) + self._mean

    def inverse_torch(self, raw):
        return raw * torch.Tensor(torch.sqrt(self._var)).to(raw.device) \
               + torch.Tensor(self._mean).to(raw.device)

    def filt(self, raw):
        return torch.clamp((raw - self._mean) / (torch.sqrt(self._var) + 1e-4), -self.clip, self.clip)

    def filt_torch(self, raw):
        return torch.clamp(
            (raw - torch.Tensor(self._mean).to(raw.device)) /
            (torch.Tensor(torch.sqrt(self._var) + 1e-4).to(raw.device)), -self.clip, self.clip
        )


class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__(self, env, train, epsilon=1e-4, clipob=10.):
        super(NormObsWithImg, self).__init__(env)
        self.count = epsilon
        self.clipob = clipob
        self.is_training = train
        state_shape = env._gym_env.observation_space.spaces["state"].shape  # [buffer_size * action_size]
        self.obs_normalizer = Normalizer(state_shape, device=env.device)
        self.state_shape = np.prod(state_shape)

    def observation(self, observation):
        if self.is_training:
            self.obs_normalizer.update_estimate(observation[..., :self.state_shape])
        img_obs = observation[..., self.state_shape:]
        return torch.concat([self.obs_normalizer.filt(observation[..., :self.state_shape]), img_obs], dim=-1)
