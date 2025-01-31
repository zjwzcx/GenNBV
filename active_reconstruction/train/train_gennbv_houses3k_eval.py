"""
This script is to check the sb3 compatibility. The training config is the same as the one provided by official RSL-rl
"""
import os
import gym
from typing import Callable

from isaacgym import gymapi
from active_reconstruction.callback import ReconstructionCallBack
import active_reconstruction
from legged_gym import OPEN_ROBOT_ROOT_DIR
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args, set_seed

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy_Train_Eval
from stable_baselines3.ppo.ppo_gennbv import PPO_GenNBV
from stable_baselines3.utils import get_time_str
from wandb_utils import team_name, project_name
from wandb_utils.wandb_callback import WandbCallback

# from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from active_reconstruction.wrapper.env_wrapper_grid_rgb_pose import ReconstructionWrapper_Grid_RGB_Pose
# from active_reconstruction.wrapper.env_wrapper_grid_obs_with_action_discrete_rgb_grayscale_state_eval import ReconstructionWrapper_Grid_Obs_With_Action_Discrete_RGB_Grayscale_State_Eval


from active_reconstruction.hybrid_encoder import Hybrid_Encoder


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_env_register(env_id: str, rank: int, seed: int = 1, args=None, env_cfg=None) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env_cfg.seed = seed
        env, env_config = task_registry.make_env(env_id, args, env_cfg)
        set_seed(seed)
        # return env, env_config
        return env
    set_random_seed(seed)
    return _init


def main():
    additional_args = [
        {
            "name": "--buffer_size",
            "type": int,
            "default": 100,
            "help": "length of buffer"
        },
        {
            "name": "--n_steps",
            "type": int,
            "default": 128,
            "help": "number of steps to collect in each env"
        },
        {
            "name": "--batch_size",
            "type": int,
            "default": 128,
            "help": "SGD batch size"
        },
        {
            "name": "--save_freq",
            "type": int,
            "default": 10000,
            "help": "save the model per <save_freq> iter"
        },
        {
            "name": "--total_iters",
            "type": int,
            "default": 1000,
            "help": "the number of training iters"
        },
        {
            "name": "--n_epochs",
            "type": int,
            "default": 5
        },
        {
            "name": "--use_target_kl",
            "type": bool,
            "default": True
        },
        {
            "name": "--target_kl",
            "type": float,
            "default": 0.05
        },
        {
            "name": "--vf_coeff",
            "type": float,
            "default": 0.8
        },
        {
            "name": "--ent_coeff",
            "type": float,
            "default": 0.01
        },
        {
            "name": "--lr",
            "type": float,
            "default": 1e-4
        },
        {
            "name": "--unflatten_terrain",
            "type": bool,
            "default": False},
        {
            "name": "--first_view_camera",
            "type": bool,
            "default": False},
        {
            "name": "--eval_device",
            "type": str,
            "default": "cuda:0",
        },
    ]
    reward_args = [
        {
            "name": "--surface_coverage",
            "type": float,
            "default": 1.0,
            "help": "surface coverage ratio"
        },
        {
            "name": "--only_positive_rewards",
            "type": bool,
            "default": False,
            "help": "If true negative total rewards are clipped at zero (avoids early termination problems)"
        },
        {
            "name": "--max_contact_force",
            "type": float,
            "default": 100,
            "help": "Forces above this value are penalized"
        },
    ]

    args = get_args(additional_args+reward_args)
    args.task = "recon_houses3k_gennbv"
    # eval_task = "reconstruction_drone_house_soft_grid_tri_cls_with_kimg_eval_batch_12_AUC_Acc" 
    if args.num_envs is None:
        args.num_envs = 256
    eval_freq = int(500000/args.num_envs)

    use_wandb = not args.stop_wandb
    args.headless = True
    # args.headless = False  # False: visualization

    # args.num_envs = 2
    use_wandb = False   # debug
    # eval_freq = 200


    # NOTE: just for visualization!!!
    ckpt_path = None

    exp_name = args.task
    seed = int(args.seed)
    trial_name = f"{exp_name}_{get_time_str()}" \
        if args.exp_name is None or len(args.exp_name) == 0 \
        else f"{args.exp_name}_{get_time_str()}"
    log_dir = os.path.join(OPEN_ROBOT_ROOT_DIR, "runs", trial_name)
    print("[LOGGING] We start logging training data into {}".format(log_dir))

    # ===== Setup the training environment =====
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # env_cfg_eval, _ = task_registry.get_cfgs(name=eval_task)

    env_cfg.visual_input.stack = args.buffer_size
    # env_cfg_eval.visual_input.stack = args.buffer_size

    # NOTE: train env
    env_train, env_cfg_train = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_cfg_dict = {key:value for key, value in env_cfg_train.__dict__.items()}

    # # NOTE: eval env
    # args_eval = copy.deepcopy(args)
    # args_eval.num_envs = 50
    # env_eval = SubprocVecEnv([
    #         make_env_register(env_id=eval_task, rank=1, seed=100, args=args_eval, env_cfg=env_cfg_eval),
    #     ]
    # )

    env = ReconstructionWrapper_Grid_Obs_With_Action_Discrete_RGB_Grayscale_State(env_train)
    # env_eval = ReconstructionWrapper_Grid_Obs_With_Action_Discrete_RGB_Grayscale_State_Eval(env_eval)

    # ===== Setup the config =====
    config = dict(
        algo=dict(
            policy=ActorCriticPolicy_Train_Eval,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=Hybrid_Encoder,
                features_extractor_kwargs=dict(
                    encoder_param={
                        "hidden_shapes": [256, 256],
                        "visual_dim": 256
                    },
                    net_param={
                        "transformer_params": [[1, 256], [1, 256]],
                        "append_hidden_shapes": [256, 256]
                    },
                    state_input_shape=(args.buffer_size * 6,),  # buffer_size * action_size (single action per iteration)
                    visual_input_shape=(
                    args.buffer_size, env_cfg.visual_input.camera_height, env_cfg.visual_input.camera_width)
                )
            ),
            env=env,
            learning_rate=args.lr,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=args.target_kl if args.use_target_kl else None,
            max_grad_norm=1,
            n_steps=args.n_steps,  # steps to collect in each env
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=0.2,
            vf_coef=args.vf_coeff,
            clip_range_vf=0.2,
            ent_coef=args.ent_coeff,
            tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device=args.sim_device,
        ),

        # Meta data
        gpu_simulation=True,
        project_name=project_name,
        team_name=team_name,
        exp_name=exp_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=log_dir
    )

    # ===== Setup the callbacks =====
    callbacks = [
        ReconstructionCallBack(
            name_prefix="rl_model",
            verbose=1,
            save_freq=args.save_freq,
            save_path=os.path.join(log_dir, "models"),
            key_list=["episode_reward"]
        )
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(trial_name=trial_name, exp_name=exp_name, project_name=project_name, config={**config, **env_cfg_dict})
        )
    callbacks = CallbackList(callbacks)

    # ===== Launch training =====
    model = PPO_GenNBV(**config["algo"])
    if ckpt_path:
        model.set_parameters(ckpt_path)

    try:
        model.learn(
            # training
            total_timesteps=args.num_envs * args.n_steps * args.total_iters,    # num_steps_per_iter: args.num_envs * args.n_steps
            callback=callbacks,
            reset_num_timesteps=True,

            # eval
            eval_env=None,
            n_eval_episodes=1,
            # eval_env=env_eval,
            # n_eval_episodes=50,
            eval_freq=eval_freq,
            eval_log_path=None,

            # logging
            tb_log_name=exp_name,  # Should place the algorithm name here!
            log_interval=1,
        )
    finally:
        env.close()


if __name__ == '__main__':
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t_start = time.time()

    main()

    t_end = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Total wall-clock time: {:.3f}min".format((t_end-t_start)/60))
