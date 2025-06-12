from legged_gym.utils.task_registry import task_registry
from gennbv.env.env_train_gennbv import Env_Train_GenNBV
from gennbv.env.env_eval_gennbv import Env_Eval_GenNBV
from gennbv.env.config_gennbv_train import Config_GenNBV_Train, DroneCfgPPO
from gennbv.env.config_gennbv_eval import Config_GenNBV_Eval
task_registry.register("train_gennbv", Env_Train_GenNBV, Config_GenNBV_Train, DroneCfgPPO)
task_registry.register("eval_gennbv", Env_Eval_GenNBV, Config_GenNBV_Eval, DroneCfgPPO)
