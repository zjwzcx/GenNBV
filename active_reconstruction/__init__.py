from legged_gym.utils.task_registry import task_registry
from active_reconstruction.env.env_config import DroneCfgPPO


# GenNBV
from active_reconstruction.env.recon_houses3k_gennbv import Recon_Houses3K_GenNBV
from active_reconstruction.env.env_config_houses3k import ReconDroneConfig_Houses3K
task_registry.register("recon_houses3k_gennbv", Recon_Houses3K_GenNBV, ReconDroneConfig_Houses3K, DroneCfgPPO)


from active_reconstruction.env.recon_houses3k_gennbv_eval import Recon_Houses3K_GenNBV_Eval
from active_reconstruction.env.env_config_houses3k_eval import ReconDroneConfig_Houses3K_Eval
task_registry.register("recon_houses3k_gennbv_eval", Recon_Houses3K_GenNBV_Eval, ReconDroneConfig_Houses3K_Eval, DroneCfgPPO)
