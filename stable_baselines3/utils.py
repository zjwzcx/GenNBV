import datetime
# from locotransformer.env_wrapper.norm_image_state_wrapper import NormObsWithImg
# from locotransformer.env_wrapper.dict_to_array_wrapper import ObservationDictionaryToArrayWrapper
# from locotransformer.env_wrapper.dict_to_array_wrapper_action_only import ObservationDictionaryToArrayWrapper_Action_Only
# from locotransformer.env_wrapper.dict_to_array_wrapper_pose import ObservationDictionaryToArrayWrapper_Pose
# from locotransformer.env_wrapper.dict_to_array_wrapper_pose_eval import ObservationDictionaryToArrayWrapper_Pose_Eval

# from active_reconstruction.wrapper.env_wrapper_pose_traj import ReconstructionWrapper_Pose_Traj
# from active_reconstruction.wrapper.env_wrapper_pose_traj_eval import ReconstructionWrapper_Pose_Traj_Eval
# from active_reconstruction.wrapper.env_wrapper_pc_mink_traj import ReconstructionWrapper_PC_Mink_Traj
# from active_reconstruction.wrapper.env_wrapper_pc_mink_traj_eval import ReconstructionWrapper_PC_Mink_Traj_Eval
# from active_reconstruction.wrapper.env_wrapper_map_traj import ReconstructionWrapper_Map_Pose_Traj
# from active_reconstruction.wrapper.env_wrapper_map_traj_depth import ReconstructionWrapper_Map_Pose_Traj_Depth
# from active_reconstruction.wrapper.env_wrapper_pose_ego_pc_2d import ReconstructionWrapper_Pose_Ego_PC_2D
# from active_reconstruction.wrapper.env_wrapper_pose_ego_map_2d import ReconstructionWrapper_Pose_Ego_Map_2D
# from active_reconstruction.wrapper.env_wrapper_pose_ego_map_world_map_2d import ReconstructionWrapper_Pose_Ego_Map_World_Map_2D
# from active_reconstruction.wrapper.env_wrapper_pose_ego_grid_3d import ReconstructionWrapper_Pose_Ego_Grid_3D
# from active_reconstruction.wrapper.env_wrapper_state_only import ReconstructionWrapper_State

from active_reconstruction.wrapper.env_wrapper_grid_rgb_pose import ReconstructionWrapper_Grid_RGB_Pose
from active_reconstruction.wrapper.env_wrapper_grid_rgb_pose_eval import ReconstructionWrapper_Grid_RGB_Pose_Eval

# from locotransformer.env_wrapper.diagonal_action import DiagonalAction
from legged_gym.env.base.base_task import BaseTask
from legged_gym import OPEN_ROBOT_ROOT_DIR
import os

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_api_key_file(wandb_key_file=None):
    wandb_key_file = wandb_key_file or "wandb_api_key_file.txt"
    root = OPEN_ROBOT_ROOT_DIR
    path = os.path.join(root, "wandb_utils", wandb_key_file)
    print("We are using this wandb key file: ", path)
    return path


def get_time_str():
    return datetime.datetime.now().strftime("%m%d_%H%M_%S")


def is_isaac_gym_env(env):
    Isacc_Gym_Env = [
        BaseTask, 
        ReconstructionWrapper_Grid_RGB_Pose,
        ReconstructionWrapper_Grid_RGB_Pose_Eval,
    ]

    for target_env in Isacc_Gym_Env:
        if isinstance(env, target_env):
            return True
    return False

