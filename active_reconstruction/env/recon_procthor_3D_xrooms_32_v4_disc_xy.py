from isaacgym import gymapi, gymtorch
import os
import torch
import random
import torch.nn.functional as F
from collections import deque
from gym.spaces import Box, Dict, MultiDiscrete
from active_reconstruction.utils_cuda import scanned_pts_to_idx_3D, pose_coord_to_idx_3D, \
                                                grid_occupancy_tri_cls, Bresenham3D_PyCUDA
from active_reconstruction.env.recon_procthor_2D_xrooms_32_base import ReconDroneEnv_ProcTHOR_2D_XRooms_32_Base
# from active_reconstruction.env.recon_procthor_2D_xrooms_32_v2 import ReconDroneEnv_ProcTHOR_2D_XRooms_32_V2
from isaacgym.torch_utils import *
from legged_gym import OPEN_ROBOT_ROOT_DIR
import bfs_cuda_3D


class ReconDroneEnv_ProcTHOR_3D_XRooms_32_Disc_XY(ReconDroneEnv_ProcTHOR_2D_XRooms_32_Base):
    def __init__(self, *args, **kwargs):

        self.visualize_flag = False
        # self.visualize_flag = True

        self.num_scene = 32

        super(ReconDroneEnv_ProcTHOR_2D_XRooms_32_Base, self).__init__(*args, **kwargs)

    def _additional_create(self, env_handle, env_index):
        assert self.cfg.return_visual_observation, "visual observation should be returned!"
        scene_idx = env_index % self.num_scene

        # NOTE: urdf load, create actor
        dataset_name = "procthor_4-room_32_obj_center"
        urdf_path = os.path.join(OPEN_ROBOT_ROOT_DIR, 
                                 f"active_reconstruction/objects/{dataset_name}/urdf")
        urdf_name = f"scene_{scene_idx}.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # default: True
        asset_options.disable_gravity = True

        asset = self.gym.load_asset(self.sim, urdf_path, urdf_name, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.env_origins[env_index][0],
                                self.env_origins[env_index][1],
                                self.env_origins[env_index][2])
        pose.r = gymapi.Quat(-np.pi/2, 0.0, 0.0, np.pi/2)

        ahandle = self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)

        self.additional_actors[env_index] = [ahandle]

    def _init_load_all(self):
        self.grid_size = 128
        dataset_name = "procthor_4-room_32"

        # [num_scene, 3]
        self.voxel_size_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                    f"active_reconstruction/gt/gt_{dataset_name}/{dataset_name}_{self.grid_size}_voxel_size_gt.pt"), map_location=self.device)

        # [num_scene, 6], (x_max, x_min, y_max, y_min, z_max, z_min)
        self.range_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                f"active_reconstruction/gt/gt_{dataset_name}/{dataset_name}_{self.grid_size}_range_gt.pt"), map_location=self.device)

        # [num_scene, grid_size, grid_size, grid_size]
        self.grids_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                f"active_reconstruction/gt/gt_{dataset_name}/{dataset_name}_{self.grid_size}_grid_gt.pt"), map_location=self.device)

        # [num_scene]
        self.num_valid_voxel_gt = self.grids_gt.sum(dim=(1, 2, 3))

        # [num_scene, 128, 128]. TODO: at initial height = 2m
        self.motion_height = 2.0
        # height_ratio = 5/10
        # self.motion_height = self.range_gt[:, 5] + (self.range_gt[:, 4] - self.range_gt[:, 5]) * height_ratio
        # self.motion_height_idx = 63

        # list of [num_point, 3]
        init_maps = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                f"active_reconstruction/gt/gt_{dataset_name}/{dataset_name}_{self.grid_size}_init_map.pt"), map_location=self.device)
        self.init_maps_list = [(torch.nonzero(init_maps[idx]) / (self.grid_size - 1) * 2 - 1) * self.range_gt[idx, :4:2]
                               for idx in range(self.num_scene)]

        # num_scene -> num_env
        self.env_to_scene = []
        for idx in range(self.num_envs):
            self.env_to_scene.append(idx % self.num_scene)
        self.env_to_scene = torch.tensor(self.env_to_scene, device=self.device)     # [num_env]

        self.range_gt_scenes = self.range_gt[self.env_to_scene]
        self.voxel_size_gt_scenes = self.voxel_size_gt[self.env_to_scene]
        self.num_valid_voxel_gt_scenes = self.num_valid_voxel_gt[self.env_to_scene]
        self.grids_gt_scenes = self.grids_gt[self.env_to_scene]

        # del self.range_gt     # used later
        del self.voxel_size_gt
        del self.num_valid_voxel_gt
        del self.grids_gt

        print("Loaded all ground truth data.")

    def _init_buffers(self):
        # load ground truth
        self._init_load_all()

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.skip = int(actor_root_state.shape[0] / self.num_envs)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[::self.skip, 0:3]
        self.base_quat = self.root_states[::self.skip, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # [num_envs, num_bodies, xyz axis]
        self.rigid_state = gymtorch.wrap_tensor(rigid_state).view(self.num_envs, -1, 13)  # [num_envs, num_bodies, 13]

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.buffer_size = self.cfg.visual_input.stack
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # NOTE: pose setup
        rpy_min = torch.tensor([0., 0., 0.], dtype=torch.float32, device=self.device).repeat(self.num_scene, 1)  # [num_scene, 3]
        rpy_max = torch.tensor([0., 0., 2 * torch.pi], dtype=torch.float32, device=self.device).repeat(self.num_scene, 1)

        if self.num_scene >= self.range_gt.shape[0]:
            self.clip_pose_world_low = torch.cat([self.range_gt.clone()[:, 1::2], rpy_min], dim=1).to(self.device)   # [num_scene, 6], (x_min, y_min, z_min, roll_min, pitch_min, yaw_min)
            self.clip_pose_world_up = torch.cat([self.range_gt.clone()[:, ::2], rpy_max], dim=1).to(self.device)     # [num_scene, 6], (x_max, y_max, z_max, roll_max, pitch_max, yaw_max)
        else:
            self.clip_pose_world_low = torch.cat([self.range_gt.clone()[:self.num_scene, 1::2], rpy_min], dim=1).to(self.device)   # [num_scene, 6], (x_min, y_min, z_min, roll_min, pitch_min, yaw_min)
            self.clip_pose_world_up = torch.cat([self.range_gt.clone()[:self.num_scene, ::2], rpy_max], dim=1).to(self.device)     # [num_scene, 6], (x_max, y_max, z_max, roll_max, pitch_max, yaw_max)

        if self.visualize_flag:
            self.pose_buf_vis = deque(maxlen=self.buffer_size)  # for visualization

        # NOTE: action space
        self.actions = torch.tensor(self.cfg.normalization.init_action, dtype=torch.int64, device=self.device).repeat(self.num_envs, 1)
        self.action_size = self.actions.shape[1]
        self.clip_actions_low = torch.tensor(self.cfg.normalization.clip_actions_low, dtype=torch.int64, device=self.device)
        self.clip_actions_up = torch.tensor(self.cfg.normalization.clip_actions_up, dtype=torch.int64, device=self.device)

        # NOTE: reward functions
        assert self.buffer_size >= 2, "buffer size should be larger than 2"
        self.recent_num = 5     # termination condition
        self.ratio_threshold_term = 0.98
        self.ratio_threshold_rew = 0.75
        self.collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reward_layout_ratio_buf = deque(maxlen=self.buffer_size)   # surface coverage ratio
        self.reward_layout_ratio_buf.extend(self.buffer_size * [torch.zeros(self.num_envs, device=self.device)])

        # NOTE: only compute once
        self.blender2opencv = torch.FloatTensor([[1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]]).to(self.device)
        intrinsics = self.get_camera_intrinsics()   # [3, 3]
        self.inv_intri = torch.linalg.inv(intrinsics).to(self.device).to(torch.float32)

        H, W = self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width
        downsample_factor = 1
        xs = torch.linspace(0, W-downsample_factor, int(W/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys = torch.linspace(0, H-downsample_factor, int(H/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        norm_coord_pixel = torch.stack([xs, ys], dim=-1)    # [H, W, 2]
        norm_coord_pixel = torch.concat((norm_coord_pixel, 
                                         torch.ones_like(norm_coord_pixel[..., :1], device=self.device)), dim=-1).view(-1, 3)  # [H*W, 3], (u, v, 1)
        self.norm_coord_pixel_around = norm_coord_pixel.repeat(self.num_cam, 1)   # [num_cam*H*W, 3]


        init_poses = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        for env_idx in range(self.num_envs):
            init_poses[env_idx, :2] = self.init_maps_list[env_idx][random.randint(0, len(self.init_maps_list[env_idx])-1)]

        init_poses[:, 2] = self.motion_height
        self.poses = init_poses.clone()
        self.pose_size = self.poses.shape[1]

        self.pose_buf = []
        self.pose_buf += 10 * [self.poses.clone()]
        self.world_pose_buf = torch.zeros(self.num_envs, self.buffer_size, self.pose_size, dtype=torch.float32, device=self.device)  # [num_env, buffer_size, action_size]
        self.ego_pose_buf = torch.zeros(self.num_envs, self.buffer_size, self.pose_size, dtype=torch.float32, device=self.device)  # [num_env, buffer_size, action_size]

        # NOTE: representations
        self.scanned_gt_grid = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                          dtype=torch.float32, device=self.device)

        self.prob_grid = torch.zeros(self.num_envs, self.grid_size//2, self.grid_size//2, self.grid_size//2,
                                    dtype=torch.float32, device=self.device)

        # [1, 1, 3, 3, 3]. Define a 3x3x3 convolutional kernel to look at 6-connected neighbors
        self.frontier_kernel = torch.tensor([[[0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]],
                                          [[0, 1, 0],
                                            [1, 0, 1],
                                            [0, 1, 0]],
                                          [[0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        self.poses_idx = torch.tensor([self.grid_size//2, self.grid_size//2], 
                                      dtype=torch.int32, device=self.device).repeat(self.num_envs, 1)

        self.zero_grid_tensor = torch.zeros((self.num_envs, self.grid_size**3), dtype=torch.int32, device=self.device)

        self._init_buffers_visual()

        # NOTE: visualization
        if self.visualize_flag:
            self.vis_obj_idx = 0
            print("Visualization object index: ", self.vis_obj_idx)

            self.scanned_pc_coord = [[] for _ in range(self.num_envs)]
            self.scanned_pc_color = [[] for _ in range(self.num_envs)]

            self.reset_once_flag = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.reset_once_cr = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.local_paths = [[] for _ in range(self.num_envs)]

            # self.save_path = f'./active_reconstruction/scripts/video/procthor_3D_xrooms_32_3D_bench_2D_action'
            self.save_path = f'./active_reconstruction/scripts/video/procthor_3D_xrooms_32_disc_xy'
            os.makedirs(self.save_path, exist_ok=True)

    def update_pose(self, ego_actions, e2w):
        """
        Update the pose of the robot in the world coordinate, including updating self.move_dist.
        Params:
            ego_actions: [num_env, 6], (x, y, z, roll, pitch, yaw), in egocentric coordinate. 
                         To unify the coordinate system, the policy predicts the normalized egocentric action in the range of [-1, 1].
            e2w: [num_env, 3, 3], egocentric to world coordinate
        """

        cur_poses = self.poses.clone()

        ego_move_xyz = ego_actions[:, :3].clone().to(torch.float32)    # [num_env, 3], z-axis = 0 for x-y action space
        ego_move_xyz[:, :2] = (ego_move_xyz[:, :2] - self.grid_size / 2) * self.voxel_size_gt_scenes[:, :2]   # world coordinate
        ego_move_xyz.unsqueeze_(-1)  # [num_env, 3, 1]

        world_move_xyz = torch.bmm(e2w, ego_move_xyz).squeeze(-1)   # [num_env, 3], world coordinate

        self.poses[:, :3] += world_move_xyz

        self.clip_pose_map_bound()

        self.move_dist = self.poses[:, :3] - cur_poses[:, :3] # NOTE: must be after clip_pose_map_bound(), in case of value truncation

    def reset(self):
        """ Initialization: Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # reset all envs, for example, self.last_actions of all envs will be reset to torch.ones(6)

        # initial pose
        self.actions = torch.clip(self.actions, self.clip_actions_low, self.clip_actions_up).to(self.device)
        self.e2w = self.create_e2w_from_poses(self.poses, self.device)    # [num_env, 4, 4]

        self.update_pose(ego_actions=self.actions, e2w=self.e2w)

        self.set_state_teleportation(self.poses)
        obs = self.post_physics_step(if_reset=True) # NOTE: look around setup
        return obs

    def step(self, actions):
        self.actions = torch.clip(actions, self.clip_actions_low, self.clip_actions_up)

        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if len(env_ids) != 0:
            self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action, 
                                                 dtype=torch.int64, device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]
            self.e2w[env_ids] = self.create_e2w_from_poses(self.poses[env_ids], self.device)    # [num_env, 3, 3]

        current_poses = self.poses.clone()
        self.update_pose(ego_actions=self.actions, e2w=self.e2w)    # update self.poses in world coordinate
        self.set_collision_cam_pose(current_poses, self.move_dist)  # set the pose of collision camera

        self.set_state_teleportation(self.poses)
        obs, rewards, dones, infos = self.post_physics_step()   # NOTE: look around setup
        return obs, rewards, dones, infos

    def update_occ_grid(self):
        # [num_env, num_cam, 4, 4], Update camera view matrices
        extrinsics = self.get_camera_view_matrix() # [num_env*num_cam, 4, 4]
        c2w = torch.linalg.inv(extrinsics.transpose(-2, -1)) @ self.blender2opencv.unsqueeze(0) # [num_env*num_cam, 4, 4]
        c2w = c2w.reshape(self.num_envs, self.num_cam, 4, 4)
        c2w[:, :, :3, 3] -= self.env_origins.unsqueeze(1)
        self.c2ws = c2w # [num_env, num_cam, 4, 4]

        if not self.visualize_flag:
            pts_target = self.back_projection_stack(downsample_factor=1)   # [num_env, num_point, 3], where num_point == num_stack * H * W
        else:
            pts_target, color_target = self.back_projection_stack(downsample_factor=1, visualize=True)   # [num_env, num_point, 3], num_point == H * W
            for env_idx in range(self.num_envs):
                if self.reset_once_flag[env_idx]:
                    continue
                self.scanned_pc_coord[env_idx].append(pts_target[env_idx].cpu().clone())
                self.scanned_pc_color[env_idx].append(color_target[env_idx].cpu().clone())

        # num_layer lists of (num_valid_pts_idx, 3)
        pts_idx_all = scanned_pts_to_idx_3D(pts_target=pts_target,
                                            range_gt_scenes=self.range_gt_scenes,
                                            voxel_size_scenes=self.voxel_size_gt_scenes,
                                            map_size=self.grid_size)

        # [num_env, 3]
        self.poses_idx_old = self.poses_idx.to(torch.int32).clone()
        pose_idx_3D = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(),    # last_pose_xyz
                                            range_gt_scenes=self.range_gt_scenes,
                                            voxel_size_scenes=self.voxel_size_gt_scenes,
                                            map_size=self.grid_size)

        pose_idx_3D_ds = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(),    # last_pose_xyz
                                            range_gt_scenes=self.range_gt_scenes,
                                            voxel_size_scenes=self.voxel_size_gt_scenes * 2,
                                            map_size=self.grid_size / 2)

        self.poses_idx = pose_idx_3D.to(torch.int32).clone()

        # [num_env, 3, 3]
        self.e2w = self.create_e2w_from_poses(poses=self.poses,
                                              device=self.device)

        # [num_env, H, W], update prob grid as representation
        occ_grids = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                dtype=torch.float32, device=self.device)
        for env_idx in range(self.num_envs):
            pts_idx_3D = pts_idx_all[env_idx]   # [num_point, 3]

            if (isinstance(pts_idx_3D, list) and len(pts_idx_3D) == 0) or pts_idx_3D.shape[0] == 0:
                continue

            pts_idx_3D_ds = torch.div(pts_idx_3D, 2, rounding_mode="floor")
            pts_idx_3D_ds = torch.unique(pts_idx_3D_ds, dim=0, sorted=False)

            # coverage
            # pts_idx_3D = pts_idx_3D.to("cpu")
            occ_grids[env_idx, pts_idx_3D[:, 0], pts_idx_3D[:, 1], pts_idx_3D[:, 2]] = 1.0

            # [num_point, 3] for representation
            ray_cast_paths_3D_ds = Bresenham3D_PyCUDA(pts_source=pose_idx_3D_ds[env_idx: env_idx+1],
                                                    pts_target=pts_idx_3D_ds,
                                                    map_size=self.grid_size//2)

            self.prob_grid[env_idx, ray_cast_paths_3D_ds[:, 0], ray_cast_paths_3D_ds[:, 1], ray_cast_paths_3D_ds[:, 2]] -= 0.05
            self.prob_grid[env_idx, pts_idx_3D_ds[:, 0], pts_idx_3D_ds[:, 1], pts_idx_3D_ds[:, 2]] = 1.0

        # NOTE: coverage computation
        self.scanned_gt_grid = torch.clip(
            self.scanned_gt_grid + occ_grids * self.grids_gt_scenes,
            max=1, min=0
        )

        # NOTE: representation computation
        # [num_env, X_ds, Y_ds, Z_ds] in world coordinate, where {-1: free, 0: unknown, 1: occupied}
        self.occ_grids_tri_cls = grid_occupancy_tri_cls(self.prob_grid,
                                                        threshold_occu=0.5,
                                                        threshold_free=0.0,
                                                        return_tri_cls_only=True)

        ego_prob_grids = self.create_ego_grids_3D_yaw0(global_map=self.occ_grids_tri_cls.clone(),
                                                        pose_idx=pose_idx_3D_ds,
                                                        local_size=self.grid_size//2)

        # [num_env, H, W], recoginize frontier
        ego_occ_masks = (ego_prob_grids != 1.0).to(torch.bool)
        ego_frontier_masks = self.compute_frontier_grid(ego_prob_grids=ego_prob_grids, 
                                                        frontier_kernel=self.frontier_kernel)
        ego_frontier_masks = ego_frontier_masks & ego_occ_masks
        ego_prob_grids[ego_frontier_masks] = 2.0

        ego_prob_grids_repre = ego_prob_grids.clone().permute(0, 3, 1, 2).unsqueeze(1)   # [num_env, 1, Z, X, Y]
        ego_prob_grids_repre = F.max_pool3d(ego_prob_grids_repre, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # [num_env, 1, Z/2, X, Y]
        self.ego_prob_grids_repre = F.max_pool3d(ego_prob_grids_repre, kernel_size=(5, 1, 1), stride=(4, 1, 1), padding=(1, 0, 0)).squeeze(1)   # [num_env, Z/8, X, Y]

    def update_observation(self):
        self.update_pose_buf()
        self.update_occ_grid()

    def check_motion_collsion_local_3D(self):
        """ check collision in motion, including rigid body collision and local planning collision"""

        # # collision by contact force
        # self.collision_rigid = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.collision_rigid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cur_pose_idx = self.poses_idx.to(torch.long)
        for env_idx in range(self.num_envs):
            self.collision_rigid[env_idx] = self.grids_gt_scenes[env_idx, cur_pose_idx[env_idx, 0], cur_pose_idx[env_idx, 1], cur_pose_idx[env_idx, 2]] == 1.0

        # collision by local planner
        H_c = int(self.depth_processed_col.shape[1] / 2) - 1
        W_c = int(self.depth_processed_col.shape[2] / 2) - 1
        depth_center_area = torch.stack([self.depth_processed_col[:, H_c, W_c], self.depth_processed_col[:, H_c, W_c+1], 
                                         self.depth_processed_col[:, H_c+1, W_c], self.depth_processed_col[:, H_c+1, W_c+1]], dim=1)     # [num_env, 4]
        depth_center_area = torch.abs(depth_center_area)
        min_vis_dist, _ = torch.min(depth_center_area, dim=1)  # [num_env]
        
        move_dist = torch.norm(self.move_dist, dim=-1)
        self.collision_vis = (min_vis_dist < move_dist - 0.15)
        self.collision_vis[self.cur_episode_length == 0] = False

        if self.visualize_flag:
            self.collision_vis_ori = self.collision_vis.clone()
        flag_no_need_local = torch.logical_or(self.cur_episode_length == 0, 
                                              torch.logical_or(self.collision_rigid, self.collision_vis == False))
        env_ids = ~flag_no_need_local   # need local planner to turn collision_vis to False

        if torch.sum(env_ids) == 0:
            pass
        else:
            occ_grids_bin_cls = 1. - (self.occ_grids_tri_cls[env_ids] >= 0.).to(torch.float32)  # [num_env, X, Y, Z]

            local_path_length = self.bfs_pathfinding_3D(occ_grids_bin_cls.to(torch.float32),
                                                        self.poses_idx_old[env_ids],
                                                        self.poses_idx[env_ids])    # [num_env]
            mask = torch.logical_or(local_path_length == -1., 
                                    local_path_length >= self.grid_size // 2)
            self.collision_vis[env_ids] = mask.clone()  # if path exists, collision_vis: True -> False

        # NOTE: collision flag
        self.collision_flag = torch.logical_or(self.collision_rigid, self.collision_vis)
        # print(self.collision_flag)

    def bfs_pathfinding_3D(self, occupancy_grids, starts, goals):
        """
        Perform BFS-based pathfinding using a custom CUDA extension.

        Parameters:
            occupancy_maps: torch.Tensor of shape [num_env, X, Y, Z]
            starts: torch.Tensor of shape [num_env, 3]
            goals: torch.Tensor of shape [num_env, 3]
            path_lengths: torch.Tensor of shape [num_env], default to -1
        Returns:
            path_lengths: torch.Tensor of shape [num_env], containing path lengths
        """
        num_env_local = occupancy_grids.shape[0]
        grid_size = occupancy_grids.shape[1]

        starts = torch.clip(starts, min=0, max=grid_size-1).to(torch.int32)
        goals = torch.clip(goals, min=0, max=grid_size-1).to(torch.int32)

        path_lengths = torch.full((num_env_local,), -1.0, 
                                  dtype=torch.float32, device=self.device)
        visited = torch.zeros((num_env_local, self.grid_size**3), dtype=torch.bool, device=self.device)
        distances = torch.zeros((num_env_local, self.grid_size**3), dtype=torch.int32, device=self.device)
        queue = torch.zeros((num_env_local, self.grid_size**3), dtype=torch.int32, device=self.device)

        # bfs_cuda_3D.BFS_CUDA_3D(occupancy_grids, starts, goals, path_lengths)
        bfs_cuda_3D.BFS_CUDA_3D(occupancy_grids.contiguous(), starts.contiguous(), goals.contiguous(), path_lengths, 
                                visited, distances, queue)
        return path_lengths

    def post_physics_step(self, if_reset=False):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1

        obs, rewards, dones, infos = self.get_step_return()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return (obs, rewards, dones, infos) if not if_reset else obs

    def get_step_return(self):
        # render sensors and refresh camera tensors
        assert self.cfg.return_visual_observation, \
            "Images should be returned in this environment!"
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        self.post_process_main_camera_tensor()  # [num_env, num_cam, H, W]
        self.gym.end_access_image_tensors(self.sim)

        self.update_observation()
        self.check_motion_collsion_local_3D()   # must be called before reward computing, if rew_collision exists
        self.compute_reward()                   # including self.check_termination()

        obs = {
            "state": self.ego_pose_buf.view(self.num_envs, -1),  # [num_env, pose_buffer_size * pose_size]
            "ego_grid": self.ego_prob_grids_repre.reshape(self.num_envs, -1), # [num_env, H * W]
        }

        if self.visualize_flag:
            # self.debug_save_image_stack()
            self.visualize_all_3D_pts()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        rewards, dones, infos = self.rew_buf, self.reset_buf.clone(), self.extras
        self.update_extra_episode_info(rewards=rewards, dones=dones)
        self.reset_buf[env_ids] = 0

        return obs, rewards, dones, infos

    def _reset_root_states(self, env_ids):
        """ Resets the root states of agents in envs to be reseted
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[::self.skip][env_ids] = self.base_init_state
            self.root_states[::self.skip][env_ids, :3] += self.env_origins[env_ids]
            self.root_states[::self.skip][env_ids, :2] += torch_rand_float(
                -1., 1., (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:   # <-
            self.root_states[::self.skip][env_ids] = self.base_init_state
            self.root_states[::self.skip][env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.clone().to(dtype=torch.int32) * self.skip
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        num_env_ids = len(env_ids)
        device = self.device

        if num_env_ids == 0:
            return

        # reset robot states
        self._reset_root_states(env_ids)

        # Reset pose buffers
        self.ego_pose_buf[env_ids] = torch.zeros(num_env_ids, self.buffer_size, self.pose_size, 
                                                 dtype=torch.float32, device=device)
        self.world_pose_buf[env_ids] = torch.zeros(num_env_ids, self.buffer_size, self.pose_size, 
                                                   dtype=torch.float32, device=device) 

        # Reset reward buffers
        zero_rew_tensor = torch.zeros(num_env_ids, dtype=torch.float32, device=device)
        for buf_idx in range(self.buffer_size):
            self.reward_layout_ratio_buf[buf_idx][env_ids] = zero_rew_tensor

        # Reset actions
        self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action, 
                                            dtype=torch.int64, device=device).repeat(len(env_ids), 1)

        # NOTE: Reset initial poses randomly
        for env_idx in env_ids:
            self.poses[env_idx, :2] = self.init_maps_list[env_idx][random.randint(0, len(self.init_maps_list[env_idx])-1)]
        self.poses[:, 2] = self.motion_height

        if self.visualize_flag:
            for env_idx in env_ids:
                self.local_paths[env_idx] = []

        # Reset maps
        self.prob_grid[env_ids] = torch.zeros(num_env_ids, self.grid_size//2, self.grid_size//2, self.grid_size//2,
                                      dtype=torch.float32, device=device)
        self.scanned_gt_grid[env_ids] = torch.zeros(num_env_ids, self.grid_size, self.grid_size, self.grid_size,
                                      dtype=torch.float32, device=device)

        # Reset episode buffer
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Update extras and episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  # 'rollout/rew_' in wandb
            self.episode_sums[key][env_ids] = 0.    # terminate and create new episode

        # log additional curriculum info and usend timeout info to the algorithm
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def update_observation_space(self):
        """ update observation and action space
        """
        if not self.cfg.return_visual_observation:
            return

        # action space: discrete
        action_space_size = (self.clip_actions_up - self.clip_actions_low + 1).cpu().numpy()  # [6]
        self.action_space = MultiDiscrete(nvec=torch.Size(action_space_size))

        # observation space
        x_max = self.range_gt[:, 0].max().item()
        x_min = self.range_gt[:, 1].min().item()
        y_max = self.range_gt[:, 2].max().item()
        y_min = self.range_gt[:, 3].min().item()
        z_max = self.range_gt[:, 4].max().item()
        z_min = self.range_gt[:, 5].min().item()

        clip_pose_world_up = [x_max, y_max, self.motion_height, 0, 0, 0]
        clip_pose_world_low = [x_min, y_min, self.motion_height, 0, 0, 0]

        pose_up_bound = np.tile(clip_pose_world_up, self.buffer_size).astype(np.float32)
        pose_low_bound = np.tile(clip_pose_world_low, self.buffer_size).astype(np.float32)

        self.observation_space = Dict(
            {
                "state": Box(low=pose_low_bound, high=pose_up_bound, shape=(self.buffer_size * self.pose_size, ), dtype=np.float32),   # view actions as state
                # "ego_grid": Box(low=-1., high=2., shape=(16 * 128 * 128, ), dtype=np.float32),
                "ego_grid": Box(low=-1., high=2., shape=(8 * 64 * 64, ), dtype=np.float32),
            }
        )
        pass

    def _reward_surface_coverage(self):
        """ Reward for exploring the surface coverage of scenes."""

        layout_coverage = self.scanned_gt_grid.sum(dim=(1, 2, 3)) / self.num_valid_voxel_gt_scenes
        self.reward_layout_ratio_buf.extend([layout_coverage.clone()])

        return self.reward_layout_ratio_buf[-1] - self.reward_layout_ratio_buf[-2]
