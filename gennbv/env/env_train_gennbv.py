"""
This environment is based on drone env, while we ignore the drone control problem by directly setting the state of drone
"""
import os
from collections import deque
from gym.spaces import Box, Dict, MultiDiscrete
from isaacgym import *
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
import numpy as np
from gennbv.utils import scanned_pts_to_idx_3D, pose_coord_to_idx_3D, \
                                        bresenham3D_pycuda, grid_occupancy_tri_cls
from gennbv.env.env_train_base import Env_Train_Base
from legged_gym import OPEN_ROBOT_ROOT_DIR


class Env_Train_GenNBV(Env_Train_Base):
    num_scene = 256

    def _additional_create(self, env_handle, env_index):
        assert self.cfg.return_visual_observation, "visual observation should be returned!"

        # NOTE: urdf load, create actor
        urdf_path = "data_gennbv/train/urdf"

        # env_index: [0, num_env-1] -> obj_index: [0, num_scene-1]
        obj_index = env_index % self.num_scene
        batch_idx = obj_index // 50 + 1
        urdf_idx = obj_index % 50 + 1
        if batch_idx == 6:
            batch_idx = 11  # skip 6-10

        urdf_name = f"house_batch{batch_idx}_setA_{urdf_idx}.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset = self.gym.load_asset(self.sim, urdf_path, urdf_name, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.env_origins[env_index][0],
                                self.env_origins[env_index][1],
                                self.env_origins[env_index][2])  # place the box at center point
        pose.r = gymapi.Quat(0, 0, 0, 1)

        ahandle = self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)

        # scale_factor = 1.0
        # self.gym.set_actor_scale(env_handle, ahandle, scale_factor)
        self.gym.set_rigid_body_segmentation_id(env_handle, ahandle, 0, self.OBJECT_SEGMENTATION_ID)

        self.additional_actors[env_index] = [ahandle]

    def _init_load_all(self):
        """
        load ground truth data for training.
        """
        # [num_scene, X, Y, Z, 4]
        grid_gt = torch.load(
            os.path.join(OPEN_ROBOT_ROOT_DIR, "data_gennbv/train/gt/train_houses3k_grid_gt.pt"),
            map_location=self.device
        )

        # [num_scene, 3]
        self.voxel_size_gt = torch.cat([grid_gt[:, 1, 0, 0, 0:1] - grid_gt[:, 0, 0, 0, 0:1],
                                        grid_gt[:, 0, 1, 0, 1:2] - grid_gt[:, 0, 0, 0, 1:2],
                                        grid_gt[:, 0, 0, 1, 2:3] - grid_gt[:, 0, 0, 0, 2:3]], dim=-1)

        # [num_scene]
        self.num_valid_voxel_gt = grid_gt[..., 3].sum(dim=(-1, -2, -3))

        # [num_scene, 6], (x_max, x_min, y_max, y_min, z_max, z_min)
        x_range = grid_gt[:, -1, 0, 0, 0:1] - grid_gt[:, 0, 0, 0, 0:1]
        y_range = grid_gt[:, 0, -1, 0, 1:2] - grid_gt[:, 0, 0, 0, 1:2]
        z_range = grid_gt[:, 0, 0, -1, 2:3] - grid_gt[:, 0, 0, 0, 2:3]
        self.range_gt = torch.cat([x_range / 2, -x_range / 2,
                                   y_range / 2, -y_range / 2,
                                   z_range, torch.zeros_like(z_range)], dim=-1)

        # [X, Y, Z]
        self.grid_size = grid_gt.shape[1]
        assert grid_gt.shape[1] == grid_gt.shape[2] == grid_gt.shape[3]

        # num_scene (scenes from dataset) -> num_env (training env)
        self.env_to_scene = []
        for env_idx in range(self.num_envs):
            self.env_to_scene.append(env_idx % self.num_scene)
        self.env_to_scene = torch.tensor(self.env_to_scene, device=self.device)     # [num_env]

        self.grid_gt = grid_gt[..., 3] # [num_scene, X, Y, Z, 4] -> [num_scene, X, Y, Z]
        self.grid_gt = self.grid_gt[self.env_to_scene]
        self.voxel_size_gt = self.voxel_size_gt[self.env_to_scene]
        self.num_valid_voxel_gt = self.num_valid_voxel_gt[self.env_to_scene]
        self.range_gt = self.range_gt[self.env_to_scene]

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

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # [num_envs, num_bodies, xyz axis]
        self.rigid_state = gymtorch.wrap_tensor(rigid_state).view(
            self.num_envs, -1, 13
        )  # [num_envs, num_bodies, 13]

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.buffer_size = self.cfg.visual_input.stack
        self.cur_reward_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # actions: current actions
        init_action = self.cfg.normalization.init_action
        action_unit = self.cfg.normalization.action_unit
        self.actions = torch.tensor(
            init_action, dtype=torch.long, device=self.device, requires_grad=False
        ).repeat(self.num_envs, 1)
        self.action_unit = torch.tensor(action_unit, device=self.device)
        self.action_size = self.actions.shape[1]
        self.action_low_world = torch.tensor(
            self.cfg.normalization.clip_pose_low, device=self.device
        )

        # NOTE: create buffers for active reconstruction
        self.clip_pose_idx_low = torch.tensor(
            self.cfg.normalization.clip_pose_idx_low, dtype=torch.int64, device=self.device
        )
        self.clip_pose_idx_up = torch.tensor(
            self.cfg.normalization.clip_pose_idx_up, dtype=torch.int64, device=self.device
        )

        pose_buf = torch.tensor(self.cfg.normalization.init_pose_buf, dtype=torch.float, 
                                device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        self.pose_buf = deque(maxlen=self.buffer_size)
        self.pose_buf.extend(self.buffer_size * [pose_buf])

        # NOTE: reward functions
        self.ratio_threshold_term = 0.99
        self.reward_ratio_buf = deque(maxlen=max(self.buffer_size, 2))   # surface coverage ratip
        self.reward_ratio_buf.extend(max(self.buffer_size, 2) * [torch.zeros(self.num_envs, device=self.device)])
        self.collision_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

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
        self.norm_coord_pixel = torch.concat((norm_coord_pixel, 
                                         torch.ones_like(norm_coord_pixel[..., :1], device=self.device)), dim=-1).view(-1, 3)  # [H*W, 3], (u, v, 1)

        # NOTE: representations
        self.scanned_gt_grid = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                          dtype=torch.float32, device=self.device)

        self.prob_grid = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                    dtype=torch.float32, device=self.device)

        self.occ_grids_tri_cls = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                            dtype=torch.float32, device=self.device)

        self._init_buffers_visual()

        self.k = 2
        self.rgb_h = 64
        self.rgb_w = 64
        self.rgb_buf = deque(maxlen=self.k)
        self.rgb_buf.extend(self.k * [torch.zeros((self.num_envs, 1, self.rgb_h, self.rgb_w),
                                                  dtype=torch.float32, device=self.device)])

        self.pts_target_list = []

    def _init_buffers_visual(self):
        if not self.cfg.return_visual_observation:
            return

        self.rgb_cam_tensors = []
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i],
                                                      gymapi.IMAGE_COLOR)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.rgb_cam_tensors.append(torch_cam_tensor)

        self.depth_cam_tensors = []
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i],
                                                      gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.depth_cam_tensors.append(torch_cam_tensor)

        self.seg_cam_tensors = []
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i],
                                                      gymapi.IMAGE_SEGMENTATION)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.seg_cam_tensors.append(torch_cam_tensor)

    def reset(self):
        """ Initialization: Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # reset all envs

        # initial pose
        self.actions = torch.clip(self.actions, self.clip_pose_idx_low, self.clip_pose_idx_up).to(self.device)    # pose idx in world coordinate
        self.poses = self.get_pose_from_discrete_action(action=self.actions)   # poses in world coordinate

        self.render()

        self.set_state(self.poses)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs = self.post_physics_step(if_reset=True)
        return obs

    def step(self, actions):
        self.actions = torch.clip(actions, self.clip_pose_idx_low, self.clip_pose_idx_up)

        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if len(env_ids) != 0:
            self.actions[env_ids] = torch.tensor(
                self.cfg.normalization.init_action, device=self.device, requires_grad=False
            ).repeat(self.num_envs, 1)[env_ids]

        self.poses = self.get_pose_from_discrete_action(action=self.actions)

        self.render()

        self.set_state(self.poses)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs, rewards, dones, infos = self.post_physics_step()
        return obs, rewards, dones, infos

    def update_observation(self):
        """
        Update observations at each step, including historyical poses, rgb images and probabilistic grids.
        """
        self.update_obs_buf()   # update historical poses and rgb images
        self.update_occ_grid()  # update probabilistic grids

    def update_obs_buf(self):
        self.pose_buf.extend([self.poses])
        self.rgb_buf.extend([self.rgb_grayscale])

    def update_occ_grid(self):
        """ Update scanned probabilistic grids. """
        pts_target = self.back_projection_fg(downsample_factor=1)   # list of target points, num_env * [n, 3]

        # num_layer lists of (num_valid_pts_idx, 3)
        pts_idx_all = scanned_pts_to_idx_3D(pts_target=pts_target,
                                            range_gt=self.range_gt,
                                            voxel_size_gt=self.voxel_size_gt,
                                            map_size=self.grid_size)

        pose_idx_3D = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(),
                                                range_gt=self.range_gt,
                                                voxel_size_gt=self.voxel_size_gt,
                                                map_size=self.grid_size)

        # [num_env, H, W], update prob grid as representation
        occ_grids = torch.zeros(self.num_envs, self.grid_size, self.grid_size, self.grid_size,
                                dtype=torch.float32, device=self.device)
        for env_idx in range(self.num_envs):
            pts_idx_3D = pts_idx_all[env_idx]   # [num_point, 3]

            if (isinstance(pts_idx_3D, list) and len(pts_idx_3D) == 0) or pts_idx_3D.shape[0] == 0:
                continue

            pts_idx_3D = torch.unique(pts_idx_3D, dim=0, sorted=False)

            # coverage
            occ_grids[env_idx, pts_idx_3D[:, 0], pts_idx_3D[:, 1], pts_idx_3D[:, 2]] = 1.0

            # [num_point, 3] for representation
            ray_cast_paths_3D = bresenham3D_pycuda(pts_source=pose_idx_3D[env_idx: env_idx+1],
                                                        pts_target=pts_idx_3D,
                                                        map_size=self.grid_size)

            self.prob_grid[env_idx, ray_cast_paths_3D[:, 0], \
                           ray_cast_paths_3D[:, 1], ray_cast_paths_3D[:, 2]] -= 0.05
            self.prob_grid[env_idx, pts_idx_3D[:, 0], \
                           pts_idx_3D[:, 1], pts_idx_3D[:, 2]] = 1.0

        # NOTE: representation computation
        # [num_env, X_ds, Y_ds, Z_ds] in world coordinate, where {-1: free, 0: unknown, 1: occupied}
        self.occ_grids_tri_cls = grid_occupancy_tri_cls(self.prob_grid,
                                                        threshold_occu=0.5,
                                                        threshold_free=0.0,
                                                        return_tri_cls_only=True)
        # NOTE: reward computation
        self.scanned_gt_grid = torch.clip(
            self.scanned_gt_grid + occ_grids * self.grid_gt,
            max=1, min=0
        )

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
        assert self.cfg.return_visual_observation, \
            "Images should be returned in this environment!"
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        self.post_process_camera_tensor()   # update self.xxx_processed, where xxx among {depth, rgb, seg}
        self.gym.end_access_image_tensors(self.sim)

        self.update_observation()
        self.compute_reward()   # including self.check_termination()

        obs = {
            # [num_env, buffer_size * pose_size]
            "state": torch.stack(tuple(self.pose_buf), dim=1),
            # [num_env, K, H, W]
            "state_rgb": torch.cat(tuple(self.rgb_buf), dim=1),
            # [num_envs, X, Y, Z]
            "grid": self.occ_grids_tri_cls,
        }

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        rewards, dones, infos = self.rew_buf, self.reset_buf.clone(), self.extras
        self.update_extra_episode_info(rewards=rewards, dones=dones)
        self.reset_buf[env_ids] = 0

        return obs, rewards, dones, infos

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

        # reset buffers
        for buf_idx in range(self.buffer_size):
            self.pose_buf[buf_idx][env_ids] = torch.tensor(
                self.cfg.normalization.init_pose_buf, dtype=torch.float32, device=self.device, requires_grad=False
            ).repeat(num_env_ids, 1)
            self.reward_ratio_buf[buf_idx][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]
        for buf_idx in range(self.k):
            self.rgb_buf[buf_idx][env_ids] = torch.zeros(
                (self.num_envs, 1, self.rgb_h, self.rgb_w), dtype=torch.float32, device=self.device
            )[env_ids]

        if self.buffer_size == 1:
            self.reward_ratio_buf[1][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]

        # Reset actions
        self.actions[env_ids] = torch.tensor(
            self.cfg.normalization.init_action, device=device
        ).repeat(len(env_ids), 1)

        # Reset maps
        self.prob_grid[env_ids] = torch.zeros(num_env_ids, self.grid_size, self.grid_size, self.grid_size,
                                      dtype=torch.float32, device=device)
        self.scanned_gt_grid[env_ids] = torch.zeros(num_env_ids, self.grid_size, self.grid_size, self.grid_size,
                                      dtype=torch.float32, device=device)

        # Reset episode buffer
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Update extras and episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.    # terminate and create new episode

        # log additional curriculum info and usend timeout info to the algorithm
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def check_termination(self):
        """ Check if environments need to be reset
        Termination conditions:
            1. collision
            2. steps == max_episode_length
            3. coverage ratio threshold
        """
        # collision
        self.collision_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0., dim=1)

        # max_step
        self.reset_buf = self.collision_buf.clone()
        if self.cfg.termination.max_step_done:
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)
            self.reset_buf |= self.time_out_buf

        # coverage ratio > threshold
        ratio_threshold = 0.99
        last_ratio = self.reward_ratio_buf[-1]
        self.reset_buf |= (last_ratio > ratio_threshold)

    def update_observation_space(self):
        """ update observation and action space
        """
        if not self.cfg.return_visual_observation:
            return

        # action space: discrete
        action_space_size = (self.clip_pose_idx_up - self.clip_pose_idx_low + 1).cpu().numpy()  # [6]
        self.action_space = MultiDiscrete(nvec=torch.Size(action_space_size))

        # observation space
        x_max = self.range_gt[:, 0].max().item()
        x_min = self.range_gt[:, 1].min().item()
        y_max = self.range_gt[:, 2].max().item()
        y_min = self.range_gt[:, 3].min().item()
        z_max = self.range_gt[:, 4].max().item()
        z_min = self.range_gt[:, 5].min().item()

        clip_pose_world_up = [x_max, y_max, z_max, 0, 1/2*np.pi, 2*np.pi]
        clip_pose_world_low = [x_min, y_min, z_min, 0, -1/2*np.pi, 0]

        pose_up_bound = np.tile(clip_pose_world_up, self.buffer_size).astype(np.float32)
        pose_low_bound = np.tile(clip_pose_world_low, self.buffer_size).astype(np.float32)

        self.observation_space = Dict(
            {
                "state": Box(low=pose_low_bound, high=pose_up_bound, 
                             shape=(self.buffer_size * self.action_size, ), dtype=np.int64),
                "state_rgb": Box(low=0, high=255, 
                                 shape=(self.k * self.rgb_h * self.rgb_w, ), dtype=np.int64),
                "grid": Box(low=-torch.inf, high=torch.inf, 
                            shape=(self.grid_size, self.grid_size, self.grid_size), dtype=np.float32),
            }
        )

    def back_projection_fg(self, downsample_factor=1, visualize=False):
        """ Back-projection of foreground depth maps to 3D points in world coordinate
        Args:
            downsample_factor: downsample factor
            visualize: if True, also return colors_world
        Returns:
            coords_world: list of [num_points, 3]
            colors_world: list of [num_points, 3] (if visualize)
        """
        depth_maps = self.depth_processed.clone()   # [num_env, H, W]
        depth_maps_fg = (self.seg_processed.clone() > 50)
        if downsample_factor != 1:
            depth_maps = depth_maps[:, ::downsample_factor, ::downsample_factor]   # [num_env, H_down, W_down]
            depth_maps_fg = depth_maps_fg[:, ::downsample_factor, ::downsample_factor]

        depth_maps[~depth_maps_fg] = 0.

        # NOTE: back-projection
        extrinsics = torch.from_numpy(self.get_camera_view_matrix()).to(self.device) # [num_env, 4, 4]
        c2w = torch.linalg.inv(extrinsics.transpose(-2, -1)) @ self.blender2opencv.unsqueeze(0)
        c2w[:, :3, 3] -= self.env_origins

        # num_point == H * W
        depth_maps = depth_maps.reshape(self.num_envs, -1)          # [num_env, H*W]
        depth_maps_fg = depth_maps_fg.reshape(self.num_envs, -1)    # [num_env, H*W]
        coords_pixel = torch.einsum('ij,jk->ijk', depth_maps, self.norm_coord_pixel)   # [num_env, num_point, 3]

        # inv_intri: [3, 3], coords_pixel: [num_env, num_point, 3]
        coords_cam = torch.einsum('ij,nkj->nki', self.inv_intri, coords_pixel)    # [num_env, num_point, 3]
        coords_cam_homo = torch.concat((coords_cam, torch.ones_like(coords_cam[..., :1], device=self.device)), dim=-1)   # [num_env, num_point, 4], homogeneous format

        # c2w: [num_env, 4, 4], coord_cam_homo: [num_env, num_point, 4]
        coords_world = torch.einsum('nij,nkj->nki', c2w, coords_cam_homo)[..., :3]    # [num_env, num_point, 4] -> [num_env, num_point, 3]
        coords_world = [coords_world[idx][depth_maps_fg[idx, :]] for idx in range(self.num_envs)]

        # if visualize:
        #     rgb_maps = self.rgb_processed.clone()[:, ::downsample_factor, ::downsample_factor]   # [num_env, H_down, W_down]
        #     colors_world = rgb_maps[..., :3].reshape(self.num_envs, -1, 3)	# [num_env, H*W, 3]
        #     return coords_world, colors_world    # [num_env, num_point, 3]
        return coords_world     # list of [num_points, 3]

    def _reward_surface_coverage(self):
        """ Reward for exploring the surface coverage of scenes."""
        layout_coverage = self.scanned_gt_grid.sum(dim=(1, 2, 3)) / self.num_valid_voxel_gt
        self.reward_ratio_buf.extend([layout_coverage.clone()])
        return self.reward_ratio_buf[-1] - self.reward_ratio_buf[-2]

    def _reward_short_path(self):
        """ Penalty for current episode_length """
        current_length = self.episode_length_buf.clone()
        extra_step = torch.clip(current_length - 30, min=0, max=2)  # reward is computed cumulatively
        return -extra_step  # penalize

    def _reward_collision(self):
        """ Penalize collisions on selected bodies.
        contact_forces: [num_env, num_bodies, 3], bodies: ('base_link', 'center_of_mass_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link') + .obj
        penalised_contact_indices: indices of ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link')
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) # [num_env], "1": collide

    def _reward_termination(self):
        """ Terminal reward / penalty """
        return self.reset_buf * ~self.time_out_buf  # terminate before reaching max_episode_length
