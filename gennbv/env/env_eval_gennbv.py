"""
This environment is based on drone env, while we ignore the drone control problem by directly setting the state of drone
"""
import os
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
from gennbv.utils import scanned_pts_to_idx_3D, pose_coord_to_idx_3D, \
                                        bresenham3D_pycuda, grid_occupancy_tri_cls
from gennbv.env.env_train_gennbv import Env_Train_GenNBV
from legged_gym import OPEN_ROBOT_ROOT_DIR
from pytorch3d.loss import chamfer_distance


class Env_Eval_GenNBV(Env_Train_GenNBV):
    num_scene = 50

    def _additional_create(self, env_handle, env_index):
        assert self.cfg.return_visual_observation, "visual observation should be returned!"

        # NOTE: urdf load, create actor
        urdf_path = "data_gennbv/eval/urdf"

        # env_index: [0, num_env-1] -> obj_index: [0, num_scene-1]
        obj_index = env_index % self.num_scene
        batch_idx = obj_index // 50 + 1
        urdf_idx = obj_index % 50 + 1
        batch_idx = 12

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
        load ground truth data for evaluation.
        """
        # [num_scene, X, Y, Z, 4]
        grid_gt = torch.load(
            os.path.join(OPEN_ROBOT_ROOT_DIR, "data_gennbv/eval/gt/eval_houses3k_grid_gt.pt"),
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

        # list of [num_gt_pts, 3], for accuracy computation
        pc_path = os.path.join(OPEN_ROBOT_ROOT_DIR, "data_gennbv/eval/gt/point_cloud")
        self.pc_gt = [
            torch.load(
                os.path.join(pc_path, f"BAT12_SETA_HOUSE{str(env_idx+1)}_pc.pt"), 
                map_location=self.device
            )
            for env_idx in range(self.num_envs)
        ]

    def _init_buffers(self):
        super()._init_buffers()
        self.ratios_accuracy = dict()

    def reset(self, return_all=True):
        """ Initialization: Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # reset all envs

        # initial pose
        self.actions = torch.clip(self.actions, self.clip_pose_idx_low, self.clip_pose_idx_up).to(self.device)    # pose idx in world coordinate
        self.poses = self.get_pose_from_discrete_action(action=self.actions)   # poses in world coordinate

        self.render()

        self.set_state(self.poses)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs, rewards, dones, infos, accuracies = self.post_physics_step_reset_vis_obs_eval()  # check_termination and reset env

        self.ratios_accuracy = dict()   # reset accuracy (init_buffer only execute once)

        if return_all:
            return obs, rewards, dones, infos, accuracies
        else:
            return obs

    def step(self, actions):
        """
        Set the position (x, y, z) and orientation (r, p, y) for the camera. actions.shape: [1, 6]
        """
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, requires_grad=False)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
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
        return obs, rewards, dones, infos, self.ratios_accuracy

    def update_occ_grid(self):
        """ Update scanned probabilistic grids. """
        pts_target = self.back_projection_fg(downsample_factor=1)   # list of target points, num_env * [n, 3]
        if len(self.pts_target_list) == 0:
            self.pts_target_list = pts_target
        else:
            self.pts_target_list = [torch.cat([self.pts_target_list[env_idx], pts_target[env_idx]], 0) for env_idx in range(self.num_envs)]

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

    def post_physics_step_reset_vis_obs_eval(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1

        obs, rewards, dones, infos = self.get_step_return()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return obs, rewards, dones, infos, self.ratios_accuracy

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
        if env_ids.shape[0] > 0 and self.cur_episode_length.sum() > 0:
            for env_idx in env_ids:
                # NOTE: accuracy: chamfer distance (cm)
                pc_backproj_voxeled = torch.unique(
                    torch.round(self.pts_target_list[env_idx], decimals=2),
                    dim=0
                )  # [num_scanned_pts, 3], resolution: 1cm
                accuracy = (chamfer_distance(pc_backproj_voxeled.unsqueeze(0), \
                                             self.pc_gt[env_idx].unsqueeze(0)) * 100)[0].unsqueeze(0)

                self.ratios_accuracy[str(env_idx.item())] = accuracy.item() if str(env_idx.item()) not in self.ratios_accuracy.keys()\
                    else self.ratios_accuracy[str(env_idx.item())]
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
        for env_idx in env_ids:
            env_idx = env_idx.item()
            if len(self.pts_target_list) != 0:
                self.pts_target_list[env_idx] = torch.empty(0, 3).to(self.device)

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
        """
        # collision
        self.collision_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0., dim=1)
        self.reset_buf |= self.collision_buf

        # max_step
        if self.cfg.termination.max_step_done:
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)
            self.reset_buf |= self.time_out_buf

    def _reward_surface_coverage(self):
        """ Reward for exploring the surface coverage of scenes."""
        layout_coverage = self.scanned_gt_grid.sum(dim=(1, 2, 3)) / self.num_valid_voxel_gt
        self.reward_ratio_buf.extend([layout_coverage.clone()])
        return self.reward_ratio_buf[-1] - self.reward_ratio_buf[-2]
