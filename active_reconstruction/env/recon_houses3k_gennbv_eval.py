import os
import torch
import numpy as np
from collections import deque
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from active_reconstruction.env.recon_houses3k_gennbv import Recon_Houses3K_GenNBV
from active_reconstruction.utils import scanned_pts_to_idx_3D, pose_coord_to_idx_3D, \
                                        grid_occupancy_tri_cls, bresenham3D_pycuda
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.spatial import cKDTree as KDTree
from torchvision.transforms import ToPILImage
from legged_gym import OPEN_ROBOT_ROOT_DIR


class Recon_Houses3K_GenNBV_Eval(Recon_Houses3K_GenNBV):
    env_to_obj = dict()

    # evaluation scenes from batch 12
    num_obj = 50
    batch_idx = 12

    def _additional_create(self, env_handle, env_index):
        assert self.cfg.return_visual_observation, "visual observation should be returned!"

        # NOTE: urdf load, create actor
        dataset_name = "houses3k"
        urdf_path = f"data_gennbv/{dataset_name}/urdf"

        # env_index: [0, num_env-1] -> obj_index: [0, num_obj-1]
        obj_index = env_index % self.num_obj
        self.env_to_obj[str(env_index)] = obj_index
        urdf_name = f"house_batch{self.batch_idx}_setA_{obj_index+1}.urdf"

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
        """ load all ground truth """

        # [num_obj, X, Y, Z], num_obj=256， X=Y=Z=20
        grids_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                               "data_gennbv/houses3k/gt/houses3k_eval_20_grid_gt.pt"), map_location=self.device)
        self.grid_size = grids_gt.shape[1]

        # [num_obj, 3]
        voxel_size_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                     "data_gennbv/houses3k/gt/houses3k_eval_20_voxel_size_gt.pt"), map_location=self.device)
        # [num_obj]
        num_valid_voxel_gt = grids_gt.sum(dim=(1, 2, 3))
        # [num_obj, 6]
        self.range_gt = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                                "data_gennbv/houses3k/gt/houses3k_eval_20_range_gt.pt"), map_location=self.device)

        # list of gt point cloud
        self.layout_pc = [(torch.nonzero(grids_gt[idx]) / (self.grid_size - 1) * 2 - 1) * self.range_gt[idx, ::2]
                          for idx in range(self.num_obj)]

        # num_obj -> num_env
        self.env_to_scene = []
        for idx in range(self.num_envs):
            self.env_to_scene.append(idx % self.num_obj)
        self.env_to_scene = torch.tensor(self.env_to_scene, device=self.device)     # [num_env]

        self.range_gt_scenes = self.range_gt[self.env_to_scene]
        self.voxel_size_gt_scenes = voxel_size_gt[self.env_to_scene]
        self.num_valid_voxel_gt_scenes = num_valid_voxel_gt[self.env_to_scene]
        self.grids_gt_scenes = grids_gt[self.env_to_scene]

        # [num_obj, X, Y, Z], X=Y=Z=128, for collision checking
        grids_gt_col = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                               "data_gennbv/houses3k/gt/houses3k_eval_128_grid_gt.pt"), map_location=self.device)
        # [num_obj, 3], for collision checking
        voxel_size_gt_col = torch.load(os.path.join(OPEN_ROBOT_ROOT_DIR,
                                               "data_gennbv/houses3k/gt/houses3k_eval_128_voxel_size_gt.pt"), map_location=self.device)

        self.grids_gt_col_scenes = grids_gt_col[self.env_to_scene]
        self.voxel_size_gt_col_scenes = voxel_size_gt_col[self.env_to_scene]

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

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # [num_envs, num_bodies, xyz axis]
        self.rigid_state = gymtorch.wrap_tensor(rigid_state).view(
            self.num_envs, -1, 13)  # [num_envs, num_bodies, 13]

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.buffer_size = self.cfg.visual_input.stack
        self.cur_reward_sum = torch.zeros(self.num_envs, 
                                          dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, 
                                              dtype=torch.float, device=self.device)

        # actions: current actions, last_actions: action at last time
        init_action = self.cfg.normalization.init_action
        action_unit = self.cfg.normalization.action_unit
        self.actions = torch.tensor(init_action, dtype=torch.long, device=self.device, 
                                    requires_grad=False).repeat(self.num_envs, 1)
        self.action_unit = torch.tensor(action_unit, device=self.device)
        self.action_size = self.actions.shape[1]
        self.action_low_world = torch.tensor(self.cfg.normalization.clip_pose_low, 
                                             device=self.device) 

        # NOTE: create buffers for active reconstruction
        self.clip_pose_idx_low = torch.tensor(self.cfg.normalization.clip_pose_idx_low, 
                                              dtype=torch.int64, device=self.device)
        self.clip_pose_idx_up = torch.tensor(self.cfg.normalization.clip_pose_idx_up, 
                                             dtype=torch.int64, device=self.device)

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

        self.env_obj_idx = []
        for idx in range(self.num_envs):
            self.env_obj_idx.append(self.env_to_obj[str(idx)])
        self.env_obj_idx = torch.tensor(self.env_obj_idx) # [num_env], convert dict to tensor

        self.k = 2
        self.rgb_h = 64
        self.rgb_w = 64
        self.rgb_buf = deque(maxlen=self.k)
        self.rgb_buf.extend(self.k * [torch.zeros((self.num_envs, 1, self.rgb_h, self.rgb_w), 
                                                  dtype=torch.float32, device=self.device)])

        self.arange_envs = torch.arange(self.num_envs, device=self.device)

        self.pts_target_list = []

        self.transforms_train_dict = dict()
        camera_properties = self.get_camera_properties()
        self.transforms_train_dict['h'] = self.cfg.visual_input.camera_height
        self.transforms_train_dict['w'] = self.cfg.visual_input.camera_width
        self.transforms_train_dict['camera_angle_x'] = camera_properties['horizontal_fov'] * np.pi / 180     # FOV, 90.0 degree -> 1.57
        self.transforms_train_dict['frames'] = []
        self.transform_normdepth2PIL = ToPILImage()


        # evaluate on all envs
        self.num_eval_round = 5
        self.reset_num_count_round = torch.zeros(self.num_obj, dtype=torch.float32, device=self.device)    # count the number of finished rounds
        self.reset_multi_round_cr = torch.zeros(self.num_obj, self.num_eval_round, dtype=torch.float32, device=self.device)
        self.reset_multi_round_AUC = torch.zeros(self.num_obj, self.num_eval_round, self.max_episode_length, dtype=torch.float32, device=self.device)
        self.reset_multi_round_chamfer_dist = torch.zeros(self.num_obj, self.num_eval_round, dtype=torch.float32, device=self.device)
        self.scanned_pc_coord = [[] for _ in range(self.num_envs)]
        # self.scanned_pc_color = [[] for _ in range(self.num_envs)]

        self.save_path = f'./active_reconstruction/scripts/video/eval_gennbv_houses3k'
        os.makedirs(self.save_path, exist_ok=True)

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
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        self.actions = torch.clip(actions, self.clip_pose_idx_low, self.clip_pose_idx_up)

        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if len(env_ids) != 0:
            self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action,
                                                device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]

        self.poses = self.get_pose_from_discrete_action(action=self.actions)

        self.render()

        self.set_state(self.poses)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        obs, rewards, dones, infos = self.post_physics_step()
        return obs, rewards, dones, infos

    def update_obs_buf(self):
        self.pose_buf.extend([self.poses])
        self.rgb_buf.extend([self.rgb_grayscale])

    def update_occ_grid(self):
        pts_target = self.back_projection_fg(downsample_factor=1)   # list of target points, num_env * [n, 3]

        # num_layer lists of (num_valid_pts_idx, 3)
        pts_idx_all = scanned_pts_to_idx_3D(pts_target=pts_target,
                                            range_gt=self.range_gt_scenes,
                                            voxel_size_gt=self.voxel_size_gt_scenes,
                                            map_size=self.grid_size)
 
        for idx in range(self.num_envs):
            if len(pts_idx_all[idx]) == 0:
                continue
            self.scanned_pc_coord[idx].append(pts_target[idx])   # [num_point, 3]

        pose_idx_3D = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(),
                                                range_gt=self.range_gt_scenes,
                                                voxel_size_gt=self.voxel_size_gt_scenes,
                                                map_size=self.grid_size)
        pose_idx_3D_col = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(), # just for collision checking
                                                range_gt=self.range_gt_scenes,
                                                voxel_size_gt=self.voxel_size_gt_col_scenes,
                                                map_size=128,
                                                if_col=True)
        # pose_idx_3D_col = pose_coord_to_idx_3D(poses=self.poses[:, :3].clone(),
        #                                         range_gt=self.range_gt_scenes,
        #                                         voxel_size_gt=self.voxel_size_gt_scenes,
        #                                         map_size=self.grid_size,
        #                                         if_col=True)
        self.poses_idx_col = pose_idx_3D_col.to(torch.int32).clone()

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
            self.scanned_gt_grid + occ_grids * self.grids_gt_scenes,
            max=1, min=0
        )

    def get_step_return(self):
        assert self.cfg.return_visual_observation, \
            "Images should be returned in this environment!"
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        self.post_process_camera_tensor()   # update self.xxx_processed, where xxx among {depth, rgb, seg}
        self.gym.end_access_image_tensors(self.sim)

        self.update_observation()
        self.check_collsion_3D()    # update collision_flag
        self.compute_reward()   # including self.check_termination()

        obs = {
            # [num_env, buffer_size * pose_size]
            "state": torch.stack(tuple(self.pose_buf), dim=1).view(self.num_envs, -1),
            # [num_env, K, H, W]
            "state_rgb": torch.cat(tuple(self.rgb_buf), dim=1).view(self.num_envs, -1),
            # [num_envs, X, Y, Z]
            "grid": self.occ_grids_tri_cls.view(self.num_envs, -1),
        }

        self.eval_all_envs_multi_round()

        env_ids = self.reset_buf.nonzero().flatten()    # [num_env_ids], to be reset
        env_ids_reset = self.arange_envs[self.reset_num_count_round[self.env_to_scene] == self.num_eval_round]
        env_ids = torch.cat([env_ids, env_ids_reset], dim=0)     # [num_env_ids]
        env_ids = torch.unique(env_ids, sorted=False)
        for env_idx in env_ids:
            scene_idx = self.env_to_scene[env_idx]
            round_idx = int(self.reset_num_count_round[scene_idx].item())
            # if round_idx == self.num_eval_round:
            #     continue

            if self.reset_multi_round_chamfer_dist[scene_idx, round_idx-1] == 0.:
                scanned_pc_coord = torch.cat(self.scanned_pc_coord[env_idx], dim=0).unsqueeze(0)   # [N=1, num_point, 2]
                scanned_kd_tree = KDTree(scanned_pc_coord[0].cpu().numpy())

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(torch.cat([self.layout_pc[scene_idx].cpu(), torch.zeros(self.layout_pc[scene_idx].shape[0], 1)], dim=1).numpy())
                # o3d.io.write_point_cloud(f"{self.save_path}/scene_{scene_idx}_round_{round_idx}_gt.pcd", pcd)

                # pcd_scanned = o3d.geometry.PointCloud()
                # pcd_scanned.points = o3d.utility.Vector3dVector(torch.cat([scanned_pc_coord[0].cpu(), torch.zeros(scanned_pc_coord[0].shape[0], 1)], dim=1).numpy())
                # o3d.io.write_point_cloud(f"{self.save_path}/scene_{scene_idx}_round_{round_idx}_scanned.pcd", pcd_scanned)

                distance, _ = scanned_kd_tree.query(self.layout_pc[scene_idx].cpu().numpy())
                self.reset_multi_round_chamfer_dist[scene_idx, round_idx-1] = 100 * np.mean(distance)

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
            self.pose_buf[buf_idx][env_ids] = torch.tensor(self.cfg.normalization.init_pose_buf, 
                                                           dtype=torch.float, device=self.device, requires_grad=False).repeat(len(env_ids), 1)
            self.reward_ratio_buf[buf_idx][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]
        for buf_idx in range(self.k):
            self.rgb_buf[buf_idx][env_ids] = torch.zeros((self.num_envs, 1, self.rgb_h, self.rgb_w), device=self.device)[env_ids].to(torch.float32)

        if self.buffer_size == 1:
            self.reward_ratio_buf[1][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]

        # Reset scanned point cloud
        for env_idx in env_ids:
            self.scanned_pc_coord[env_idx] = []

        # Reset actions
        self.actions[env_ids] = torch.tensor(self.cfg.normalization.init_action,
                                            device=device).repeat(len(env_ids), 1)

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

    def eval_all_envs_multi_round(self):
        """ Evaluate on all envs + average CR over 10 rounds. """
        for env_idx in range(self.num_envs):
            scene_idx = self.env_to_scene[env_idx]
            round_idx = int(self.reset_num_count_round[scene_idx].item())
            if self.reset_num_count_round[scene_idx] == self.num_eval_round or self.cur_episode_length[env_idx] == 0 or \
                round_idx >= self.num_eval_round or self.reset_buf[env_idx] == 0:
                continue

            self.reset_multi_round_cr[scene_idx, round_idx] = self.reward_ratio_buf[-1][env_idx]
            self.reset_num_count_round[scene_idx] += 1

        process = self.reset_num_count_round.sum()
        print(int(process.item()))

        if process == self.num_obj * self.num_eval_round:
            torch.save(self.reset_multi_round_cr, os.path.join(self.save_path, "reset_multi_round_cr.pt"))  # [num_env, num_round]
            torch.save(self.reset_multi_round_chamfer_dist, os.path.join(self.save_path, "reset_multi_round_chamfer_dist.pt"))  # [num_env, num_round]
            torch.save(self.reset_multi_round_AUC, os.path.join(self.save_path, "reset_multi_round_AUC.pt"))    # [num_env, num_round, max_episode_length=50]

            # print("All CR: ", self.reset_multi_round_cr)
            print("*"*50)
            mean_auc = torch.zeros(self.num_obj, self.num_eval_round, dtype=torch.float32, device=self.device)
            for step_idx in range(self.max_episode_length):
                mean_auc += self.reset_multi_round_AUC[:, :, step_idx] * (self.max_episode_length - step_idx)
            mean_auc /= self.max_episode_length
            print("[AUC] Average All: ", mean_auc.mean(dim=(0,1)))
            print("*"*50)
            print("[CR] Average All: ", self.reset_multi_round_cr.mean(dim=(0,1)))
            print("*"*50)
            print("[Comp.] Average All: ", self.reset_multi_round_chamfer_dist.mean(dim=(0,1)))
            print("*"*50)
            print("Done")
            exit()

    def check_collsion_3D(self):
        """ Check 3D collision. 
        We use occupancy collision to replace rigid body collision 
        because we found Isaac Gym cannot correctly support correct convex decomposition
        for complex collision meshes. """

        # # collision by rigid body (occupancy)
        # collision_rigid = torch.any(torch.norm(self.contact_forces[:,\
        #                                     self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.collision_flag = collision_rigid

        # collision by occupancy
        self.collision_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cur_pose_idx = self.poses_idx_col.to(torch.long)
        no_collision_flag = cur_pose_idx.sum(dim=1) == -3    # no collision
        for env_idx in range(self.num_envs):
            if not no_collision_flag[env_idx]:
                self.collision_flag[env_idx] = self.grids_gt_col_scenes[env_idx, cur_pose_idx[env_idx, 0], \
                    cur_pose_idx[env_idx, 1], cur_pose_idx[env_idx, 2]] == 1.0

    def check_termination(self):
        """ Check if environments need to be reset
        Termination conditions:
            1. collision
            2. steps == max_episode_length
            3. coverage ratio threshold
        """
        # collision
        self.reset_buf = self.collision_flag.clone()

        # max_step
        if self.cfg.termination.max_step_done:
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)    # [num_envs]
            self.reset_buf |= self.time_out_buf

        # coverage ratio > threshold
        last_ratio = self.reward_ratio_buf[-1]
        self.reset_buf |= (last_ratio > self.ratio_threshold_term)

    def get_pose_from_discrete_action(self, action):
        actions = action * self.action_unit + self.action_low_world
        return actions

    def back_projection_fg(self, downsample_factor=1, visualize=False):
        """ Back-projection of depth maps to 3D points in world coordinate
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
        layout_coverage = self.scanned_gt_grid.sum(dim=(1, 2, 3)) / self.num_valid_voxel_gt_scenes
        self.reward_ratio_buf.extend([layout_coverage.clone()])

        rew_coverage = self.reward_ratio_buf[-1] - self.reward_ratio_buf[-2]

        env_ids = torch.arange(self.num_envs, device=self.device)
        scene_ids = self.env_to_scene[env_ids]
        round_ids = self.reset_num_count_round[scene_ids].long()
        mask_round = round_ids < self.num_eval_round

        scene_ids = scene_ids[mask_round]
        env_ids = env_ids[mask_round]
        round_ids = round_ids[mask_round]
        self.reset_multi_round_AUC[scene_ids, round_ids, self.cur_episode_length[env_ids].long()] = rew_coverage[mask_round].clone()

        return rew_coverage
