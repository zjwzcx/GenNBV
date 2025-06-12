"""
This environment is based on drone env, while we ignore the drone control problem by directly setting the state of drone
"""
import os
from collections import deque
import isaacgym
from gym.spaces import Box, Dict
from isaacgym import *
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from PIL import Image as im
import torch

from gennbv.utils import getURDFParameter
from torchvision.transforms.functional import rgb_to_grayscale
from legged_gym.env.base.drone_visual_input import DroneVisualInputEnv
from legged_gym import OPEN_ROBOT_ROOT_DIR
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry


class Env_Train_Base(DroneVisualInputEnv):
    DEPTH_SENSE_DIST = -50
    CAMERA_ANGLE = 0
    OBJECT_SEGMENTATION_ID = 255
    PLANE_SEGMENTATION_ID = 0

    def _create_envs(self):
        if self.cfg.return_visual_observation:
            assert self.cfg.env.num_envs <= 1024, \
                "Please set num_envs <= 1024, since more envs may make the GPU broken"
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=OPEN_ROBOT_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.collapse_fixed_joints = False
        # asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # urdf parameters
        self.KF = getURDFParameter(asset_path, "kf")
        self.KM = getURDFParameter(asset_path, "km")

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)   # ['base_link', 'center_of_mass_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link']
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)   # 6
        self.num_dofs = len(self.dof_names)
        prop_names = [s for s in body_names if self.cfg.asset.prop_name in s]   # ['prop0_link', 'prop1_link', 'prop2_link', 'prop3_link']
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # self.gym.set_actor_scale(env_handle, actor_handle, self.cfg.asset.scale)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self._additional_create(env_handle, i)

        self.prop_indices = torch.zeros(len(prop_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(prop_names)):
            self.prop_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], prop_names[i]
            )

        # NOTE: four contact sensors separately on four rotors to check collision: ['prop0_link', 'prop1_link', 'prop2_link', 'prop3_link', 'base_link']
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.add_camera_to_actors()

    def _create_terrain(self):
        self._create_ground_plane()

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.max_episode_length = self.cfg.max_episode_length   # step

    def _init_load(self):
        # self.grid_gt = torch.load(SAVE_PATH_GT_GRID).to(self.device)   # [X, Y, Z, 4]
        # self.num_valid_voxel_gt = self.grid_gt[..., 3].sum()
        # self.pcd_range = pcd_maxmin(self.grid_gt, print_range=False)
        # self.grid_backproj = None
        pass

    def _init_buffers(self):
        """
        Some buffers are useless!
        """

        self._init_load()

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

        self.prop_contact_forces = self.contact_forces[:, self.prop_indices, :]
        self.prop_lin_vel = self.rigid_state[:, self.prop_indices, 7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        # actions: current actions, last_actions: action at last time
        init_action = self.cfg.init_action
        self.actions = torch.tensor(init_action, dtype=torch.float, device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        self.last_actions = self.actions.clone()
        self.action_size = self.actions.shape[1]


        self.last_root_vel = torch.zeros_like(self.root_states[::self.skip, 7:13])

        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.prop_air_time = torch.zeros(
            self.num_envs, self.prop_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.prop_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.last_prop_contact_forces = torch.zeros(
            self.num_envs, len(self.prop_indices), 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_prop_lin_vel = torch.zeros(
            self.num_envs, len(self.prop_indices), 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # placeholder for compatibility, useless
        self.feet_indices = self.prop_indices
        self.last_dof_pos = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = self.dof_pos = self.dof_vel = self.last_dof_acc = self.last_dof_vel = self.last_dof_pos
        self.feet_air_time = torch.zeros(
            self.num_envs, self.prop_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_feet_contact_forces = self.last_feet_lin_vel = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # =============== additional create, copied from LeggedVisualEnv =================
        # additional stat
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


        # NOTE: create buffers for active reconstruction
        self.action_low_bound = torch.Tensor(self.cfg.normalization.clip_actions_low).to(self.device)
        self.action_up_bound = torch.Tensor(self.cfg.normalization.clip_actions_up).to(self.device)

        self.buffer_size = self.cfg.visual_input.stack
        self.obs_buf = deque(maxlen=self.buffer_size)
        self.action_buf = deque(maxlen=self.buffer_size)

        # initialize buffers with zero_tensor
        self.obs_buf.extend(self.buffer_size * [torch.zeros((self.num_envs, self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width), device=self.device)])
        self.action_buf.extend(self.buffer_size * [self.actions.clone()])
        self.reward_ratio_buf = deque(maxlen=max(self.buffer_size, 2))   # surface coverage ratip
        self.reward_ratio_buf.extend(max(self.buffer_size, 2) * [torch.zeros(self.num_envs, device=self.device)])

        self.collision_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._init_buffers_visual()

    def _init_buffers_visual(self):
        if not self.cfg.return_visual_observation:
            return

        # # NOTE: save current visual observations
        # self.rgb_cam_tensors = []
        # for i in range(self.num_envs):
        #     im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i],
        #                                               gymapi.IMAGE_COLOR)
        #     torch_cam_tensor = gymtorch.wrap_tensor(im)
        #     self.rgb_cam_tensors.append(torch_cam_tensor[..., :3])

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

    def _additional_create(self, env_handle, env_index):
        assert self.cfg.return_visual_observation, "visual observation should be returned!"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments  # default: True
        asset_options.disable_gravity = True

        # # blender, lego_car
        # asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        # asset = self.gym.load_asset(self.sim, asset_root, "blender_lego_car.urdf", asset_options)
        # scale_factor = 1.0

        # # blender, drums
        # asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        # asset = self.gym.load_asset(self.sim, asset_root, "blender_drums.urdf", asset_options)
        # scale_factor = 1.0

        # # blender, holed_cube
        # asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        # asset = self.gym.load_asset(self.sim, asset_root, "blender_holed_cube.urdf", asset_options)
        # scale_factor = 0.1

        # # libertystatue
        # asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        # asset = self.gym.load_asset(self.sim, asset_root, "libertystatue.urdf", asset_options)
        # scale_factor = 3.0

        # # lego_city (original version)
        # asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        # asset = self.gym.load_asset(self.sim, asset_root, "lego_city.urdf", asset_options)
        # scale_factor = 0.006

        # lego_city (large-scale)
        asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "objects")
        asset = self.gym.load_asset(self.sim, asset_root, "lego_city_large.urdf", asset_options)
        scale_factor = 0.02

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.env_origins[env_index][0], self.env_origins[env_index][1], self.env_origins[env_index][2])  # place the box at center point
        pose.r = gymapi.Quat(0, 0, 0, 1)
        ahandle = self.gym.create_actor(env_handle, asset, pose, None, env_index, 0)

        # scale up/down
        self.gym.set_actor_scale(env_handle, ahandle, scale_factor)
        self.gym.set_rigid_body_segmentation_id(env_handle, ahandle, 0, self.OBJECT_SEGMENTATION_ID)

        # set color
        # color = gymapi.Vec3(1, 0.8, 0.4)
        # self.gym.set_rigid_body_color(env_handle, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        self.additional_actors[env_index] = [ahandle]

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        plane_params.segmentation_id = self.PLANE_SEGMENTATION_ID
        self.gym.add_ground(self.sim, plane_params)

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.max_episode_length = self.cfg.max_episode_length   # step

    def add_camera_to_actors(self):
        """
        Copied from legged_visual_input with slight modification on the camera pose (Z-axis + 0.1m)
        """
        if not self.cfg.return_visual_observation:
            return

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.cfg.visual_input.camera_width
        camera_properties.height = self.cfg.visual_input.camera_height
        camera_properties.far_plane = self.cfg.visual_input.far_plane
        camera_properties.near_plane = self.cfg.visual_input.near_plane
        camera_properties.horizontal_fov = self.cfg.visual_input.horizontal_fov
        camera_properties.supersampling_horizontal = self.cfg.visual_input.supersampling_horizontal
        camera_properties.supersampling_vertical = self.cfg.visual_input.supersampling_vertical
        camera_properties.enable_tensors = True

        for i in range(len(self.envs)):
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_offset = gymapi.Vec3(0, 0, 0.1)  # NOTE: robot-mounted camera is 0.1m higher than robot
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.CAMERA_ANGLE))
            actor_handle = self.actor_handles[i]
            body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], actor_handle, 0)

            self.gym.attach_camera_to_body(
                cam_handle, self.envs[i], body_handle, gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_TRANSFORM
            )
            self.camera_handles.append(cam_handle)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name] # [num_envs]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            pass
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        self.check_termination()

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def post_physics_step_reset_vis_obs(self):
        """ special degisn for rewards about visual observation.
            check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[::self.skip, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 7:10]) # vec from world wcoord to actor coord
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # NOTE: compute 'get step return' in advance for compute_reward
        self.check_termination()
        obs = self.get_step_return()[0]    # save visual observations at initial actions, check termination(), compute observations() and self.reward()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[::self.skip, 7:13]
        self.last_feet_contact_forces[:] = self.contact_forces[:, self.feet_indices, :]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_feet_lin_vel[:] = self.rigid_state[:, self.feet_indices, 7:10]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[::self.skip, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.skip, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        obs, rewards, dones, infos = self.get_step_return()  # include self.compute_observations(), self.compute_reward() and self.reset_idx()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[::self.skip, 7:13]
        self.last_feet_contact_forces[:] = self.contact_forces[:, self.feet_indices, :]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_feet_lin_vel[:] = self.rigid_state[:, self.feet_indices, 7:10]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return obs, rewards, dones, infos

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # reset all envs, for example, self.last_actions of all envs will be reset to torch.ones(6)

        self.actions = torch.clip(self.actions, self.action_low_bound, self.action_up_bound).to(self.device)

        self.render()

        # Set two times, an interesting "feature"/bug of IsaacGym
        self.set_state(self.actions)
        self.gym.simulate(self.sim)
        self.set_state(self.actions)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        obs = self.post_physics_step_reset_vis_obs()

        return obs

    def step(self, actions):
        """
        Set the position (x, y, z) and orientation (r, p, y) for the camera. actions.shape: [1, 6]
        """
        self.actions = torch.clip(actions, self.action_low_bound, self.action_up_bound).to(self.device)

        env_ids = [idx for idx in range(self.num_envs) if self.episode_length_buf[idx] == 0]
        if len(env_ids) != 0:
            self.actions[env_ids] = torch.tensor(self.cfg.init_action, dtype=torch.float, device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]

        self.render()

        # Set two times, an interesting "feature"/bug of IsaacGym
        self.set_state(self.actions)
        self.gym.simulate(self.sim)
        self.set_state(self.actions)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        obs, rewards, dones, infos = self.post_physics_step()    # include self.get step_return(); compute_observations() in get step_return()

        return obs, rewards, dones, infos

    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        rgb_images = torch.stack(self.rgb_cam_tensors)[..., :3].permute(0, 3, 1, 2) # [num_env, 3, H, W]
        rgb_images = torch.nn.functional.interpolate(rgb_images, size=(self.rgb_h, self.rgb_w))
        self.rgb_grayscale = rgb_to_grayscale(rgb_images).to(torch.float32)   # [num_env, 1, H, W], weighted sum of RGB
        assert self.rgb_h == rgb_images.shape[2], self.rgb_w == rgb_images.shape[3]

        depth_images = torch.stack(self.depth_cam_tensors)
        depth_images = torch.nan_to_num(depth_images, neginf=0)
        depth_images = torch.clamp(depth_images, min=self.DEPTH_SENSE_DIST)     # depth min: -8
        depth_images = abs(depth_images)    # depth range: [0.0, 8.0]
        if not self.cfg.visual_input.normalization:
            depth_images = depth_images * 255
        self.depth_processed = depth_images # [num_env, H, W]

        seg_images = torch.stack(self.seg_cam_tensors)
        seg_images = torch.nan_to_num(seg_images, neginf=0)
        if not self.cfg.visual_input.normalization:
            seg_images = seg_images * 255
        self.seg_processed = seg_images

    def get_step_return(self):
        # render sensors and refresh camera tensors
        assert self.cfg.return_visual_observation, "Images should be returned in this environment!"
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.post_process_camera_tensor()   # update self.xxx_processed
        self.compute_observations()         # stack self.xxx_processed into self.xxx_buf

        obs = {
                "state": torch.stack(tuple(self.action_buf), dim=1).view(self.num_envs, -1),  # [num_env, buffer_size*action_size]
                "image": torch.stack(tuple(self.obs_buf), dim=1), # [num_env, H, W, buffer_size]
                # "seg_image": torch.stack(tuple(self.seg_buf), dim=1),  # [num_env, buffer_size, H, W]
        }
        # self.debug_save_image(obs)

        self.gym.end_access_image_tensors(self.sim)

        self.compute_reward()
        # self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # self.reset_buf is updated in check termination()
        self.reset_idx(env_ids)

        rewards, dones, infos = self.rew_buf, self.reset_buf, self.extras
        self.update_extra_episode_info(rewards=rewards, dones=dones)

        assert not self.cfg.return_privileged_observation
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
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._additional_reset(env_ids)

        # reset pedestrians
        if hasattr(self, "num_pedestrians") and self.num_pedestrians:
            self.reset_pedestrians(env_ids)

        # reset buffers
        for buf_idx in range(self.buffer_size):
            self.obs_buf[buf_idx][env_ids] = torch.zeros((self.num_envs, self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width), device=self.device)[env_ids]
            self.action_buf[buf_idx][env_ids] = torch.tensor(self.cfg.init_action, dtype=torch.float, device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]
            self.reward_ratio_buf[buf_idx][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]
        if self.buffer_size == 1:
            self.reward_ratio_buf[1][env_ids] = torch.zeros(self.num_envs, device=self.device)[env_ids]

        self.actions[env_ids] = torch.tensor(self.cfg.init_action, dtype=torch.float, device=self.device, requires_grad=False).repeat(self.num_envs, 1)[env_ids]
        self.last_actions[env_ids] = self.actions[env_ids].clone()
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        if self.grid_backproj is not None:
            for env_idx in env_ids:
                self.grid_backproj[env_idx] = None

        # fill extras. env_ids: terminated envs
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

    def update_extra_episode_info(self, rewards, dones):
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().detach().numpy().tolist())
        self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().detach().numpy().tolist())
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0

        self.extras["episode"]["episode_reward"] = np.mean(self.rewbuffer) if len(self.rewbuffer) > 0 else 0.
        self.extras["episode"]["episode_length"] = np.mean(self.lenbuffer) if len(self.lenbuffer) > 0 else 0.

    def _reset_dofs(self, env_ids):
        pass

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
        if self.cfg.termination.max_step_done:  # <-
            self.time_out_buf = (self.episode_length_buf >= self.max_episode_length)
            self.reset_buf |= self.time_out_buf # in-place OR

        # coverage ratio > threshold
        ratio_threshold = 0.95
        last_ratio = self.reward_ratio_buf[-1]
        self.reset_buf |= (last_ratio > ratio_threshold)

    def get_pose_from_discrete_action(self, action):
        actions = action * self.action_unit + self.action_low_world
        return actions

    def compute_observations(self):
        """
        stack current visual observations/actions into obs_buf/action_buf.

        buffer: deque(maxlen=self.buffer_size).

        depth_processed: [num_env, H, W]
        rgb_processed: [num_env, H, W, 3]
        seg_prosessed: [num_env, H, W]
        actions: [num_env, action_size]
        """
        self.obs_buf.extend([self.depth_processed])
        self.action_buf.extend([self.actions])

        # if self.add_noise:    # TODO
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def set_state(self, actions):
        # implement discrete action space as another choice, this can be achieved by using an env wrapper
        if self.cfg.position_use_polar_coordinates: # (r, \theta, \phi) -> (x, y, z)
            x = actions[..., 0:1] * torch.cos(actions[..., 2:3]) * torch.cos(actions[..., 1:2])
            y = actions[..., 0:1] * torch.cos(actions[..., 2:3]) * torch.sin(actions[..., 1:2])
            z = actions[..., 0:1] * torch.sin(actions[..., 2:3])
            position = torch.concat([x, y, z], dim=-1)
        else:
            position = actions[..., 0:3]

        if self.cfg.direction_use_vector:   # convert direction (3:x, 4:y, 5:z) vector to rpy
            length = torch.norm(actions[..., 3:], dim=-1, keepdim=True)
            phi = -torch.arcsin(actions[..., -1:] / length)
            project_len_on_x_y = torch.cos(phi) * length
            theta = torch.where(
                actions[..., 4:5] > 0, torch.arccos(actions[..., 3:4] / project_len_on_x_y),
                torch.pi * 2 - torch.arccos(actions[..., 3:4] / project_len_on_x_y)
            )
            heading = torch.concat([torch.zeros_like(actions[..., 4:5]), phi, theta], dim=-1)
        else:
            heading = actions[..., 3:]

        # set position
        self.root_states[::self.skip, 0:3] = position + self.env_origins
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        # set rpy
        quat = quat_from_euler_xyz(heading[..., 0], heading[..., 1], heading[..., 2])
        self.root_states[::self.skip, 3:7] = quat
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def sample_actions(self, action=None):
        if action is None:
            actions = torch.ones(self.num_envs, self.num_actions, device=self.device)
        else:
            if len(action) == 6:    # list, for data generator scripts
                actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
                actions[:] = torch.tensor(action, device=self.device)     # all distributed envs share the same initial action (Tensor)
            elif len(action.shape) == 1:
                assert len(action) == 6, "when specify actions, the dim should be 6"
                actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
                actions[:] = action     # all distributed envs share the same initial action (Tensor)
            else:
                assert action.shape[-1] == 6
                actions = action        # assign every initial action for each env
        return actions

    def update_observation_space(self):
        """ update observation and action space
        """
        if not self.cfg.return_visual_observation:
            return

        H, W = self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width
        action_low_bound = self.action_low_bound.cpu().numpy()
        action_up_bound = self.action_up_bound.cpu().numpy()

        # choose next **one** action
        self.action_space = Box(low=action_low_bound, high=action_up_bound, shape=(self.action_size, ), dtype=np.float32)


        action_low_bound = np.tile(action_low_bound, self.buffer_size)
        action_up_bound = np.tile(action_up_bound, self.buffer_size)

        self.observation_space = Dict(
            {
                "state": Box(low=action_low_bound, high=action_up_bound, shape=(self.buffer_size * self.action_size, ), dtype=np.float32),   # view actions as state
                "image": Box(low=0, high=255, shape=(self.buffer_size, H, W), dtype=np.float32),   # stacked depth maps

                # "image": Box(low=0, high=255, shape=(self.buffer_size, H, W, 3), dtype=np.float64),   # RGB images
                # "depth_image": Box(low=0, high=255, shape=(self.buffer_size, H, W), dtype=np.float64),
                # "seg_image": Box(low=0, high=255, shape=(self.buffer_size, H, W), dtype=np.float32)
            }
        )

    def get_camera_properties(self):
        """
        All env.envs share the same camera_properties
        """
        return self.cfg.visual_input.__dict__

    def get_camera_transform(self):
        assert len(self.camera_handles) == self.num_envs, "We assume the number of envs equals to th number of cameras"
        ret = []
        for k, handle in enumerate(self.camera_handles):
            to_add = self.gym.get_camera_transform(self.sim, self.envs[k], handle)
            # if return_pos_quat:
            #     to_add = {"pos": to_add.p, "quat": to_add.r}
            ret.append(to_add)
        return np.array(ret)

    def get_camera_view_matrix(self):
        """
        return Extrinsics.t() instead of Extrinsics. E * P = P * E.t()
        """
        assert len(self.camera_handles) == self.num_envs, "We assume the number of envs equals to th number of cameras"
        ret = []
        for k, handle in enumerate(self.camera_handles):
            ret.append(self.gym.get_camera_view_matrix(self.sim, self.envs[k], handle))
        return np.array(ret)

    def get_camera_intrinsics(self):
        H, W = self.cfg.visual_input.camera_height, self.cfg.visual_input.camera_width
        FOV_x = self.cfg.visual_input.horizontal_fov / 180 * np.pi
        FOV_y = FOV_x * H / W   # Vertical field of view is calculated from height to width ratio

        focal_x = 0.5 * W / np.tan(0.5 * FOV_x)
        # focal_y = focal_x
        # focal_y = 0.5 * H / np.tan(0.5 * FOV_x)
        focal_y = 0.5 * H / np.tan(0.5 * FOV_y)
        cx, cy = W / 2, H / 2
        intrinsics = torch.tensor([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]]).float()

        # focal = 0.5 * W / np.tan(0.5 * FOV)
        # cx, cy = W / 2, H / 2
        # intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]]).float()

        return intrinsics

    def debug_save_image(self, o):
        save_path = self.cfg.debug_save_path if self.cfg.debug_save_path != None else './debug_save_image'
        os.makedirs(save_path, exist_ok=True)

        if self.cfg.debug_save_image_tensor and self.cfg.return_visual_observation:
            for i in range(self.num_envs):
                # write tensor to image
                fname = os.path.join(save_path, "cam-frame%d-env%d.png" % (self.common_step_counter, i))
                cam_img = o["image"][i].clone()
                if self.cfg.visual_input.normalization:
                    cam_img *= 255
                cam_img = cam_img.cpu().detach().numpy()
                image = im.fromarray(
                    cam_img.astype(np.uint8), mode="RGBA")
                image.save(fname)

                # write tensor to image
                fname = os.path.join(save_path, "depth_cam-frame%d-env%d.png" % (self.common_step_counter, i))
                cam_img = o["depth_image"][i].clone()
                if self.cfg.visual_input.normalization:
                    cam_img *= 255
                cam_img = cam_img.cpu().detach().numpy()
                image = im.fromarray(cam_img.astype(np.uint8), mode="L")
                image.save(fname)

                # write tensor to image
                fname = os.path.join(save_path, "seg_cam-frame%d-env%d.png" % (self.common_step_counter, i))
                cam_img = o["seg_image"][i].clone()
                if self.cfg.visual_input.normalization:
                    cam_img *= 255
                cam_img = cam_img.cpu().detach().numpy()
                image = im.fromarray(cam_img.astype(np.uint8), mode="L")
                image.save(fname)

    @property
    def cam_type(self):
        return self.cfg.visual_input.type

    def render(self, sync_frame_time=True):
        if self.cfg.camera.first_view_camera and self.viewer is not None:
            position = self.base_pos[self.cfg.camera.env_to_track]
            r, p, y = isaacgym.torch_utils.get_euler_xyz(self.base_quat)
            camera_x = -torch.cos(y) * self.cfg.camera.dist + position[0]
            camera_y = -torch.sin(y) * self.cfg.camera.dist + position[1]
            height = self.cfg.camera.height + position[-1]
            self.set_camera(
                (camera_x[self.cfg.camera.env_to_track], camera_y[self.cfg.camera.env_to_track], height), (position)
            )
        events = {i.action: i.value > 0 for i in self.gym.query_viewer_action_events(self.viewer)}
        for evt in self.keyboard_press_event.keys():
            if evt in events and events[evt]:
                self.keyboard_press_event[evt] = True
            else:
                self.keyboard_press_event[evt] = False
        return super().render()

    def _reward_collision(self):
        """ Penalize collisions on selected bodies.
        contact_forces: [num_env, num_bodies, 3], bodies: ('base_link', 'center_of_mass_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link') + .obj
        penalised_contact_indices: indices of ('base_link', 'prop0_link', 'prop1_link', 'prop2_link', 'prop3_link')
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) # [num_env], "1": collide

    def _reward_termination(self):
        """ Terminal reward / penalty """
        return self.reset_buf * ~self.time_out_buf


if __name__ == "__main__":
    import gennbv

    args = get_args()
    args.headless = False
    args.sim_device = "cuda"

    env_cfg, _ = task_registry.get_cfgs(name="reconstruction_drone")
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.position_use_polar_coordinates = False
    env_cfg.direction_use_vector = False
    env_cfg.camera.first_view_camera = True
    env, _ = task_registry.make_env("reconstruction_drone", args, env_cfg)
    env.debug_viz = True
    env.reset()
    print(env.get_camera_properties())

    blender2opencv = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    while True:
        o, r, d, i = env.step(env.sample_actions([0., 0., 8., 0., 90 / 180 * np.pi, 0.]))
        obs = {"state": o["state"][0].cpu().detach().numpy(),
               "image": o["image"][0].cpu().detach().numpy(),
               # "depth_image": o["depth_image"][0].cpu().detach().numpy(),
               # "seg_image": o["seg_image"][0].cpu().detach().numpy(),
               }
        assert env.observation_space.contains(obs), "OBS shape error!!"

        # extrinsics = torch.FloatTensor(env.get_camera_view_matrix().astype(float)) # [num_envs, 4, 4]
        # c2ws = []
        # for idx in range(env_cfg.env.num_envs):
        #     c2w = torch.linalg.inv(extrinsics[idx]).t() @ blender2opencv
        #     c2ws.append(c2w)
        # c2w = torch.stack(c2ws, dim=0)    # [N, 4, 4]
        # pass
