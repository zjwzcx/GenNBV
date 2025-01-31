from collections import deque

from isaacgym import *
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.env.base.drone_visual_input import DroneVisualInputEnv
from legged_gym import *
from legged_gym import OPEN_ROBOT_ROOT_DIR
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from active_reconstruction.utils import getURDFParameter
import torch
from legged_gym.utils.math import wrap_to_pi


class ActiveReconstructionDroneEnv(DroneVisualInputEnv):
    """
    This environment is modified from LeggedLocomotionEnv by simply replacing the quadrupedal robot with a drone.
    """
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

    def _init_buffers(self):
        """ 
        Some buffers are useless!
        """
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
        self.base_quat = self.root_states[::self.skip, 3:7]
        self.base_pos = self.root_states[::self.skip, 0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_state).view(
            self.num_envs, -1, 13
        )  # shape: num_envs, num_bodies, 13

        self.prop_contact_forces = self.contact_forces[:, self.prop_indices, :]
        self.prop_lin_vel = self.rigid_state[:, self.prop_indices, 7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
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
        self.x_dist_buffer = deque(maxlen=100)
        self.y_dist_buffer = deque(maxlen=100)
        self.z_dist_buffer = deque(maxlen=100)

        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not self.cfg.return_visual_observation:
            return
        self.cam_tensors = []
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], self.cam_type)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.cam_tensors.append(torch_cam_tensor)
        if self.cam_type == gymapi.IMAGE_COLOR:
            channel = 4
            self.visual_obs_buf = torch.zeros(
                self.num_envs,
                self.cfg.visual_input.stack,
                self.cfg.visual_input.camera_height,
                self.cfg.visual_input.camera_width,
                channel,
                dtype=torch.float64,
                device=self.device,
                requires_grad=False
            )
        elif self.cam_type == gymapi.IMAGE_DEPTH:
            # Note: Depth image only contain one channel
            self.visual_obs_buf = torch.zeros(
                self.num_envs,
                self.cfg.visual_input.stack,
                self.cfg.visual_input.camera_height,
                self.cfg.visual_input.camera_width,
                dtype=torch.float64,
                device=self.device,
                requires_grad=False
            )
        else:
            raise ValueError("please indictae the channel of {} image".format(self.cam_type))

    def reset_idx(self, env_ids):
        # z_dist = np.abs((self.base_pos[env_ids][..., 2] - self.env_origins[:, 2][env_ids]).cpu().detach().numpy())
        # self.z_dist_buffer.extend(z_dist)
        return super(ActiveReconstructionDroneEnv, self).reset_idx(env_ids)

    def update_extra_episode_info(self, rewards, dones):
        super(ActiveReconstructionDroneEnv, self).update_extra_episode_info(rewards, dones)
        # self.extras["episode"]["episode_moving_dist_z"] = np.mean(self.z_dist_buffer) if len(self.z_dist_buffer) > 0 else 0.

    def _reset_dofs(self, env_ids):
        pass

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # clip and scale the action, the RPM will be bounded to [0, clip_actions * scale_actions]
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.actions = self.actions * self.cfg.normalization.scale_actions
        self.actions += self.cfg.normalization.base_rpm
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self._additional_step()
            self.process_actions(self.actions)
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        o, privileged_o, r, d, i = self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        # render sensors and refresh camera tensors
        if self.cfg.return_visual_observation:
            sim = self.sim
            self.gym.step_graphics(sim)
            self.gym.render_all_camera_sensors(sim)
            self.gym.start_access_image_tensors(sim)
            self.post_process_camera_tensor()
            o = {"state": o, "image": self.visual_obs_buf}
            self.debug_save_image(o)
            self.gym.end_access_image_tensors(sim)
        self.update_extra_episode_info(rewards=r, dones=d)
        if self.cfg.return_privileged_observation:
            return {"obs": o, "privileged_obs": privileged_o}, r, d, i
        else:
            return o, r, d, i

    def compute_observations(self):
        """
        Computes observations
        """
        r, p, y = torch_utils.get_euler_xyz(self.base_quat)
        self.obs_buf = torch.cat(
            (
                self.base_pos[..., -1:],  # z
                torch.unsqueeze(wrap_to_pi(r), -1),  # roll
                torch.unsqueeze(wrap_to_pi(p), -1),  # pitch
                torch.unsqueeze(wrap_to_pi(y), -1),  # yaw
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                # no Dof in Drone
                # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                # self.dof_vel * self.obs_scales.dof_vel,
                self.actions  # last action
            ),
            dim=-1
        )
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[::self.skip, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def process_actions(self, actions):
        """
        Take rpm as actions, converting rpm to forces
        LQY:
            1) ground effect
            2) donewash
        """
        self._physics(actions)

    def _physics(self, rpm):
        """
        convert rpm to force
        """
        forces = torch.pow(rpm, 2) * self.KF
        torques = torch.pow(rpm, 2) * self.KM
        z_torque = (-torques[..., 0] + torques[..., 1] - torques[..., 2] + torques[..., 3])

        force_to_set = torch.zeros(
            (self.num_envs, self.num_bodies + len(self.additional_actors[0]), 3), device=self.device, dtype=torch.float
        )
        torques_to_set = torch.zeros(
            (self.num_envs, self.num_bodies + len(self.additional_actors[0]), 3), device=self.device, dtype=torch.float
        )
        force_to_set[..., 2:6, -1] = forces
        torques_to_set[..., :2, -1] = torch.unsqueeze(z_torque, -1)
        ret = self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(force_to_set), gymtorch.unwrap_tensor(torques_to_set),
            gymapi.CoordinateSpace.LOCAL_SPACE
        )
        assert ret, "Fail to set forces/torques"

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        # noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12:16] = 0.  # previous actions
        return noise_vec

    def sample_actions(self, action=None):
        if action is None:
            actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        else:
            assert len(action) == 4, "when specify actions, the dim should be 4"
            actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
            actions[:] = torch.Tensor(action)
        return actions

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > .0, dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf


if __name__ == "__main__":

    args = get_args()
    args.headless = True
    args.sim_device = "cuda"
    env_cfg, _ = task_registry.get_cfgs(name="drone")
    env_cfg.env.num_envs = 10
    env_cfg.camera.first_view_camera = True
    env_cfg.return_visual_observation = True
    env, _ = task_registry.make_env("drone", args, env_cfg)
    env.debug_viz = True
    env.reset()
    while True:
        o, r, d, i = env.step(env.sample_actions())
        obs = {"state": o["state"][0].cpu().detach().numpy(), "image": o["image"][0].cpu().detach().numpy()}
        assert env.observation_space.contains(obs), "OBS shape error!!"
