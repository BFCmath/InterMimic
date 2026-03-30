from __future__ import annotations

from enum import Enum
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from ... import g1_native_spec
from ...utils import torch_utils
from ...utils.g1_native_dataset import G1NativeMotionDataset, discover_g1_native_sequences
from ...utils.path_utils import resolve_data_path
from .humanoid_g1 import Humanoid_G1


class InterMimicG1Native(Humanoid_G1):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg['env'].get('stateInit', 'Random')
        self._state_init = InterMimicG1Native.StateInit[state_init]
        self._hybrid_init_prob = cfg['env'].get('hybridInitProb', 0.5)
        self.motion_file = cfg['env']['motion_file']
        self.reward_weights = cfg['env']['rewardWeights']
        self.rollout_length = cfg['env']['rolloutLength']
        self.enable_object_tracking = bool(cfg['env'].get('enableObjectTracking', False))
        self.require_object_data = bool(cfg['env'].get('requireObjectData', False))
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        self.data_sub = cfg['env'].get('dataSub', [])
        self._hidden_object_height = -10.0
        self.ref_hoi_obs_size = g1_native_spec.native_observation_size()

        metadata = discover_g1_native_sequences(self.motion_file, self.data_sub)
        self.sequence_names = [info.sequence_name for info in metadata]
        self.object_name = sorted({info.object_name for info in metadata if info.has_object})
        self._has_any_object = any(info.has_object for info in metadata)
        self._num_motions_discovered = len(metadata)
        self.dataset_index = torch.tensor([info.subject_index for info in metadata], dtype=torch.long)
        if self.object_name:
            self.object_id = torch.tensor([
                self.object_name.index(info.object_name) if info.has_object else -1
                for info in metadata
            ], dtype=torch.long)
            self.obj2motion = torch.stack([self.object_id == idx for idx in range(len(self.object_name))], dim=0)
        else:
            self.object_id = torch.full((len(metadata),), -1, dtype=torch.long)
            self.obj2motion = torch.zeros((0, len(metadata)), dtype=torch.bool)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.motion_lib = G1NativeMotionDataset(self.motion_file, allowed_subjects=self.data_sub, device=self.device)
        self.num_motions = self.motion_lib.num_sequences
        self.max_episode_length = self.motion_lib.max_episode_length
        self.dataset_index = self.motion_lib.subject_indices
        self.object_id = self.motion_lib.object_ids
        self.obj2motion = self.motion_lib.obj2motion
        self.object_name = self.motion_lib.object_names

        if self.require_object_data and not bool(self.motion_lib.has_object_sequence.all()):
            raise ValueError('requireObjectData=True, but some native G1 sequences do not contain object data')

        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._track_reset_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self._curr_ref_state = self.motion_lib.get_state(torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                                                         torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        self._build_target_tensors()
        self._log_native_dataset_summary()

    def _log_native_dataset_summary(self):
        motion_lengths = self.max_episode_length.detach().cpu()
        print(
            '[InterMimicG1Native] dataset summary: '
            f'motions={self.num_motions} '
            f'objects={len(self.object_name)} '
            f'has_any_object={self._has_any_object} '
            f'obs_dim={self.get_obs_size()} '
            f'act_dim={self.get_action_size()} '
            f'rollout_length={self.rollout_length}'
        )
        print(
            '[InterMimicG1Native] sequence lengths: '
            f'min={int(motion_lengths.min().item())} '
            f'max={int(motion_lengths.max().item())} '
            f'mean={float(motion_lengths.float().mean().item()):.1f}'
        )
        if self.object_name:
            print(f'[InterMimicG1Native] objects: {self.object_name}')
        if self.data_sub:
            print(f'[InterMimicG1Native] subject filter: {self.data_sub}')

    def _setup_character_props(self, key_bodies):
        self._dof_obs_size = g1_native_spec.G1_DOF_COUNT
        self._num_actions = g1_native_spec.G1_DOF_COUNT
        self._num_actions_hand = 0
        self._num_actions_wrist = 0
        self._num_obs = g1_native_spec.native_observation_size()

    def get_num_amp_obs(self):
        return self.get_obs_size()

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        if self._has_any_object:
            self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _load_target_asset(self):
        asset_root = resolve_data_path('assets', 'objects')
        self._target_asset = []
        for object_name in self.object_name:
            asset_file = object_name + '.urdf'
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.density = self.object_density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = 64
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000
            self._target_asset.append(self.gym.load_asset(self.sim, str(asset_root), asset_file, asset_options))

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, 'humanoid', col_group, col_filter, segmentation_id)
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop['driveMode'] = gymapi.DOF_MODE_POS
            dof_prop['stiffness'] = list(g1_native_spec.PD_STIFFNESS)
            dof_prop['damping'] = list(g1_native_spec.PD_DAMPING)
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        body_names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)
        body_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, humanoid_handle)
        for body_idx, idx_range in enumerate(body_shape_indices):
            name = body_names[body_idx]
            start, count = idx_range.start, idx_range.count
            for si in range(start, start + count):
                sp = shape_props[si]
                if 'right' in name:
                    if 'ankle' in name:
                        sp.filter = 2
                    elif 'knee' in name:
                        sp.filter = 6
                    elif 'hip' in name:
                        sp.filter = 12
                if 'left' in name:
                    if 'ankle' in name:
                        sp.filter = 16
                    elif 'knee' in name:
                        sp.filter = 48
                    elif 'hip' in name:
                        sp.filter = 96
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, shape_props)
        self.humanoid_handles.append(humanoid_handle)

        if self._has_any_object:
            self._build_target(env_id, env_ptr)

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        object_idx = env_id % len(self.object_name)
        target_handle = self.gym.create_actor(
            env_ptr,
            self._target_asset[object_idx],
            default_pose,
            self.object_name[object_idx],
            col_group,
            col_filter,
            segmentation_id,
        )
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for prop in props:
            prop.restitution = 0.05
            prop.friction = 0.6
            prop.rolling_friction = 0.01
            prop.torsion_friction = 0.01
            prop.rest_offset = 0.015 if self.object_name[object_idx] in ('plasticbox', 'trashcan') else 0.002
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)
        self._target_handles.append(target_handle)

    def _build_target_tensors(self):
        if not self._has_any_object:
            self._target_states = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
            self._tar_actor_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            return
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

    def _sample_motion_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._has_any_object:
            return torch.randint(self.num_motions, (env_ids.shape[0],), device=self.device, dtype=torch.long)

        samples = []
        for env_id in env_ids.tolist():
            object_bucket = env_id % len(self.object_name)
            if self.require_object_data or self.enable_object_tracking:
                candidates = torch.where(self.object_id == object_bucket)[0]
            else:
                candidates = torch.where(torch.logical_or(~self.motion_lib.has_object_sequence, self.object_id == object_bucket))[0]
            if candidates.numel() == 0:
                raise RuntimeError(f'No native G1 sequences found for object bucket {object_bucket}')
            samples.append(candidates[torch.randint(candidates.numel(), (1,), device=self.device)].item())
        return torch.tensor(samples, device=self.device, dtype=torch.long)

    def _sample_start_frames(self, sequence_ids: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        if self._state_init == InterMimicG1Native.StateInit.Start:
            return torch.zeros(sequence_ids.shape[0], device=self.device, dtype=torch.long)
        if self._state_init == InterMimicG1Native.StateInit.Hybrid:
            random_mask = torch.bernoulli(torch.full((sequence_ids.shape[0],), self._hybrid_init_prob, device=self.device)).bool()
        else:
            random_mask = torch.ones(sequence_ids.shape[0], device=self.device, dtype=torch.bool)
        starts = torch.zeros(sequence_ids.shape[0], device=self.device, dtype=torch.long)
        for idx, seq_id in enumerate(sequence_ids.tolist()):
            max_len = int(self.max_episode_length[seq_id].item())
            max_start = max(1, max_len - self.rollout_length)
            if random_mask[idx]:
                starts[idx] = torch.randint(max_start, (1,), device=self.device, dtype=torch.long)
        return starts

    def _reset_target(self, env_ids: torch.Tensor):
        if not self._has_any_object:
            return
        ref = self.motion_lib.get_state(self.data_id[env_ids], self.progress_buf[env_ids])
        object_mask = ref['object_mask']
        self._target_states[env_ids, 0:3] = torch.where(
            object_mask > 0.5,
            ref['object_pos'],
            torch.tensor([0.0, 0.0, self._hidden_object_height], device=self.device, dtype=ref['object_pos'].dtype),
        )
        self._target_states[env_ids, 3:7] = torch.where(
            object_mask.repeat(1, 4) > 0.5,
            ref['object_rot'],
            g1_native_spec.identity_quat(object_mask.shape[0], device=self.device),
        )
        self._target_states[env_ids, 7:10] = ref['object_lin_vel'] * object_mask
        self._target_states[env_ids, 10:13] = ref['object_ang_vel'] * object_mask

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = g1_native_spec.normalize_quat(root_rot)
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

    def _reset_actors(self, env_ids):
        sequence_ids = self._sample_motion_ids(env_ids)
        frame_ids = self._sample_start_frames(sequence_ids, env_ids)
        ref = self.motion_lib.get_state(sequence_ids, frame_ids)
        self.data_id[env_ids] = sequence_ids
        self.dataset_id[env_ids] = self.motion_lib.subject_indices[sequence_ids]
        self.progress_buf[env_ids] = frame_ids
        self.start_times[env_ids] = frame_ids
        self.prev_actions[env_ids] = 0.0
        self._track_reset_mask[env_ids] = False
        self._set_env_state(
            env_ids,
            ref['root_pos'],
            ref['root_rot'],
            ref['dof_pos'],
            ref['root_lin_vel'],
            ref['root_ang_vel'],
            ref['dof_vel'],
        )
        self._reset_target(env_ids)

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if not self._has_any_object:
            return
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _update_hist_hoi_obs(self, env_ids=None):
        return

    def _gather_current_state(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            if self._has_any_object:
                object_states = self._target_states
            else:
                object_states = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
        else:
            root_states = self._humanoid_root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            if self._has_any_object:
                object_states = self._target_states[env_ids]
            else:
                object_states = torch.zeros((env_ids.shape[0], 13), device=self.device, dtype=torch.float32)
        return {
            'root_pos': root_states[:, 0:3],
            'root_rot': g1_native_spec.normalize_quat(root_states[:, 3:7]),
            'root_lin_vel': root_states[:, 7:10],
            'root_ang_vel': root_states[:, 10:13],
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'object_pos': object_states[:, 0:3],
            'object_rot': g1_native_spec.normalize_quat(object_states[:, 3:7]),
            'object_lin_vel': object_states[:, 7:10],
            'object_ang_vel': object_states[:, 10:13],
            'object_mask': self.motion_lib.has_object_sequence[self.data_id if env_ids is None else self.data_id[env_ids]].float().unsqueeze(-1),
        }

    def _compute_hoi_observations(self, env_ids=None):
        seq_ids = self.data_id if env_ids is None else self.data_id[env_ids]
        frame_ids = self.progress_buf if env_ids is None else self.progress_buf[env_ids]
        self._curr_ref_state = self.motion_lib.get_state(seq_ids, frame_ids)

    def _build_current_features(self, current_state):
        heading_rot = torch_utils.calc_heading_quat_inv(current_state['root_rot'])
        local_root_lin_vel = quat_rotate(heading_rot, current_state['root_lin_vel'])
        local_root_ang_vel = quat_rotate(heading_rot, current_state['root_ang_vel'])
        object_mask = current_state['object_mask']
        local_object_pos = quat_rotate(heading_rot, current_state['object_pos'] - current_state['root_pos']) * object_mask
        local_object_lin_vel = quat_rotate(heading_rot, current_state['object_lin_vel']) * object_mask
        local_object_ang_vel = quat_rotate(heading_rot, current_state['object_ang_vel']) * object_mask
        local_object_rot = torch_utils.quat_to_tan_norm(quat_mul(heading_rot, current_state['object_rot'])) * object_mask.repeat(1, 6)
        return torch.cat((
            current_state['root_pos'][:, 2:3],
            local_root_lin_vel,
            local_root_ang_vel,
            current_state['dof_pos'],
            current_state['dof_vel'],
            local_object_pos,
            local_object_rot,
            local_object_lin_vel,
            local_object_ang_vel,
            object_mask,
        ), dim=-1)

    def _build_future_features(self, current_state, future_state):
        heading_rot = torch_utils.calc_heading_quat_inv(current_state['root_rot'])
        local_root_pos_delta = quat_rotate(heading_rot, future_state['root_pos'] - current_state['root_pos'])
        root_rot_delta = torch_utils.quat_mul_norm(torch_utils.quat_inverse(current_state['root_rot']), future_state['root_rot'])
        root_rot_delta_obs = torch_utils.quat_to_tan_norm(root_rot_delta)
        local_root_lin_vel_delta = quat_rotate(heading_rot, future_state['root_lin_vel'] - current_state['root_lin_vel'])
        local_root_ang_vel_delta = quat_rotate(heading_rot, future_state['root_ang_vel'] - current_state['root_ang_vel'])
        object_mask = future_state['object_mask']
        local_object_pos_delta = quat_rotate(heading_rot, future_state['object_pos'] - current_state['object_pos']) * object_mask
        object_rot_delta = torch_utils.quat_mul_norm(torch_utils.quat_inverse(current_state['object_rot']), future_state['object_rot'])
        object_rot_delta_obs = torch_utils.quat_to_tan_norm(object_rot_delta) * object_mask.repeat(1, 6)
        local_object_lin_vel_delta = quat_rotate(heading_rot, future_state['object_lin_vel'] - current_state['object_lin_vel']) * object_mask
        local_object_ang_vel_delta = quat_rotate(heading_rot, future_state['object_ang_vel'] - current_state['object_ang_vel']) * object_mask
        return torch.cat((
            local_root_pos_delta,
            root_rot_delta_obs,
            local_root_lin_vel_delta,
            local_root_ang_vel_delta,
            future_state['dof_pos'] - current_state['dof_pos'],
            future_state['dof_vel'] - current_state['dof_vel'],
            local_object_pos_delta,
            object_rot_delta_obs,
            local_object_lin_vel_delta,
            local_object_ang_vel_delta,
        ), dim=-1)

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            obs_buf = self.obs_buf
        else:
            obs_buf = self.obs_buf[env_ids]
        current_state = self._gather_current_state(env_ids)
        current_features = self._build_current_features(current_state)
        future_features = []
        for offset in g1_native_spec.FUTURE_OFFSETS:
            next_frames = torch.minimum(self.progress_buf[env_ids] + offset, self.max_episode_length[self.data_id[env_ids]] - 1)
            future_state = self.motion_lib.get_state(self.data_id[env_ids], next_frames)
            future_features.append(self._build_future_features(current_state, future_state))
        obs = torch.cat((current_features, *future_features), dim=-1)
        if env_ids.shape[0] == self.num_envs:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

    def _compute_reward(self, actions):
        current_state = self._gather_current_state()
        ref_state = self._curr_ref_state
        weights = self.reward_weights

        root_pos_err = torch.sum((ref_state['root_pos'] - current_state['root_pos']) ** 2, dim=-1)
        root_rot_diff = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_state['root_rot']), current_state['root_rot'])
        root_rot_angle, _ = torch_utils.quat_to_angle_axis(root_rot_diff)
        root_vel_err = torch.mean((ref_state['root_lin_vel'] - current_state['root_lin_vel']) ** 2, dim=-1)
        root_ang_vel_err = torch.mean((ref_state['root_ang_vel'] - current_state['root_ang_vel']) ** 2, dim=-1)
        dof_pos_err = torch.mean((ref_state['dof_pos'] - current_state['dof_pos']) ** 2, dim=-1)
        dof_vel_err = torch.mean((ref_state['dof_vel'] - current_state['dof_vel']) ** 2, dim=-1)
        action_rate_err = torch.mean((actions - self.prev_actions) ** 2, dim=-1)
        root_vel_track_err = root_vel_err + root_ang_vel_err

        reward_root_pos = torch.exp(-weights['rootPos'] * root_pos_err)
        reward_root_rot = torch.exp(-weights['rootRot'] * root_rot_angle.abs())
        reward_root_vel = torch.exp(-weights['rootVel'] * root_vel_track_err)
        reward_dof_pos = torch.exp(-weights['dofPos'] * dof_pos_err)
        reward_dof_vel = torch.exp(-weights['dofVel'] * dof_vel_err)
        reward_action_rate = torch.exp(-weights['actionRate'] * action_rate_err)

        reward = (
            reward_root_pos
            * reward_root_rot
            * reward_root_vel
            * reward_dof_pos
            * reward_dof_vel
            * reward_action_rate
        )

        if self.enable_object_tracking:
            object_mask = ref_state['object_mask'].squeeze(-1)
            object_pos_err = torch.sum((ref_state['object_pos'] - current_state['object_pos']) ** 2, dim=-1)
            object_rot_diff = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_state['object_rot']), current_state['object_rot'])
            object_rot_angle, _ = torch_utils.quat_to_angle_axis(object_rot_diff)
            object_reward = torch.exp(-weights['objPos'] * object_pos_err) * torch.exp(-weights['objRot'] * object_rot_angle.abs())
            reward_object = torch.where(object_mask > 0.5, object_reward, torch.ones_like(object_reward))
            reward = reward * reward_object
            object_reset = torch.logical_and(object_mask > 0.5, object_pos_err.sqrt() > g1_native_spec.RESET_THRESHOLDS['object_pos_l2'])
            object_pos_rmse = torch.where(object_mask > 0.5, object_pos_err.sqrt(), torch.zeros_like(object_pos_err))
        else:
            object_reset = torch.zeros_like(root_pos_err, dtype=torch.bool)
            reward_object = torch.ones_like(root_pos_err)
            object_pos_rmse = torch.zeros_like(root_pos_err)

        self.rew_buf[:] = reward
        self.extras['reward_root_pos'] = reward_root_pos
        self.extras['reward_root_rot'] = reward_root_rot
        self.extras['reward_root_vel'] = reward_root_vel
        self.extras['reward_dof_pos'] = reward_dof_pos
        self.extras['reward_dof_vel'] = reward_dof_vel
        self.extras['reward_action_rate'] = reward_action_rate
        self.extras['reward_object'] = reward_object
        self.extras['err_root_pos_rmse'] = root_pos_err.sqrt()
        self.extras['err_root_rot_angle'] = root_rot_angle.abs()
        self.extras['err_root_vel_mse'] = root_vel_track_err
        self.extras['err_dof_pos_rmse'] = torch.sqrt(dof_pos_err)
        self.extras['err_dof_vel_mse'] = dof_vel_err
        self.extras['err_action_rate_mse'] = action_rate_err
        self.extras['err_object_pos_rmse'] = object_pos_rmse
        self._track_reset_mask = (
            (root_pos_err.sqrt() > g1_native_spec.RESET_THRESHOLDS['root_pos_l2'])
            | (root_rot_angle.abs() > g1_native_spec.RESET_THRESHOLDS['root_rot_angle'])
            | (torch.sqrt(dof_pos_err) > g1_native_spec.RESET_THRESHOLDS['dof_rmse'])
            | object_reset
        )
        self.prev_actions[:] = actions

    def _compute_reset(self):
        _, terminated = self.compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self.obs_buf,
            self._rigid_body_pos,
            self.max_episode_length[self.data_id],
            self._enable_early_termination,
            self._termination_heights,
            self.start_times,
            self.rollout_length,
        )
        terminated = torch.where(self._track_reset_mask, torch.ones_like(terminated), terminated)
        horizon_reset = torch.logical_or(
            self.progress_buf >= self.max_episode_length[self.data_id] - 1,
            self.progress_buf - self.start_times >= self.rollout_length - 1,
        )
        self._terminate_buf[:] = terminated
        self.reset_buf[:] = torch.where(torch.logical_or(horizon_reset, terminated.bool()), torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
