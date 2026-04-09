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
    DEFAULT_SPARSE_BODY_TO_HUMAN_JOINT = {
        'pelvis': 0,
        'torso_link': 9,
        'left_knee_link': 2,
        'right_knee_link': 6,
        'left_ankle_pitch_link': 3,
        'left_ankle_roll_link': 3,
        'left_foot_contact_point': 10,
        'right_ankle_pitch_link': 7,
        'right_ankle_roll_link': 7,
        'right_foot_contact_point': 11,
        'left_shoulder_pitch_link': 15,
        'left_shoulder_roll_link': 15,
        'left_shoulder_yaw_link': 15,
        'left_elbow_link': 16,
        'right_shoulder_pitch_link': 34,
        'right_shoulder_roll_link': 34,
        'right_shoulder_yaw_link': 34,
        'right_elbow_link': 35,
    }
    DEFAULT_BALANCE_BODY_NAMES = (
        'left_shoulder_roll_link',
        'right_shoulder_roll_link',
        'right_hip_yaw_link',
        'left_hip_yaw_link',
    )

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
        self.reward_tracking_scales = cfg['env'].get('rewardTrackingScales', {})
        self.rollout_length = cfg['env']['rolloutLength']
        self.enable_object_tracking = bool(cfg['env'].get('enableObjectTracking', False))
        self.require_object_data = bool(cfg['env'].get('requireObjectData', False))
        self.enable_fake_reward = bool(cfg['env'].get('fakeReward', False))
        self.enable_hdmi_reward = bool(cfg['env'].get('hdmiReward', False))
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        self.data_sub = cfg['env'].get('dataSub', [])
        self.reset_noise = cfg['env'].get('resetNoise', {})
        self.termination_cfg = cfg['env'].get('imitationTermination', {})
        self.action_scale_mult = float(cfg['env'].get('actionScaleMult', 1.0))
        self.fake_reward_cfg = cfg['env'].get('fakeRewardConfig', {})
        self.hdmi_reward_cfg = cfg['env'].get('hdmiRewardConfig', {})
        balance_cfg = cfg['env'].get('balanceReward', {})
        self._balance_body_names = tuple(balance_cfg.get('bodyNames', self.DEFAULT_BALANCE_BODY_NAMES))
        if len(self._balance_body_names) != 4:
            raise ValueError('balanceReward.bodyNames must contain exactly 4 rigid body names')
        self._balance_max_area = float(balance_cfg.get('maxArea', 0.25))
        if self._balance_max_area <= 0.0:
            raise ValueError('balanceReward.maxArea must be > 0')
        tracked_body_names = cfg['env'].get('trackedBodyNames', [])
        proxy_joint_indices = cfg['env'].get('trackedBodyHumanJointIndices')
        self._tracked_body_names = tuple(tracked_body_names)
        if proxy_joint_indices is None:
            proxy_joint_indices = [self.DEFAULT_SPARSE_BODY_TO_HUMAN_JOINT[name] for name in self._tracked_body_names]
        if len(proxy_joint_indices) != len(self._tracked_body_names):
            raise ValueError('trackedBodyHumanJointIndices must align with trackedBodyNames')
        self._tracked_body_human_joint_indices = tuple(int(idx) for idx in proxy_joint_indices)
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
        self._drift_violation_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self._curr_ref_state = self.motion_lib.get_state(torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                                                         torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        self._tracked_body_ids = self._build_tracked_body_ids_tensor()
        self._balance_body_ids = self._build_balance_body_ids_tensor()
        self._foot_body_names = tuple(self.hdmi_reward_cfg.get('footBodyNames', ('left_ankle_roll_link', 'right_ankle_roll_link')))
        self._foot_body_ids = self._build_body_ids_tensor(self._foot_body_names, 'HDMI foot')
        self._tracked_body_human_joint_ids = to_torch(self._tracked_body_human_joint_indices, device=self.device, dtype=torch.long)
        self._foot_in_contact = torch.zeros((self.num_envs, len(self._foot_body_names)), device=self.device, dtype=torch.bool)
        self._foot_air_time = torch.zeros((self.num_envs, len(self._foot_body_names)), device=self.device, dtype=torch.float32)
        if self._pd_control and self.action_scale_mult != 1.0:
            self._pd_action_scale *= self.action_scale_mult
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

    def _build_tracked_body_ids_tensor(self):
        if not self._tracked_body_names:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        return self._build_body_ids_tensor(self._tracked_body_names, 'tracked')

    def _build_balance_body_ids_tensor(self):
        return self._build_body_ids_tensor(self._balance_body_names, 'balance')

    def _build_body_ids_tensor(self, body_names, label):
        if not body_names:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            if body_id == -1:
                raise ValueError(f'Unknown {label} body for native G1 task: {body_name}')
            body_ids.append(body_id)
        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _compute_balance_area_xy(self, body_pos):
        # body_pos shape: [num_envs, 4, 3]
        xy = body_pos[..., :2]
        x = xy[..., 0]
        y = xy[..., 1]
        x_next = torch.roll(x, shifts=-1, dims=1)
        y_next = torch.roll(y, shifts=-1, dims=1)
        signed_area = 0.5 * torch.sum(x * y_next - y * x_next, dim=1)
        return torch.abs(signed_area)

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

    def _sample_uniform_noise(self, shape, magnitude, dtype):
        if magnitude <= 0.0:
            return torch.zeros(shape, device=self.device, dtype=dtype)
        return (2.0 * torch.rand(shape, device=self.device, dtype=dtype) - 1.0) * magnitude

    def _apply_reset_noise(self, ref):
        noisy_root_pos = ref['root_pos'].clone()
        root_pos_noise = self.reset_noise.get('rootPos', [0.0, 0.0, 0.0])
        noisy_root_pos[:, 0] += self._sample_uniform_noise((ref['root_pos'].shape[0],), float(root_pos_noise[0]), ref['root_pos'].dtype)
        noisy_root_pos[:, 1] += self._sample_uniform_noise((ref['root_pos'].shape[0],), float(root_pos_noise[1]), ref['root_pos'].dtype)
        noisy_root_pos[:, 2] += self._sample_uniform_noise((ref['root_pos'].shape[0],), float(root_pos_noise[2]), ref['root_pos'].dtype)

        noisy_root_rot = ref['root_rot'].clone()
        root_rot_noise = self.reset_noise.get('rootRot', [0.0, 0.0, 0.0])
        roll = self._sample_uniform_noise((ref['root_rot'].shape[0],), float(root_rot_noise[0]), ref['root_rot'].dtype)
        pitch = self._sample_uniform_noise((ref['root_rot'].shape[0],), float(root_rot_noise[1]), ref['root_rot'].dtype)
        yaw = self._sample_uniform_noise((ref['root_rot'].shape[0],), float(root_rot_noise[2]), ref['root_rot'].dtype)
        root_rot_delta = quat_from_euler_xyz(roll, pitch, yaw)
        noisy_root_rot = g1_native_spec.normalize_quat(quat_mul(root_rot_delta, noisy_root_rot))

        noisy_dof_pos = ref['dof_pos'] + self._sample_uniform_noise(ref['dof_pos'].shape, float(self.reset_noise.get('dofPos', 0.0)), ref['dof_pos'].dtype)
        noisy_dof_pos = torch.clamp(noisy_dof_pos, self.dof_limits_lower.unsqueeze(0), self.dof_limits_upper.unsqueeze(0))

        noisy_root_lin_vel = ref['root_lin_vel'] + self._sample_uniform_noise(ref['root_lin_vel'].shape, float(self.reset_noise.get('rootLinVel', 0.0)), ref['root_lin_vel'].dtype)
        noisy_root_ang_vel = ref['root_ang_vel'] + self._sample_uniform_noise(ref['root_ang_vel'].shape, float(self.reset_noise.get('rootAngVel', 0.0)), ref['root_ang_vel'].dtype)
        noisy_dof_vel = ref['dof_vel'] + self._sample_uniform_noise(ref['dof_vel'].shape, float(self.reset_noise.get('dofVel', 0.0)), ref['dof_vel'].dtype)
        return noisy_root_pos, noisy_root_rot, noisy_dof_pos, noisy_root_lin_vel, noisy_root_ang_vel, noisy_dof_vel

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
        self._drift_violation_counts[env_ids] = 0
        self._foot_in_contact[env_ids] = False
        self._foot_air_time[env_ids] = 0.0
        noisy_root_pos, noisy_root_rot, noisy_dof_pos, noisy_root_lin_vel, noisy_root_ang_vel, noisy_dof_vel = self._apply_reset_noise(ref)
        self._set_env_state(
            env_ids,
            noisy_root_pos,
            noisy_root_rot,
            noisy_dof_pos,
            noisy_root_lin_vel,
            noisy_root_ang_vel,
            noisy_dof_vel,
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
            'tracked_body_pos': self._rigid_body_pos[:, self._tracked_body_ids, :] if env_ids is None else self._rigid_body_pos[env_ids][:, self._tracked_body_ids, :],
            'balance_body_pos': self._rigid_body_pos[:, self._balance_body_ids, :] if env_ids is None else self._rigid_body_pos[env_ids][:, self._balance_body_ids, :],
            'foot_body_vel': self._rigid_body_vel[:, self._foot_body_ids, :] if env_ids is None else self._rigid_body_vel[env_ids][:, self._foot_body_ids, :],
            'foot_contact_forces': self._contact_forces[:, self._foot_body_ids, :] if env_ids is None else self._contact_forces[env_ids][:, self._foot_body_ids, :],
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
        scales = self.reward_tracking_scales

        root_pos_err = torch.sum((ref_state['root_pos'] - current_state['root_pos']) ** 2, dim=-1)
        root_rot_diff = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_state['root_rot']), current_state['root_rot'])
        root_rot_angle, _ = torch_utils.quat_to_angle_axis(root_rot_diff)
        root_vel_err = torch.mean((ref_state['root_lin_vel'] - current_state['root_lin_vel']) ** 2, dim=-1)
        root_ang_vel_err = torch.mean((ref_state['root_ang_vel'] - current_state['root_ang_vel']) ** 2, dim=-1)
        dof_pos_err = torch.mean((ref_state['dof_pos'] - current_state['dof_pos']) ** 2, dim=-1)
        dof_vel_err = torch.mean((ref_state['dof_vel'] - current_state['dof_vel']) ** 2, dim=-1)
        action_rate_err = torch.mean((actions - self.prev_actions) ** 2, dim=-1)
        root_vel_track_err = root_vel_err + root_ang_vel_err
        sparse_body_err = torch.zeros_like(root_pos_err)
        balance_area_xy = torch.zeros_like(root_pos_err)
        balance_area_norm = torch.zeros_like(root_pos_err)

        lower_soft = self.dof_limits_lower.unsqueeze(0) + float(self.termination_cfg.get('jointLimitMargin', 0.05))
        upper_soft = self.dof_limits_upper.unsqueeze(0) - float(self.termination_cfg.get('jointLimitMargin', 0.05))
        joint_limit_violation = torch.relu(lower_soft - current_state['dof_pos']) + torch.relu(current_state['dof_pos'] - upper_soft)
        joint_limit_err = torch.mean(joint_limit_violation ** 2, dim=-1)

        reward_root_pos = torch.exp(-scales['rootPos'] * root_pos_err)
        reward_root_rot = torch.exp(-scales['rootRot'] * root_rot_angle.abs())
        reward_root_vel = torch.exp(-scales['rootVel'] * root_vel_track_err)
        reward_dof_pos = torch.exp(-scales['dofPos'] * dof_pos_err)
        reward_dof_vel = torch.exp(-scales['dofVel'] * dof_vel_err)
        reward_sparse_body_pos = torch.zeros_like(root_pos_err)
        reward_balance = torch.zeros_like(root_pos_err)
        reward_hdmi_joint_vel_l2 = torch.zeros_like(root_pos_err)
        reward_hdmi_loco_survival = torch.zeros_like(root_pos_err)
        reward_hdmi_feet_slip = torch.zeros_like(root_pos_err)
        reward_hdmi_feet_air_time = torch.zeros_like(root_pos_err)
        reward_hdmi_feet_survival = torch.zeros_like(root_pos_err)

        if self.enable_fake_reward:
            ref_human_joints = ref_state['source_human_joints'].view(ref_state['source_human_joints'].shape[0], -1, 3)
            ref_sparse_body_pos = ref_human_joints[:, self._tracked_body_human_joint_ids, :]
            ref_sparse_root = ref_human_joints[:, 0:1, :]
            ref_sparse_body_rel = ref_sparse_body_pos - ref_sparse_root
            cur_sparse_body_rel = current_state['tracked_body_pos'] - current_state['root_pos'].unsqueeze(1)
            sparse_body_err = torch.mean((ref_sparse_body_rel - cur_sparse_body_rel) ** 2, dim=(1, 2))
            reward_sparse_body_pos = torch.exp(-float(self.fake_reward_cfg.get('sparseBodyPosScale', 6.0)) * sparse_body_err)

            balance_area_xy = self._compute_balance_area_xy(current_state['balance_body_pos'])
            balance_area_norm = balance_area_xy / self._balance_max_area
            reward_balance = torch.exp(-float(self.fake_reward_cfg.get('balanceScale', 4.0)) * balance_area_norm)

        if self.enable_hdmi_reward:
            reward_hdmi_joint_vel_l2 = -torch.sum(torch.clamp(current_state['dof_vel'] ** 2, max=5.0), dim=-1)
            reward_hdmi_loco_survival = torch.ones_like(root_pos_err)

            contact_force_threshold = float(self.hdmi_reward_cfg.get('feetContactForceThreshold', 1.0))
            feet_slip_tolerance = float(self.hdmi_reward_cfg.get('feetSlipTolerance', 0.0))
            foot_contact_force = torch.norm(current_state['foot_contact_forces'], dim=-1)
            foot_in_contact = foot_contact_force > contact_force_threshold
            foot_xy_speed = torch.norm(current_state['foot_body_vel'][..., :2], dim=-1)
            reward_hdmi_feet_slip = -(foot_in_contact.float() * torch.clamp(foot_xy_speed - feet_slip_tolerance, min=0.0, max=1.0)).sum(dim=-1)

            first_contact = foot_in_contact & (~self._foot_in_contact)
            reward_hdmi_feet_air_time = torch.sum(
                torch.clamp(self._foot_air_time - float(self.hdmi_reward_cfg.get('feetAirTimeThreshold', 0.5)), max=0.0)
                * first_contact.float(),
                dim=-1,
            )
            reward_hdmi_feet_survival = torch.ones_like(root_pos_err)

            self._foot_air_time = torch.where(
                foot_in_contact,
                torch.zeros_like(self._foot_air_time),
                self._foot_air_time + self.dt,
            )
            self._foot_in_contact = foot_in_contact

        root_pos_rmse = root_pos_err.sqrt()
        drift_violation = (
            (root_pos_rmse > float(self.termination_cfg['rootPosThreshold']))
            | (root_rot_angle.abs() > float(self.termination_cfg['rootRotThreshold']))
        )
        self._drift_violation_counts = torch.where(
            drift_violation,
            self._drift_violation_counts + 1,
            torch.zeros_like(self._drift_violation_counts),
        )

        reward = (
            weights['rootPos'] * reward_root_pos
            + weights['rootRot'] * reward_root_rot
            + weights['rootVel'] * reward_root_vel
            + weights['dofPos'] * reward_dof_pos
            + weights['dofVel'] * reward_dof_vel
            + float(self.fake_reward_cfg.get('sparseBodyPosWeight', 0.15)) * reward_sparse_body_pos
            + float(self.fake_reward_cfg.get('balanceWeight', 0.05)) * reward_balance
            + float(self.hdmi_reward_cfg.get('jointVelL2Weight', 0.0005)) * reward_hdmi_joint_vel_l2
            + float(self.hdmi_reward_cfg.get('locoSurvivalWeight', 1.0)) * reward_hdmi_loco_survival
            + float(self.hdmi_reward_cfg.get('feetSlipWeight', 0.5)) * reward_hdmi_feet_slip
            + float(self.hdmi_reward_cfg.get('feetAirTimeWeight', 5.0)) * reward_hdmi_feet_air_time
            + float(self.hdmi_reward_cfg.get('feetSurvivalWeight', 1.0)) * reward_hdmi_feet_survival
            - weights['actionRate'] * action_rate_err
            - weights.get('jointLimit', 0.0) * joint_limit_err
            - weights.get('termination', 0.0) * drift_violation.float()
        )
        reward = torch.clamp(reward, min=0.0)

        if self.enable_object_tracking:
            object_mask = ref_state['object_mask'].squeeze(-1)
            object_pos_err = torch.sum((ref_state['object_pos'] - current_state['object_pos']) ** 2, dim=-1)
            object_rot_diff = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_state['object_rot']), current_state['object_rot'])
            object_rot_angle, _ = torch_utils.quat_to_angle_axis(object_rot_diff)
            object_reward = torch.exp(-scales['objPos'] * object_pos_err) * torch.exp(-scales['objRot'] * object_rot_angle.abs())
            reward_object = torch.where(object_mask > 0.5, object_reward, torch.zeros_like(object_reward))
            reward = reward + weights.get('objPos', 0.0) * reward_object
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
        self.extras['reward_sparse_body_pos'] = reward_sparse_body_pos
        self.extras['reward_balance'] = reward_balance
        self.extras['reward_hdmi_joint_vel_l2'] = reward_hdmi_joint_vel_l2
        self.extras['reward_hdmi_loco_survival'] = reward_hdmi_loco_survival
        self.extras['reward_hdmi_feet_slip'] = reward_hdmi_feet_slip
        self.extras['reward_hdmi_feet_air_time'] = reward_hdmi_feet_air_time
        self.extras['reward_hdmi_feet_survival'] = reward_hdmi_feet_survival
        self.extras['reward_object'] = reward_object
        self.extras['err_root_pos_rmse'] = root_pos_rmse
        self.extras['err_root_rot_angle'] = root_rot_angle.abs()
        self.extras['err_root_vel_mse'] = root_vel_track_err
        self.extras['err_dof_pos_rmse'] = torch.sqrt(dof_pos_err)
        self.extras['err_dof_vel_mse'] = dof_vel_err
        self.extras['err_sparse_body_pos_rmse'] = torch.sqrt(sparse_body_err)
        self.extras['err_balance_area_xy'] = balance_area_xy
        self.extras['err_balance_area_norm'] = balance_area_norm
        self.extras['err_action_rate_mse'] = action_rate_err
        self.extras['err_joint_limit_mse'] = joint_limit_err
        self.extras['err_object_pos_rmse'] = object_pos_rmse
        self.extras['drift_violation_count'] = self._drift_violation_counts.float()
        self._track_reset_mask = (
            (self._drift_violation_counts >= int(self.termination_cfg['persistenceSteps']))
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
