"""Single source of truth for the native G1 teacher pipeline.

This module centralizes the persistent assumptions for the native
``G1 -> G1`` teacher PPO path so the converter, dataset loader,
configs, and runtime task all agree on the same morphology and
serialization contract.

The current native path intentionally targets Holosoma's ``g1_29dof``
morphology instead of the legacy Isaac Gym ``g1_29dof_with_hand``
morphology because the observed Holosoma qpos layout is:

- root position: 3
- root quaternion (MuJoCo ``wxyz``): 4
- robot joints: 29
- object pose (optional): 7

All persisted native dataset quaternions use Isaac Gym order
``xyzw`` in world coordinates.
"""

from __future__ import annotations

from typing import Iterable

import torch

SCHEMA_VERSION = 'g1_native_v1'
PERSISTED_QUATERNION_CONVENTION = 'xyzw'
HOLOSOMA_QPOS_QUATERNION_CONVENTION = 'wxyz'

FUTURE_OFFSETS = (1, 16)

ROOT_POS_SIZE = 3
ROOT_ROT_SIZE = 4
ROOT_LIN_VEL_SIZE = 3
ROOT_ANG_VEL_SIZE = 3
OBJECT_POS_SIZE = 3
OBJECT_ROT_SIZE = 4
OBJECT_LIN_VEL_SIZE = 3
OBJECT_ANG_VEL_SIZE = 3
OBJECT_MASK_SIZE = 1

REQUIRED_SEQUENCE_FIELDS = (
    'schema_version',
    'quat_convention',
    'sequence_name',
    'subject_tag',
    'subject_index',
    'object_name',
    'fps',
    'has_object',
    'joint_names',
    'body_names',
    'root_pos',
    'root_rot',
    'root_lin_vel',
    'root_ang_vel',
    'dof_pos',
    'dof_vel',
)

OPTIONAL_SEQUENCE_FIELDS = (
    'object_pos',
    'object_rot',
    'object_lin_vel',
    'object_ang_vel',
    'object_mask',
    'source_human_joints',
    'source_cost',
)

G1_29DOF_JOINT_NAMES = (
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
)
G1_DOF_COUNT = len(G1_29DOF_JOINT_NAMES)

G1_BODY_NAMES = (
    'pelvis',
    'pelvis_contour_link',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'left_foot_contact_point',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'right_foot_contact_point',
    'waist_yaw_link',
    'waist_roll_link',
    'torso_link',
    'logo_link',
    'head_link',
    'imu_in_torso',
    'imu_in_pelvis',
    'mid360_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link',
    'left_wrist_pitch_link',
    'left_wrist_yaw_link',
    'left_rubber_hand',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link',
    'right_wrist_pitch_link',
    'right_wrist_yaw_link',
    'right_rubber_hand',
    'LL_FOOT',
    'LR_FOOT',
)

G1_KEY_BODY_NAMES = (
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_roll_link',
    'left_foot_contact_point',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_roll_link',
    'right_foot_contact_point',
    'torso_link',
    'head_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'left_rubber_hand',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
    'right_rubber_hand',
)

G1_CONTACT_BODY_NAMES = (
    'left_foot_contact_point',
    'right_foot_contact_point',
    'left_knee_link',
    'right_knee_link',
    'torso_link',
    'left_elbow_link',
    'right_elbow_link',
    'left_rubber_hand',
    'right_rubber_hand',
)

IDENTITY_KEY_INDICES = tuple(range(len(G1_KEY_BODY_NAMES)))
IDENTITY_CONTACT_INDICES = tuple(range(len(G1_CONTACT_BODY_NAMES)))

PD_STIFFNESS = (500.0,) * 12 + (300.0,) * 3 + (200.0,) * 14
PD_DAMPING = (50.0,) * 12 + (30.0,) * 3 + (20.0,) * 14

RESET_THRESHOLDS = {
    'root_pos_l2': 1.0,
    'root_rot_angle': 1.75,
    'dof_rmse': 0.75,
    'object_pos_l2': 1.0,
}

G1_VS_SMPLX_MISMATCH_NOTES = (
    'G1 has 29 robot DoFs in the Holosoma retarget path, not the legacy 153-channel SMPL-X tensor contract.',
    'G1 body names and body count come from the Holosoma g1_29dof URDF, not the 52 SMPL-X joints used by OMOMO NEW.',
    'Holosoma qpos stores the root quaternion in MuJoCo wxyz order; native persisted data is normalized to Isaac Gym xyzw order.',
    'Object pose is optional in native data. A sequence can remain trainable in body-only mode with a hidden object actor.',
    'The native task resets robot root pose and DoFs directly from the reference frame instead of reusing the legacy init pose shim.',
    'The legacy G1 path padded observations and rewards into SMPL-X-shaped slots. Native G1 removes that padding contract entirely.',
    'Holosoma provides source human joints, but those 52 human points are not a first-class G1 body pose target. Native v1 stores them as provenance only.',
    'Body rotation and contact graph targets remain deferred until a reliable G1-native FK/body-state export is available during conversion.',
)

IMPLEMENTED_V1_FEATURES = (
    'Native Holosoma qpos conversion into a typed G1 dataset with root, joint, velocity, and optional object state.',
    'Native G1 motion loader with schema validation, quaternion convention validation, and subject/object metadata.',
    'Reference-faithful G1 reset using native root pose, root velocity, joint position, and joint velocity.',
    'Native G1 PPO observation and reward path without SMPL-X slot padding.',
    'Optional object tracking support gated by dataset availability and config.',
)

OPEN_GAPS = (
    'Native v1 does not yet export or consume G1 body_pos/body_rot trajectories for dense body tracking rewards.',
    'Native v1 does not yet export or consume contact labels or interaction graphs from the converted dataset.',
    'Running teacher PPO still requires an Isaac Gym runtime environment exactly as described in the repository README.',
)

CURRENT_FEATURE_SIZE = (
    1
    + ROOT_LIN_VEL_SIZE
    + ROOT_ANG_VEL_SIZE
    + G1_DOF_COUNT
    + G1_DOF_COUNT
    + OBJECT_POS_SIZE
    + 6
    + OBJECT_LIN_VEL_SIZE
    + OBJECT_ANG_VEL_SIZE
    + OBJECT_MASK_SIZE
)

FUTURE_FEATURE_SIZE = (
    ROOT_POS_SIZE
    + 6
    + ROOT_LIN_VEL_SIZE
    + ROOT_ANG_VEL_SIZE
    + G1_DOF_COUNT
    + G1_DOF_COUNT
    + OBJECT_POS_SIZE
    + 6
    + OBJECT_LIN_VEL_SIZE
    + OBJECT_ANG_VEL_SIZE
)


def native_observation_size() -> int:
    return CURRENT_FEATURE_SIZE + len(FUTURE_OFFSETS) * FUTURE_FEATURE_SIZE


def identity_quat(batch_shape: Iterable[int] | int, *, device: torch.device | str | None = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(batch_shape, int):
        shape = (batch_shape, 4)
    else:
        shape = tuple(batch_shape) + (4,)
    quat = torch.zeros(shape, device=device, dtype=dtype)
    quat[..., 3] = 1.0
    return quat


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)


def quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    return torch.stack((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dim=-1)


def quat_to_exp_map_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat(quat)
    quat = torch.where(quat[..., 3:4] < 0.0, -quat, quat)
    xyz = quat[..., :3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / sin_half.clamp_min(1e-8)
    exp_map = axis * angle
    return torch.where(sin_half < 1e-6, 2.0 * xyz, exp_map)


def mujoco_wxyz_to_isaac_xyzw(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f'Expected quaternion with last dim 4, got {quat.shape}')
    return normalize_quat(torch.cat((quat[..., 1:], quat[..., :1]), dim=-1))


def isaac_xyzw_to_mujoco_wxyz(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f'Expected quaternion with last dim 4, got {quat.shape}')
    quat = normalize_quat(quat)
    return torch.cat((quat[..., 3:], quat[..., :3]), dim=-1)


def finite_difference_linear(values: torch.Tensor, fps: float) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.clone()
    vel = (values[1:] - values[:-1]) * fps
    return torch.cat((torch.zeros_like(values[:1]), vel), dim=0)


def finite_difference_angular(quat_xyzw: torch.Tensor, fps: float) -> torch.Tensor:
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f'Expected quaternion tensor, got {quat_xyzw.shape}')
    if quat_xyzw.shape[0] == 0:
        return quat_xyzw[..., :3].clone()
    quat_xyzw = normalize_quat(quat_xyzw)
    if quat_xyzw.shape[0] == 1:
        return torch.zeros(quat_xyzw.shape[:-1] + (3,), device=quat_xyzw.device, dtype=quat_xyzw.dtype)
    delta = quat_mul_xyzw(quat_xyzw[1:], quat_conjugate_xyzw(quat_xyzw[:-1]))
    ang_vel = quat_to_exp_map_xyzw(delta) * fps
    return torch.cat((torch.zeros_like(ang_vel[:1]), ang_vel), dim=0)


def infer_holosoma_has_object(qpos_dim: int) -> bool:
    if qpos_dim == ROOT_POS_SIZE + ROOT_ROT_SIZE + G1_DOF_COUNT:
        return False
    if qpos_dim == ROOT_POS_SIZE + ROOT_ROT_SIZE + G1_DOF_COUNT + OBJECT_POS_SIZE + OBJECT_ROT_SIZE:
        return True
    raise ValueError(
        f'Unsupported Holosoma qpos dimension {qpos_dim}. '
        f'Expected {ROOT_POS_SIZE + ROOT_ROT_SIZE + G1_DOF_COUNT} or '
        f'{ROOT_POS_SIZE + ROOT_ROT_SIZE + G1_DOF_COUNT + OBJECT_POS_SIZE + OBJECT_ROT_SIZE}.'
    )


def split_holosoma_qpos(qpos: torch.Tensor) -> dict[str, torch.Tensor | bool]:
    has_object = infer_holosoma_has_object(qpos.shape[-1])
    root_pos = qpos[..., 0:3]
    root_rot = mujoco_wxyz_to_isaac_xyzw(qpos[..., 3:7])
    dof_start = 7
    dof_end = dof_start + G1_DOF_COUNT
    dof_pos = qpos[..., dof_start:dof_end]

    if has_object:
        object_pos = qpos[..., dof_end:dof_end + OBJECT_POS_SIZE]
        object_rot = mujoco_wxyz_to_isaac_xyzw(
            qpos[..., dof_end + OBJECT_POS_SIZE:dof_end + OBJECT_POS_SIZE + OBJECT_ROT_SIZE]
        )
        object_mask = torch.ones(qpos.shape[:-1] + (1,), device=qpos.device, dtype=qpos.dtype)
    else:
        object_pos = torch.zeros(qpos.shape[:-1] + (OBJECT_POS_SIZE,), device=qpos.device, dtype=qpos.dtype)
        object_rot = identity_quat(qpos.shape[:-1], device=qpos.device, dtype=qpos.dtype)
        object_mask = torch.zeros(qpos.shape[:-1] + (1,), device=qpos.device, dtype=qpos.dtype)

    return {
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'object_pos': object_pos,
        'object_rot': object_rot,
        'object_mask': object_mask,
        'has_object': has_object,
    }


def parse_subject_tag(sequence_name: str) -> str:
    stem = sequence_name[:-4] if sequence_name.endswith('.npz') else sequence_name
    return stem.split('_')[0]


def parse_subject_index(subject_tag: str) -> int:
    if not subject_tag.startswith('sub'):
        raise ValueError(f'Expected subject tag like sub3, got {subject_tag}')
    return int(subject_tag[3:])


def parse_object_name(sequence_name: str) -> str:
    stem = sequence_name[:-4] if sequence_name.endswith('.npz') else sequence_name
    parts = stem.split('_')
    if not parts:
        raise ValueError(f'Could not parse object name from {sequence_name}')
    if parts[-1] == 'original' and len(parts) >= 4:
        return parts[-3]
    if len(parts) >= 3:
        return parts[-2]
    raise ValueError(f'Could not parse object name from {sequence_name}')
