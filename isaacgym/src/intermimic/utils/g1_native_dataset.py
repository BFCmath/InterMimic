from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .. import g1_native_spec
from .path_utils import resolve_repo_path


@dataclass(frozen=True)
class G1NativeSequenceInfo:
    path: Path
    sequence_name: str
    subject_tag: str
    subject_index: int
    object_name: str
    has_object: bool


def _to_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_string(value.item())
        if value.size == 1:
            return _to_string(value.reshape(-1)[0])
    return str(value)


def discover_g1_native_sequences(motion_root: str, allowed_subjects: Sequence[str] | None = None) -> list[G1NativeSequenceInfo]:
    root = resolve_repo_path(motion_root)
    if not root.exists():
        raise FileNotFoundError(f'Native G1 dataset root does not exist: {root}')
    allowed = set(allowed_subjects) if allowed_subjects else None
    infos: list[G1NativeSequenceInfo] = []
    for path in sorted(root.glob('*.npz')):
        with np.load(path, allow_pickle=True) as data:
            sequence_name = _to_string(data['sequence_name']) if 'sequence_name' in data else path.stem
            subject_tag = _to_string(data['subject_tag']) if 'subject_tag' in data else g1_native_spec.parse_subject_tag(sequence_name)
            if allowed is not None and subject_tag not in allowed:
                continue
            subject_index = int(data['subject_index']) if 'subject_index' in data else g1_native_spec.parse_subject_index(subject_tag)
            object_name = _to_string(data['object_name']) if 'object_name' in data else g1_native_spec.parse_object_name(sequence_name)
            if 'has_object' in data:
                has_object = bool(np.array(data['has_object']).reshape(()).item())
            else:
                has_object = 'object_pos' in data and 'object_rot' in data
        infos.append(G1NativeSequenceInfo(
            path=path,
            sequence_name=sequence_name,
            subject_tag=subject_tag,
            subject_index=subject_index,
            object_name=object_name,
            has_object=has_object,
        ))
    if not infos:
        raise FileNotFoundError(f'No native G1 sequences found under {root}')
    return infos


class G1NativeMotionDataset:
    STATE_FIELDS = (
        'root_pos',
        'root_rot',
        'root_lin_vel',
        'root_ang_vel',
        'dof_pos',
        'dof_vel',
        'object_pos',
        'object_rot',
        'object_lin_vel',
        'object_ang_vel',
        'object_mask',
        'source_human_joints',
    )

    def __init__(self, motion_root: str, *, allowed_subjects: Sequence[str] | None = None,
                 device: torch.device | str = 'cpu') -> None:
        self.device = torch.device(device)
        self.motion_root = resolve_repo_path(motion_root)
        self.sequence_infos = discover_g1_native_sequences(motion_root, allowed_subjects)
        self.sequence_names = [info.sequence_name for info in self.sequence_infos]
        self.subject_tags = [info.subject_tag for info in self.sequence_infos]
        self.subject_indices = torch.tensor([info.subject_index for info in self.sequence_infos], dtype=torch.long, device=self.device)
        self.object_names = sorted({info.object_name for info in self.sequence_infos if info.has_object})
        if self.object_names:
            self.object_ids = torch.tensor([
                self.object_names.index(info.object_name) if info.has_object else -1
                for info in self.sequence_infos
            ], dtype=torch.long, device=self.device)
            self.obj2motion = torch.stack([self.object_ids == idx for idx in range(len(self.object_names))], dim=0)
        else:
            self.object_ids = torch.full((len(self.sequence_infos),), -1, dtype=torch.long, device=self.device)
            self.obj2motion = torch.zeros((0, len(self.sequence_infos)), dtype=torch.bool, device=self.device)
        self.has_object_sequence = torch.tensor([info.has_object for info in self.sequence_infos], dtype=torch.bool, device=self.device)
        self._load_sequences()

    def _load_sequences(self) -> None:
        sequences = [self._load_single_sequence(info.path) for info in self.sequence_infos]
        self.max_episode_length = torch.tensor([seq['root_pos'].shape[0] for seq in sequences], dtype=torch.long, device=self.device)
        self.num_sequences = len(sequences)
        max_length = int(self.max_episode_length.max().item())
        for field in self.STATE_FIELDS:
            padded = []
            for seq in sequences:
                data = seq[field]
                pad_frames = max_length - data.shape[0]
                if pad_frames:
                    padded_data = F.pad(data, (0, 0, 0, pad_frames), 'constant', 0.0)
                    if field.endswith('_rot'):
                        padded_data[-pad_frames:, 3] = 1.0
                else:
                    padded_data = data
                padded.append(padded_data)
            setattr(self, field, torch.stack(padded, dim=0).to(self.device))

    def _validate_frame_count(self, path: Path, expected_frames: int, **tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            if tensor.shape[0] != expected_frames:
                raise ValueError(f'Field {name} in {path} has {tensor.shape[0]} frames, expected {expected_frames}')

    def _load_single_sequence(self, path: Path) -> dict[str, torch.Tensor]:
        with np.load(path, allow_pickle=True) as data:
            missing = [name for name in g1_native_spec.REQUIRED_SEQUENCE_FIELDS if name not in data]
            if missing:
                raise ValueError(f'Missing required native G1 fields in {path}: {missing}')
            schema_version = _to_string(data['schema_version'])
            if schema_version != g1_native_spec.SCHEMA_VERSION:
                raise ValueError(f'Unsupported schema version {schema_version!r} in {path}')
            quat_convention = _to_string(data['quat_convention'])
            if quat_convention != g1_native_spec.PERSISTED_QUATERNION_CONVENTION:
                raise ValueError(f'Quaternion convention mismatch in {path}: {quat_convention!r}')

            joint_names = [_to_string(name) for name in data['joint_names']]
            if tuple(joint_names) != g1_native_spec.G1_29DOF_JOINT_NAMES:
                raise ValueError(f'Joint order mismatch in {path}')
            body_names = [_to_string(name) for name in data['body_names']]
            if tuple(body_names) != g1_native_spec.G1_BODY_NAMES:
                raise ValueError(f'Body name order mismatch in {path}')

            fps = float(np.array(data['fps']).reshape(()).item())
            root_pos = torch.from_numpy(data['root_pos']).to(torch.float32)
            root_rot = g1_native_spec.normalize_quat(torch.from_numpy(data['root_rot']).to(torch.float32))
            dof_pos = torch.from_numpy(data['dof_pos']).to(torch.float32)
            root_lin_vel = torch.from_numpy(data['root_lin_vel']).to(torch.float32)
            root_ang_vel = torch.from_numpy(data['root_ang_vel']).to(torch.float32)
            dof_vel = torch.from_numpy(data['dof_vel']).to(torch.float32)

            if root_pos.ndim != 2 or root_pos.shape[-1] != g1_native_spec.ROOT_POS_SIZE:
                raise ValueError(f'root_pos in {path} must have shape [T, 3], got {tuple(root_pos.shape)}')
            if root_rot.ndim != 2 or root_rot.shape[-1] != g1_native_spec.ROOT_ROT_SIZE:
                raise ValueError(f'root_rot in {path} must have shape [T, 4], got {tuple(root_rot.shape)}')
            if dof_pos.ndim != 2 or dof_pos.shape[-1] != g1_native_spec.G1_DOF_COUNT:
                raise ValueError(f'Expected {g1_native_spec.G1_DOF_COUNT} DOFs in {path}, got {tuple(dof_pos.shape)}')
            self._validate_frame_count(
                path,
                root_pos.shape[0],
                root_rot=root_rot,
                root_lin_vel=root_lin_vel,
                root_ang_vel=root_ang_vel,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
            )

            if 'has_object' in data:
                has_object = bool(np.array(data['has_object']).reshape(()).item())
            else:
                has_object = 'object_pos' in data and 'object_rot' in data

            if has_object:
                required_object_fields = ('object_pos', 'object_rot', 'object_lin_vel', 'object_ang_vel')
                missing_object = [name for name in required_object_fields if name not in data]
                if missing_object:
                    raise ValueError(f'Missing required object fields in {path}: {missing_object}')
                object_pos = torch.from_numpy(data['object_pos']).to(torch.float32)
                object_rot = g1_native_spec.normalize_quat(torch.from_numpy(data['object_rot']).to(torch.float32))
                object_lin_vel = torch.from_numpy(data['object_lin_vel']).to(torch.float32)
                object_ang_vel = torch.from_numpy(data['object_ang_vel']).to(torch.float32)
                object_mask = torch.from_numpy(data['object_mask']).to(torch.float32) if 'object_mask' in data else torch.ones((root_pos.shape[0], 1), dtype=torch.float32)
                self._validate_frame_count(
                    path,
                    root_pos.shape[0],
                    object_pos=object_pos,
                    object_rot=object_rot,
                    object_lin_vel=object_lin_vel,
                    object_ang_vel=object_ang_vel,
                    object_mask=object_mask,
                )
            else:
                object_pos = torch.zeros((root_pos.shape[0], 3), dtype=torch.float32)
                object_rot = g1_native_spec.identity_quat(root_pos.shape[0])
                object_lin_vel = torch.zeros((root_pos.shape[0], 3), dtype=torch.float32)
                object_ang_vel = torch.zeros((root_pos.shape[0], 3), dtype=torch.float32)
                object_mask = torch.zeros((root_pos.shape[0], 1), dtype=torch.float32)

            if 'source_human_joints' in data:
                source_human_joints = torch.from_numpy(data['source_human_joints']).to(torch.float32)
                if source_human_joints.ndim != 3 or source_human_joints.shape[-1] != 3:
                    raise ValueError(
                        f'source_human_joints in {path} must have shape [T, J, 3], got {tuple(source_human_joints.shape)}'
                    )
                self._validate_frame_count(path, root_pos.shape[0], source_human_joints=source_human_joints)
                source_human_joints = source_human_joints.view(source_human_joints.shape[0], -1)
            else:
                source_human_joints = torch.zeros((root_pos.shape[0], 52 * 3), dtype=torch.float32)

        return {
            'root_pos': root_pos,
            'root_rot': root_rot,
            'root_lin_vel': root_lin_vel,
            'root_ang_vel': root_ang_vel,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'object_pos': object_pos,
            'object_rot': object_rot,
            'object_lin_vel': object_lin_vel,
            'object_ang_vel': object_ang_vel,
            'object_mask': object_mask,
            'source_human_joints': source_human_joints,
        }

    def get_state(self, sequence_ids: torch.Tensor, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for field in self.STATE_FIELDS:
            tensor = getattr(self, field)
            state[field] = tensor[sequence_ids, frame_ids]
        return state
