from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .. import g1_native_spec
from .path_utils import resolve_repo_path


def _to_numpy_strings(values: tuple[str, ...]) -> np.ndarray:
    return np.array(values, dtype=np.str_)


def convert_sequence(input_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    with np.load(input_path, allow_pickle=True) as data:
        if 'qpos' not in data:
            raise ValueError(f'{input_path} does not contain qpos')
        qpos = torch.from_numpy(data['qpos']).to(torch.float32)
        split = g1_native_spec.split_holosoma_qpos(qpos)
        fps = int(np.array(data['fps']).reshape(()).item()) if 'fps' in data else 30
        sequence_name = input_path.stem
        subject_tag = g1_native_spec.parse_subject_tag(sequence_name)
        subject_index = g1_native_spec.parse_subject_index(subject_tag)
        object_name = g1_native_spec.parse_object_name(sequence_name)

        root_pos = split['root_pos']
        root_rot = split['root_rot']
        dof_pos = split['dof_pos']
        object_pos = split['object_pos']
        object_rot = split['object_rot']
        object_mask = split['object_mask']
        has_object = bool(split['has_object'])

        payload: dict[str, np.ndarray] = {
            'schema_version': np.array(g1_native_spec.SCHEMA_VERSION),
            'quat_convention': np.array(g1_native_spec.PERSISTED_QUATERNION_CONVENTION),
            'sequence_name': np.array(sequence_name),
            'subject_tag': np.array(subject_tag),
            'subject_index': np.array(subject_index, dtype=np.int64),
            'object_name': np.array(object_name),
            'fps': np.array(fps, dtype=np.int64),
            'has_object': np.array(has_object),
            'joint_names': _to_numpy_strings(g1_native_spec.G1_29DOF_JOINT_NAMES),
            'body_names': _to_numpy_strings(g1_native_spec.G1_BODY_NAMES),
            'root_pos': root_pos.cpu().numpy(),
            'root_rot': root_rot.cpu().numpy(),
            'root_lin_vel': g1_native_spec.finite_difference_linear(root_pos, fps).cpu().numpy(),
            'root_ang_vel': g1_native_spec.finite_difference_angular(root_rot, fps).cpu().numpy(),
            'dof_pos': dof_pos.cpu().numpy(),
            'dof_vel': g1_native_spec.finite_difference_linear(dof_pos, fps).cpu().numpy(),
            'object_pos': object_pos.cpu().numpy(),
            'object_rot': object_rot.cpu().numpy(),
            'object_lin_vel': g1_native_spec.finite_difference_linear(object_pos, fps).cpu().numpy(),
            'object_ang_vel': g1_native_spec.finite_difference_angular(object_rot, fps).cpu().numpy(),
            'object_mask': object_mask.cpu().numpy(),
        }
        if 'human_joints' in data:
            payload['source_human_joints'] = data['human_joints'].astype(np.float32)
        if 'cost' in data:
            payload['source_cost'] = np.array(data['cost'])

    output_path = output_dir / input_path.name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'{output_path} already exists; pass --overwrite to replace it')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path


def convert_directory(input_dir: str, output_dir: str, overwrite: bool = False) -> list[Path]:
    src = resolve_repo_path(input_dir)
    dst = resolve_repo_path(output_dir, must_exist=False)
    converted: list[Path] = []
    for path in sorted(src.glob('*.npz')):
        converted.append(convert_sequence(path, dst, overwrite=overwrite))
    if not converted:
        raise FileNotFoundError(f'No .npz files found under {src}')
    return converted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert Holosoma G1 qpos data into the native InterMimic G1 dataset format.')
    parser.add_argument('--input', required=True, help='Input directory containing Holosoma .npz files.')
    parser.add_argument('--output', required=True, help='Output directory for native G1 .npz files.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing native files.')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    converted = convert_directory(args.input, args.output, overwrite=args.overwrite)
    print(f'Converted {len(converted)} files into {resolve_repo_path(args.output, must_exist=False)}')


if __name__ == '__main__':
    main()
