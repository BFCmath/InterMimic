#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "InterAct" / "OMOMO_holosoma_G1"
NATIVE_DATA_DIR = REPO_ROOT / "InterAct" / "OMOMO_holosoma_G1_native"
ENV_CFG_PATH = REPO_ROOT / "isaacgym" / "src" / "intermimic" / "data" / "cfg" / "omomo_g1_native.yaml"
HOL_DATA_TYPE_PATH = (
    REPO_ROOT
    / "holosoma"
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "config_types"
    / "data_type.py"
)


def load_smplh_demo_joints() -> list[str]:
    text = HOL_DATA_TYPE_PATH.read_text()
    match = re.search(r"SMPLH_DEMO_JOINTS = \[(.*?)\n\]", text, re.S)
    if not match:
        raise RuntimeError(f"Could not find SMPLH_DEMO_JOINTS in {HOL_DATA_TYPE_PATH}")
    return ast.literal_eval("[" + match.group(1) + "\n]")


def load_tracked_mapping() -> tuple[list[str], list[int]]:
    text = ENV_CFG_PATH.read_text()
    names_match = re.search(r"trackedBodyNames:\s*\[(.*?)\]", text, re.S)
    idx_match = re.search(r"trackedBodyHumanJointIndices:\s*\[(.*?)\]", text, re.S)
    if not names_match or not idx_match:
        raise RuntimeError(f"Could not find tracked body mapping in {ENV_CFG_PATH}")
    names = ast.literal_eval("[" + names_match.group(1) + "]")
    indices = ast.literal_eval("[" + idx_match.group(1) + "]")
    return names, indices


def expected_mapping_from_holosoma() -> dict[str, str]:
    return {
        "pelvis": "Pelvis",
        "torso_link": "Torso",
        "left_knee_link": "L_Knee",
        "right_knee_link": "R_Knee",
        "left_ankle_roll_link": "L_Ankle",
        "right_ankle_roll_link": "R_Ankle",
        "left_shoulder_roll_link": "L_Shoulder",
        "right_shoulder_roll_link": "R_Shoulder",
        "left_elbow_link": "L_Elbow",
        "right_elbow_link": "R_Elbow",
    }


def check_raw_file(path: Path) -> list[str]:
    issues: list[str] = []
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        required = {"qpos", "human_joints", "fps"}
        missing = sorted(required - keys)
        if missing:
            issues.append(f"{path.name}: missing raw fields {missing}")
            return issues

        qpos = data["qpos"]
        human_joints = data["human_joints"]
        fps = int(np.array(data["fps"]).reshape(()).item())

        if qpos.ndim != 2:
            issues.append(f"{path.name}: qpos ndim={qpos.ndim}, expected 2")
        elif qpos.shape[1] not in (36, 43):
            issues.append(f"{path.name}: qpos width={qpos.shape[1]}, expected 36 or 43")

        if human_joints.ndim != 3 or human_joints.shape[1:] != (52, 3):
            issues.append(f"{path.name}: human_joints shape={human_joints.shape}, expected [T, 52, 3]")

        if qpos.shape[0] != human_joints.shape[0]:
            issues.append(
                f"{path.name}: qpos frames={qpos.shape[0]} != human_joints frames={human_joints.shape[0]}"
            )

        if fps <= 0:
            issues.append(f"{path.name}: fps={fps}, expected positive")

    return issues


def check_native_file(path: Path) -> list[str]:
    issues: list[str] = []
    with np.load(path, allow_pickle=True) as data:
        required = {
            "joint_names",
            "body_names",
            "root_pos",
            "root_rot",
            "dof_pos",
            "source_human_joints",
        }
        missing = sorted(required - set(data.files))
        if missing:
            issues.append(f"{path.name}: missing native fields {missing}")
            return issues

        if data["dof_pos"].shape[1] != 29:
            issues.append(f"{path.name}: dof_pos width={data['dof_pos'].shape[1]}, expected 29")

        if data["source_human_joints"].shape[1:] != (52, 3):
            issues.append(
                f"{path.name}: source_human_joints shape={data['source_human_joints'].shape}, expected [T, 52, 3]"
            )

        if data["root_pos"].shape[0] != data["source_human_joints"].shape[0]:
            issues.append(
                f"{path.name}: root_pos frames={data['root_pos'].shape[0]} "
                f"!= source_human_joints frames={data['source_human_joints'].shape[0]}"
            )

    return issues


def compare_mapping() -> tuple[list[str], list[str]]:
    smplh_joints = load_smplh_demo_joints()
    tracked_names, tracked_indices = load_tracked_mapping()
    expected = expected_mapping_from_holosoma()
    findings: list[str] = []
    mismatches: list[str] = []

    for body_name, idx in zip(tracked_names, tracked_indices):
        actual_name = smplh_joints[idx]
        expected_name = expected.get(body_name)
        line = f"{body_name}: cfg index {idx} -> {actual_name}"
        if expected_name is not None:
            line += f" | expected {expected_name}"
        findings.append(line)
        if expected_name is not None and actual_name != expected_name:
            mismatches.append(line)

    return findings, mismatches


def run_validation() -> int:
    raw_files = sorted(RAW_DATA_DIR.glob("*.npz"))
    native_files = sorted(NATIVE_DATA_DIR.glob("*.npz"))

    print("Raw files:")
    for path in raw_files:
        print(f"  - {path.name}")

    print("\nNative files:")
    for path in native_files:
        print(f"  - {path.name}")

    raw_issues: list[str] = []
    for path in raw_files:
        raw_issues.extend(check_raw_file(path))

    native_issues: list[str] = []
    for path in native_files:
        native_issues.extend(check_native_file(path))

    findings, mapping_issues = compare_mapping()

    print("\nTracked-body mapping:")
    for line in findings:
        print(f"  - {line}")

    print("\nSummary:")
    if raw_issues:
        print("  raw data issues:")
        for issue in raw_issues:
            print(f"    - {issue}")
    else:
        print("  raw data contract: OK")

    if native_issues:
        print("  native data issues:")
        for issue in native_issues:
            print(f"    - {issue}")
    else:
        print("  native data contract: OK")

    if mapping_issues:
        print("  trackedBodyHumanJointIndices mismatches:")
        for issue in mapping_issues:
            print(f"    - {issue}")
    else:
        print("  trackedBodyHumanJointIndices: OK")

    return 1 if raw_issues or native_issues or mapping_issues else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate native G1 Holosoma data and sparse-body mapping.")
    parser.parse_args()
    raise SystemExit(run_validation())


if __name__ == "__main__":
    main()
