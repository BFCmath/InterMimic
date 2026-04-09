#!/usr/bin/env python
"""
Record a trained InterMimicG1Native policy to MP4.

Flow:
1. load a checkpoint in Isaac Gym,
2. reset onto a selected native G1 reference clip,
3. dump rollout qpos in MuJoCo layout,
4. render that dump headlessly via MuJoCo in the hdmi env.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for path in [REPO_ROOT, os.path.join(REPO_ROOT, "isaacgym", "src")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaacgym import gymapi  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from intermimic import g1_native_spec  # noqa: E402
from intermimic import run as intermimic_run  # noqa: E402
from intermimic.run import RLGPUAlgoObserver, build_alg_runner  # noqa: E402
from intermimic.utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402
from intermimic.utils.g1_native_dataset import discover_g1_native_sequences  # noqa: E402
from intermimic.utils.path_utils import resolve_repo_path  # noqa: E402


def pop_recording_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--reference_motion",
        type=str,
        default="",
        help="Native reference clip path or sequence name to force during rollout",
    )
    parser.add_argument("--motion_file", type=str, default="")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum rollout steps. Defaults to the chosen clip length when omitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--dump_only", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render_env", type=str, default="hdmi")
    parser.add_argument(
        "--conda_bin",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/bin/conda",
    )
    rec_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return rec_args


def ensure_flag(flag, value=None):
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def reference_motion_dir(reference_motion: str, motion_file_override: str) -> str:
    if motion_file_override:
        return motion_file_override
    if reference_motion:
        candidate = Path(reference_motion)
        if not candidate.is_absolute():
            candidate = resolve_repo_path(reference_motion)
        if candidate.is_file():
            return str(candidate.parent)
    return "InterAct/OMOMO_holosoma_G1_native"


def normalize_sequence_name(reference_motion: str) -> str:
    if not reference_motion:
        return ""
    ref = Path(reference_motion)
    if ref.suffix == ".npz":
        return ref.stem
    return reference_motion


def build_player(rec_args):
    defaults = {
        "--task": "InterMimicG1Native",
        "--cfg_env": "isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml",
        "--cfg_train": "isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_native.yaml",
        "--num_envs": "1",
        "--checkpoint": rec_args.checkpoint,
        "--motion_file": reference_motion_dir(rec_args.reference_motion, rec_args.motion_file),
    }
    for flag, value in defaults.items():
        ensure_flag(flag, value)

    ensure_flag("--headless")
    ensure_flag("--test")

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["numEnvs"] = 1
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    cfg_train["params"]["seed"] = set_seed(rec_args.seed, cfg_train["params"].get("torch_deterministic", False))
    cfg_train["params"]["config"]["multi_gpu"] = False
    cfg_train["params"]["config"]["num_actors"] = 1

    intermimic_run.args = args
    intermimic_run.cfg = cfg
    intermimic_run.cfg_train = cfg_train

    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    player = runner.create_player()
    player.restore(rec_args.checkpoint)
    return player


def resolve_sequence_index(task, reference_motion: str) -> int:
    if not reference_motion:
        return 0
    wanted = normalize_sequence_name(reference_motion)
    for idx, name in enumerate(task.motion_lib.sequence_names):
        if name == wanted:
            return idx
    raise ValueError(
        f"Sequence {wanted!r} not found in motion dataset. "
        f"Available: {task.motion_lib.sequence_names}"
    )


def force_sequence_reset(player, sequence_idx: int, start_frame: int = 0):
    task = player.env.task
    env_ids = torch.tensor([0], device=task.device, dtype=torch.long)
    sequence_ids = torch.tensor([sequence_idx], device=task.device, dtype=torch.long)
    frame_ids = torch.tensor([start_frame], device=task.device, dtype=torch.long)
    ref = task.motion_lib.get_state(sequence_ids, frame_ids)

    task.data_id[env_ids] = sequence_ids
    task.dataset_id[env_ids] = task.motion_lib.subject_indices[sequence_ids]
    task.progress_buf[env_ids] = frame_ids
    task.start_times[env_ids] = frame_ids
    task.prev_actions[env_ids] = 0.0
    task._track_reset_mask[env_ids] = False
    task._set_env_state(
        env_ids,
        ref["root_pos"],
        ref["root_rot"],
        ref["dof_pos"],
        ref["root_lin_vel"],
        ref["root_ang_vel"],
        ref["dof_vel"],
    )
    task._reset_target(env_ids)
    task._reset_env_tensors(env_ids)
    task._compute_hoi_observations(env_ids)
    task._compute_observations(env_ids)


def get_dump_path(rec_args):
    if rec_args.dump_path:
        return Path(rec_args.dump_path)
    return Path(rec_args.output).with_suffix(".npz")


def xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[[3, 0, 1, 2]]


def current_qpos(task):
    state = task._gather_current_state()
    root_pos = state["root_pos"][0].detach().cpu()
    root_rot = xyzw_to_wxyz(state["root_rot"][0].detach().cpu())
    dof_pos = state["dof_pos"][0].detach().cpu()
    object_mask = bool(state["object_mask"][0, 0].item() > 0.5)

    parts = [root_pos, root_rot, dof_pos]
    if object_mask:
        object_pos = state["object_pos"][0].detach().cpu()
        object_rot = xyzw_to_wxyz(state["object_rot"][0].detach().cpu())
        parts.extend([object_pos, object_rot])
    qpos = torch.cat(parts, dim=0).numpy().astype(np.float32)
    expected = 43 if object_mask else 36
    if qpos.shape[0] != expected:
        raise ValueError(f"Expected qpos width {expected}, got {qpos.shape[0]}")
    return qpos


def render_rollout(rec_args, dump_path):
    conda_bin = Path(rec_args.conda_bin)
    if not conda_bin.is_file():
        raise FileNotFoundError(f"Conda executable not found: {conda_bin}")

    render_script = Path(REPO_ROOT) / "render_g1_native_true.py"
    cmd = [
        str(conda_bin),
        "run",
        "-n",
        rec_args.render_env,
        "python",
        str(render_script),
        "--motion",
        str(dump_path),
        "--output",
        rec_args.output,
        "--fps",
        str(rec_args.fps),
        "--width",
        str(rec_args.width),
        "--height",
        str(rec_args.height),
    ]
    print("Render command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def record_policy(player, rec_args):
    dump_path = get_dump_path(rec_args)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    Path(rec_args.output).parent.mkdir(parents=True, exist_ok=True)

    obs_dict = player.env_reset()
    batch_size = player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()

    sequence_idx = resolve_sequence_index(player.env.task, rec_args.reference_motion)
    force_sequence_reset(player, sequence_idx, start_frame=0)
    selected_name = player.env.task.motion_lib.sequence_names[sequence_idx]
    selected_len = int(player.env.task.max_episode_length[sequence_idx].item())
    with np.load(player.env.task.motion_lib.sequence_infos[sequence_idx].path, allow_pickle=True) as data:
        clip_fps = int(data["fps"]) if "fps" in data else rec_args.fps
    max_steps = rec_args.max_steps if rec_args.max_steps > 0 else selected_len
    player.max_steps = max_steps
    obs_dict = {"obs": player.env.task.obs_buf}

    print(f"[record_g1_native] selected sequence: {selected_name}")
    print(f"[record_g1_native] clip length: {selected_len}")
    print(f"[record_g1_native] rollout steps: {max_steps}")
    print(f"[record_g1_native] clip fps: {clip_fps}")

    cr = torch.zeros(batch_size, dtype=torch.float32, device=player.device)
    steps = torch.zeros(batch_size, dtype=torch.float32, device=player.device)
    done_indices = []
    qpos_frames = [current_qpos(player.env.task)]

    for step_idx in range(max_steps):
        obs_dict = player.env_reset(done_indices)
        action = player.get_action(obs_dict, player.is_determenistic)
        obs_dict, reward, done, info = player.env_step(player.env, action)
        player._post_step(info)

        cr += reward
        steps += 1
        qpos_frames.append(current_qpos(player.env.task))

        all_done_indices = done.nonzero(as_tuple=False)
        done_indices = all_done_indices[::player.num_agents]
        done_count = len(done_indices)

        if done_count > 0:
            cur_rewards = cr[done_indices].sum().item()
            cur_steps = steps[done_indices].sum().item()
            if player.print_stats:
                print(f"reward: {cur_rewards / done_count} steps: {cur_steps / done_count}")
            break

        if done_indices.numel() > 0:
            done_indices = done_indices[:, 0]

        if (step_idx + 1) % 50 == 0:
            print(f"Recorded {step_idx + 1}/{max_steps} rollout steps")

    qpos = np.stack(qpos_frames, axis=0)
    np.savez_compressed(
        dump_path,
        qpos=qpos,
        fps=np.array(clip_fps, dtype=np.int32),
        sequence_name=np.array(selected_name),
    )
    print(f"[record_g1_native] Saved rollout dump to {dump_path}")

    if not rec_args.dump_only:
        render_rollout(rec_args, dump_path)


def main():
    rec_args = pop_recording_args()
    player = build_player(rec_args)
    record_policy(player, rec_args)


if __name__ == "__main__":
    main()
