from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse
from pathlib import Path

import imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = (
    REPO_ROOT
    / "holosoma"
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
)


def infer_object_name_from_motion(npz_path: str) -> str:
    stem = Path(npz_path).stem
    if stem.endswith("_original"):
        stem = stem[: -len("_original")]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer object name from motion file: {npz_path}")
    return parts[1]


def _load_qpos_and_fps(npz_path: str):
    data = np.load(npz_path)
    if "qpos" not in data:
        raise KeyError(f"{npz_path} does not contain 'qpos'")
    qpos = data["qpos"]
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos shape (T, nq), got {qpos.shape}")
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def resolve_scene_xml(npz_path: str, xml_path: str | None) -> Path:
    if xml_path:
        return Path(xml_path).resolve()

    import mujoco

    qpos, _ = _load_qpos_and_fps(npz_path)
    qpos_width = int(qpos.shape[1])
    object_name = infer_object_name_from_motion(npz_path)
    candidates = [
        MODELS_ROOT / f"g1_29dof_w_{object_name}.xml",
        MODELS_ROOT / "g1_29dof.xml",
    ]
    candidates = [path for path in candidates if path.is_file()]

    for candidate in candidates:
        model = mujoco.MjModel.from_xml_path(str(candidate))
        if model.nq == qpos_width:
            return candidate
        if candidate.name == "g1_29dof.xml" and qpos_width == model.nq + 7:
            return candidate

    raise FileNotFoundError(
        f"No compatible MuJoCo scene XML found for qpos width {qpos_width}. "
        f"Tried: {', '.join(str(path) for path in candidates) or 'none'}"
    )


def render_native_qpos(npz_path, output_path, xml_path, fps=None, width=1280, height=720):
    import mujoco

    print(f"[render_g1_native_true] Loading motion: {npz_path}", flush=True)
    qpos, file_fps = _load_qpos_and_fps(npz_path)
    fps = int(fps) if fps is not None else file_fps

    print(f"[render_g1_native_true] Loading scene XML: {xml_path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if qpos.shape[1] == model.nq + 7:
        print(
            "[render_g1_native_true] Motion includes object pose but XML is robot-only; "
            "dropping the final 7 object entries.",
            flush=True,
        )
        qpos = qpos[:, : model.nq]
    elif qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {qpos.shape[1]} does not match model.nq {model.nq}")

    print(
        f"[render_g1_native_true] Frames={qpos.shape[0]} nq={model.nq} fps={fps} size={width}x{height}",
        flush=True,
    )
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 140
    camera.elevation = -18
    camera.lookat[:] = [0.8, 0.0, 0.9]

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[render_g1_native_true] Writing video: {output}", flush=True)

    with imageio.get_writer(output, fps=fps, codec="libx264") as writer:
        for idx in range(qpos.shape[0]):
            mj_data.qpos[:] = qpos[idx]
            mujoco.mj_forward(model, mj_data)

            if pelvis_id >= 0:
                camera.lookat[:] = mj_data.xpos[pelvis_id]
                camera.lookat[2] += 0.1

            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render().copy())

            if idx == 0 or (idx + 1) % 25 == 0 or (idx + 1) == qpos.shape[0]:
                print(
                    f"[render_g1_native_true] Rendered frame {idx + 1}/{qpos.shape[0]}",
                    flush=True,
                )

    renderer.close()
    print(f"[render_g1_native_true] Saved video to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Render native G1 rollout qpos npz to MP4 in headless MuJoCo.")
    parser.add_argument("--motion", required=True, help="Path to recorded rollout .npz containing qpos")
    parser.add_argument("--xml", default=None, help="Optional MuJoCo XML path override")
    parser.add_argument("--output", default="recordings/g1_native.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    xml_path = resolve_scene_xml(args.motion, args.xml)
    render_native_qpos(
        npz_path=args.motion,
        output_path=args.output,
        xml_path=xml_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
