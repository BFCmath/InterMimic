# How To Run `intermimic-fork`

This is the operational runbook for the current `intermimic-fork` workspace.

It uses the same Isaac Gym shell setup that already works for the other `InterMimic/docs` in this workspace, but the commands below are specific to:

- `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork`

This guide is for the current native G1 teacher path:

- task: `InterMimicG1Native`
- raw input: `InterAct/OMOMO_holosoma_G1/*.npz`
- converted input: `InterAct/OMOMO_holosoma_G1_native/*.npz`

For implementation status and training trace details, see:

- [g1_native_implementation_audit.md](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/docs/g1_native_implementation_audit.md)

## 1. Working Shell Setup

Use the same conda environment pattern that already works for the main InterMimic teacher pipeline.

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate intermimic-gym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If you want to keep training in tmux:

```bash
tmux new -s intermimic-fork
```

Why this setup matters:

- Isaac Gym in this codebase expects a supported Python environment
- the conda env provides the working Python and PyTorch stack
- `LD_LIBRARY_PATH` must include `$CONDA_PREFIX/lib` so Isaac Gym can find `libpython`

Viewer note:

- in this workspace, if `DISPLAY` is not set, the native launcher now auto-adds `--headless`
- this avoids Isaac Gym viewer-related segmentation faults in container or remote-shell sessions

## 2. Preconditions

Before launching, this fork expects all of the following:

- raw Holosoma retarget data exists in `InterAct/OMOMO_holosoma_G1`
- the sibling `holosoma` repo exists inside this fork
- the Holosoma G1 URDF exists at:
  - `holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf`

The default native env config is:

- [omomo_g1_native.yaml](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml)

Important defaults there:

- `motion_file: InterAct/OMOMO_holosoma_G1_native`
- `robotType: 'g1_29dof.urdf'`
- `assetRoot: "../holosoma/src/holosoma/holosoma/data/robots/g1"`
- `numEnvs: 512`

Why `512` and not `2048`:

- in this workspace, the native G1 path builds and starts training at `512`
- the previous `2048` default was able to segfault during Isaac Gym env creation on this machine
- once the path is stable, you can scale back up deliberately

## 3. Preflight Check

Before debugging training, confirm Isaac Gym imports in the active shell:

```bash
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"
python -c "from isaacgym import gymapi; import intermimic; print('isaac gym ok')"
```

If this fails, stop there and fix the environment first.

## 4. Convert The Holosoma Dataset

If the native dataset has not been generated yet, run:

```bash
sh isaacgym/scripts/convert_g1_native_dataset.sh
```

Default behavior:

- input: `InterAct/OMOMO_holosoma_G1`
- output: `InterAct/OMOMO_holosoma_G1_native`

Custom input/output:

```bash
sh isaacgym/scripts/convert_g1_native_dataset.sh \
  InterAct/OMOMO_holosoma_G1 \
  InterAct/OMOMO_holosoma_G1_native \
  --overwrite
```

What this does:

- reads Holosoma `qpos`
- converts quaternion convention from MuJoCo `wxyz` to Isaac `xyzw`
- writes native G1 fields like `root_pos`, `root_rot`, `dof_pos`, `dof_vel`, and optional object state

## 5. Launch Native G1 Teacher Training

Default launch:

```bash
sh isaacgym/scripts/train_g1_native.sh
```

If no display server is available, that command will automatically run headless.

Launch with an explicit run name:

```bash
sh isaacgym/scripts/train_g1_native.sh my_run_name
```

This launches:

- task: `InterMimicG1Native`
- env config: `isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml`
- RL config: `isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_native.yaml`

If you pass `my_run_name` as the first positional argument, the launcher rewrites the runtime train config so:

- experiment folder becomes `output/my_run_name`
- checkpoint base name becomes `my_run_name`
- checkpoints are saved under `output/my_run_name/nn`

## 6. Useful Launch Variants

Small smoke test:

```bash
sh isaacgym/scripts/train_g1_native.sh --headless --num_envs 64 --minibatch_size 2048 --max_iterations 1
```

Named smoke test:

```bash
sh isaacgym/scripts/train_g1_native.sh debug_native_g1 --headless --num_envs 64 --minibatch_size 2048 --max_iterations 1
```

Headless smoke test:

```bash
sh isaacgym/scripts/train_g1_native.sh --headless --num_envs 64 --minibatch_size 2048 --max_iterations 1
```

Run a smaller debug launch with a fixed seed:

```bash
sh isaacgym/scripts/train_g1_native.sh --headless --num_envs 128 --minibatch_size 4096 --max_iterations 10 --seed 42
```

Override the motion directory from CLI:

```bash
sh isaacgym/scripts/train_g1_native.sh \
  --motion_file InterAct/OMOMO_holosoma_G1_native \
  --headless
```

Override PPO rollout and minibatch size from CLI:

```bash
sh isaacgym/scripts/train_g1_native.sh \
  --horizon_length 16 \
  --minibatch_size 4096
```

Recommended first real launch in this workspace:

```bash
sh isaacgym/scripts/train_g1_native.sh --headless
```

Recommended scale-up pattern:

1. `--num_envs 64 --minibatch_size 2048 --max_iterations 1`
2. `--num_envs 128 --minibatch_size 4096`
3. default `512`

## 6.5 Record A Checkpoint To MP4

The fork now has a native validation path that mirrors the working main-repo flow:

1. load a saved `InterMimicG1Native` checkpoint in Isaac Gym
2. force one chosen native G1 clip
3. dump the rollout as MuJoCo-style `qpos`
4. render that dump to `.mp4` with headless MuJoCo in the `hdmi` env

Run this from the same `intermimic-gym` shell used for training:

```bash
sh isaacgym/scripts/record_g1_native.sh \
  --checkpoint output/fork_1/nn/fork_1.pth \
  --reference_motion InterAct/OMOMO_holosoma_G1_native/sub3_largebox_003_original.npz \
  --output output/fork_1/sub3_largebox_003.mp4
```

Important notes:

- `--reference_motion` can be either a full native `.npz` path or just a sequence name such as `sub3_largebox_003_original`
- if `--max_steps` is omitted, the recorder uses the chosen clip length automatically
- the rollout dump is also saved next to the video as `output/fork_1/sub3_largebox_003.npz`
- the final render subprocess uses conda env `hdmi` by default, because that is the working OSMesa MuJoCo environment on this host

If the `hdmi` env is not visible from your current shell, activate conda by path first:

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/etc/profile.d/conda.sh
conda activate /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/hdmi
```

## 7. What The Launcher Sets Up

The launcher script:

- changes nothing in your conda env
- accepts an optional first positional argument as `run_name`
- prepends these paths to `PYTHONPATH`:
  - repo root
  - `isaacgym/src`
- runs:

```bash
python -m intermimic.run \
  --task InterMimicG1Native \
  --cfg_env isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml \
  --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_native.yaml
```

When `run_name` is provided, the launcher first creates a temporary train config with:

- `params.config.name = run_name`
- `params.config.full_experiment_name = run_name`

## 8. Expected Startup Signals

If launch is correct, you should see:

- seed print from `set_seed`
- task/env creation logs
- `num_envs`, `num_actions`, `num_obs`, `num_states`
- for this path, `num_actions` should be `29`
- for this path, `num_obs` should be `257`

Those come from:

- [run.py](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/run.py)

## 9. Where Outputs Go

By default, `intermimic.run` writes under:

- `output/`

You can override that with:

```bash
sh isaacgym/scripts/train_g1_native.sh --output_path output/g1_native_debug
```

If you use the new run-name form:

```bash
sh isaacgym/scripts/train_g1_native.sh my_run_name
```

then the main checkpoint path becomes:

- `output/my_run_name/nn/my_run_name`

and TensorBoard summaries go to:

- `output/my_run_name/summaries`

## 10. Common Failure Modes

### `ImportError` or Isaac Gym cannot import

Likely cause:

- wrong shell
- wrong Python version
- `intermimic-gym` not activated
- `LD_LIBRARY_PATH` not exported

Fix:

```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate intermimic-gym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"
python -c "from isaacgym import gymapi; print('isaac gym ok')"
```

### `FileNotFoundError` for native G1 dataset root

Cause:

- `InterAct/OMOMO_holosoma_G1_native` does not exist yet
- or `--motion_file` points to the wrong directory

Fix:

```bash
sh isaacgym/scripts/convert_g1_native_dataset.sh --overwrite
```

Then retry training.

### `FileNotFoundError` for G1 URDF or mesh assets

Cause:

- the sibling `holosoma` repo is missing
- or the relative path in `omomo_g1_native.yaml` is broken

Expected local path:

- `holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf`

### Segmentation fault when launching without `--headless`

Cause:

- Isaac Gym viewer path in a shell/container without a valid display

Fix:

- use `--headless`
- or rely on the launcher auto-headless behavior when `DISPLAY` is unset

### Training starts but quickly collapses

That is no longer a shell/setup issue. Then inspect:

- reward weights in [omomo_g1_native.yaml](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml)
- dataset quality in `InterAct/OMOMO_holosoma_G1`
- reset thresholds in [g1_native_spec.py](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/g1_native_spec.py)
- the current implementation limits documented in [g1_native_implementation_audit.md](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/docs/g1_native_implementation_audit.md)

### `AssertionError` from PPO batch size when using very small `--num_envs`

Cause:

- PPO requires `batch_size % minibatch_size == 0`
- `batch_size = num_envs * horizon_length`
- with the default `horizon_length = 32`, small `num_envs` values require a smaller `--minibatch_size`

Examples:

- `num_envs=64` -> batch size `2048` -> use `--minibatch_size 2048`
- `num_envs=128` -> batch size `4096` -> use `--minibatch_size 4096`
- `num_envs=512` -> batch size `16384` -> default `minibatch_size 16384` already works

## 11. Minimal Copy-Paste Sequence

If you just want the exact commands:

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate intermimic-gym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/isaacgym/src:$PWD:$PYTHONPATH"
python -c "from isaacgym import gymapi; print('isaac gym ok')"
sh isaacgym/scripts/convert_g1_native_dataset.sh --overwrite
sh isaacgym/scripts/train_g1_native.sh debug_native_g1 --headless --num_envs 64 --minibatch_size 2048 --max_iterations 1
```

After that smoke test passes, remove the debug flags and launch the real run:

```bash
sh isaacgym/scripts/train_g1_native.sh
```
