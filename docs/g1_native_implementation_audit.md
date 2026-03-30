# InterMimic Fork: Current G1-Native Implementation Audit

This document traces what is implemented so far in `intermimic-fork` for the Holosoma-based `G1 -> G1` path, and what is still inherited from the original `SMPL-X -> SMPL-X` InterMimic design.

It focuses on the current fork state under:

- `isaacgym/src/intermimic/run.py`
- `isaacgym/src/intermimic/utils/parse_task.py`
- `isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py`
- `isaacgym/src/intermimic/utils/convert_g1_native_dataset.py`
- `isaacgym/src/intermimic/utils/g1_native_dataset.py`
- `isaacgym/src/intermimic/g1_native_spec.py`
- `isaacgym/src/intermimic/learning/intermimic_agent.py`

## 1. Executive Status

Short version:

- The fork does implement a real native `G1 -> G1` teacher PPO path.
- It is not just a renamed SMPL-X pipeline.
- The new path removes the old 153-DoF SMPL-X tensor contract for observations, rewards, and resets.
- The trainer is still the same PPO trainer used by the existing codebase.
- The native path currently supervises only root pose/velocity, joint pose/velocity, and optional object pose.
- Dense G1 body tracking, contact labels, and interaction-graph supervision are not yet present in the native dataset/runtime path.

So the author’s statement is still directionally true for the original codebase, but this fork has already started breaking that limitation by adding a new native branch. The branch is usable as a first-stage `teacher PPO on G1 states`, but it is not yet a full feature-equivalent replacement for the original OMOMO SMPL-X supervision.

## 2. What Is Implemented So Far

### 2.1 New native task registration

Implemented:

- `InterMimicG1Native` is imported and registered in [`isaacgym/src/intermimic/utils/parse_task.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/parse_task.py#L29).
- `train_g1_native.sh` launches `python -m intermimic.run --task InterMimicG1Native ...`.

Meaning:

- The fork has a dedicated runtime path, not a temporary local script hack.

### 2.2 Native G1 dataset conversion

Implemented:

- Raw Holosoma files in `InterAct/OMOMO_holosoma_G1/*.npz` contain `qpos`, `human_joints`, `fps`, `cost`.
- The converter in [`isaacgym/src/intermimic/utils/convert_g1_native_dataset.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/convert_g1_native_dataset.py) transforms raw Holosoma `qpos` into a typed native dataset:
  - `root_pos`
  - `root_rot`
  - `dof_pos`
  - `root_lin_vel`
  - `root_ang_vel`
  - `dof_vel`
  - optional object state
  - provenance fields such as `source_human_joints` and `source_cost`
- Quaternion convention is explicitly normalized from Holosoma MuJoCo `wxyz` to Isaac Gym `xyzw`.

Meaning:

- The fork already has a clean serialization boundary between retarget output and PPO training input.

### 2.3 Native dataset loader with schema checks

Implemented in [`isaacgym/src/intermimic/utils/g1_native_dataset.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/g1_native_dataset.py#L67):

- dataset discovery by directory
- per-sequence metadata extraction
- schema version validation
- quaternion convention validation
- exact joint-order validation against `g1_native_spec.G1_29DOF_JOINT_NAMES`
- exact body-name validation against `g1_native_spec.G1_BODY_NAMES`
- zero-padding to max sequence length across the dataset
- indexed access via `get_state(sequence_ids, frame_ids)`

Meaning:

- This is already stronger than the legacy `.pt` loading path, which was mostly positional slicing on a fixed SMPL-X layout.

### 2.4 Native G1 runtime task

Implemented in [`isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L18):

- native observation size = `257`
- action size = `29`
- direct reset from native reference state
- optional object actor loading and target state reset
- reference-conditioned observation construction from:
  - current robot state
  - future reference deltas at offsets `(1, 16)`
- reward terms for:
  - root position
  - root rotation
  - root linear/angular velocity
  - joint position
  - joint velocity
  - action-rate penalty
  - optional object position/rotation
- reset logic driven by:
  - tracking error thresholds
  - environment termination
  - rollout horizon
  - sequence end

Meaning:

- The new task is no longer pretending G1 is SMPL-X.

### 2.5 PPO trainer reuse

Implemented:

- No new algorithm was introduced.
- The fork reuses the existing PPO runner, env wrapper, agent, model, and network builder:
  - [`run.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/run.py#L206)
  - [`intermimic_agent.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/learning/intermimic_agent.py#L471)
  - [`intermimic_network_builder.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/learning/intermimic_network_builder.py)

Meaning:

- The adaptation so far is mostly environment-side and dataset-side, not algorithm-side.

## 3. What Is Not Implemented Yet

These are the main gaps if the target is a strong G1-native replacement for the original OMOMO teacher.

### 3.1 No dense G1 body trajectory supervision in native data

Missing today:

- `body_pos`
- `body_rot`
- `body_pos_vel`
- `body_rot_vel`

Effect:

- The native reward cannot yet mimic the original SMPL-X-style dense body tracking objective.
- Current reward only tracks root + joints + optional object, not all rigid bodies in FK space.

### 3.2 No native contact labels

Missing today:

- per-body contact targets analogous to legacy `contact_human`
- object contact target analogous to legacy `contact_obj`

Effect:

- Native training does not yet explicitly teach contact timing or support/hand-object contact structure.

### 3.3 No native interaction graph supervision

Missing today:

- no converted `ig`
- no native equivalent of the legacy SDF-based interaction graph target in the dataset

Effect:

- The original interaction-centric signal is absent from the native branch.

### 3.4 No native body-name to reward-body FK export from conversion

The converter stores `body_names`, but not per-frame G1 rigid-body transforms.

Effect:

- The task cannot compare simulator body transforms against reference body transforms, because those reference body transforms were never exported into the dataset.

### 3.5 Dataset coverage is currently tiny

Local repo state checked in this fork:

- `InterAct/OMOMO_holosoma_G1`: `2` files
- `InterAct/OMOMO_holosoma_G1_native`: `2` files

Effect:

- The path is structurally implemented, but the checked-in dataset is only a smoke-test scale dataset, not a train-scale corpus.

## 4. Current Training Flow, End to End

This section is the exact high-level pseudocode of the current native teacher path.

### 4.1 Data conversion

Entry:

- `sh isaacgym/scripts/convert_g1_native_dataset.sh`

Pseudocode:

```text
for each raw Holosoma file in InterAct/OMOMO_holosoma_G1/*.npz:
    load qpos [T, 43] or [T, 36] depending on object presence
    split qpos into:
        root_pos = qpos[:, 0:3]
        root_rot_wxyz = qpos[:, 3:7]
        dof_pos = qpos[:, 7:36]
        optional object_pos = qpos[:, 36:39]
        optional object_rot_wxyz = qpos[:, 39:43]

    convert root/object quaternion from MuJoCo wxyz to Isaac xyzw
    finite-difference:
        root_lin_vel
        root_ang_vel
        dof_vel
        object_lin_vel
        object_ang_vel

    save native .npz with schema fields and metadata
```

Main source:

- [`convert_g1_native_dataset.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/convert_g1_native_dataset.py)

### 4.2 Launch and config loading

Entry:

- `sh isaacgym/scripts/train_g1_native.sh`

Pseudocode:

```text
python -m intermimic.run
    --task InterMimicG1Native
    --cfg_env omomo_g1_native.yaml
    --cfg_train train/rlg/omomo_g1_native.yaml
```

Inside [`run.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/run.py#L216):

```text
args = get_args()
cfg, cfg_train = load_cfg(args)
seed = set_seed(...)
apply CLI overrides into cfg/cfg_train
register algo/model/network builders
runner.load(cfg_train)
runner.reset()
runner.run(vargs)
```

### 4.3 Environment creation

Inside [`run.py:create_rlgpu_env`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/run.py#L53):

```text
detect rank / local_rank
bind CUDA device
parse sim params
task, env = parse_task(args, cfg, cfg_train, sim_params)
return env
```

Inside [`parse_task.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/parse_task.py#L39):

```text
task = InterMimicG1Native(...)
env = VecTaskPythonWrapper(task, rl_device, clip_obs, clip_actions)
```

### 4.4 Task initialization

Inside [`InterMimicG1Native.__init__`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L25):

```text
read env config:
    motion_file
    reward weights
    rollout length
    robot type
    object settings
    data subset

discover native sequences from directory
build metadata:
    sequence names
    object names
    object-to-motion mapping

call parent constructor to create Isaac Gym sim and G1 environments

load full native dataset into G1NativeMotionDataset on device
cache:
    num_motions
    max_episode_length
    subject indices
    object ids

allocate runtime buffers:
    dataset_id
    track_reset_mask
    prev_actions
    curr_ref_state
    target tensors
```

### 4.5 Dataset loading

Inside [`G1NativeMotionDataset`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/utils/g1_native_dataset.py#L82):

```text
discover sequence files
for each sequence:
    load native .npz
    validate required fields
    validate schema version
    validate quaternion convention
    validate exact joint order
    validate exact body name order
    load tensors:
        root_pos, root_rot
        root_lin_vel, root_ang_vel
        dof_pos, dof_vel
        optional object state

pad all sequences to the dataset max length
stack tensors into [num_sequences, max_T, dim]
```

### 4.6 Environment reset path

Inside [`InterMimicG1Native._reset_actors`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L251):

```text
sample sequence_id for each resetting env
sample frame_id depending on stateInit:
    Start -> 0
    Random -> random valid frame
    Hybrid -> mixture

ref = motion_lib.get_state(sequence_id, frame_id)

write bookkeeping:
    data_id
    dataset_id
    progress_buf
    start_times
    prev_actions = 0

set simulator state directly from ref:
    root_pos
    root_rot
    dof_pos
    root_lin_vel
    root_ang_vel
    dof_vel

if object exists:
    set target actor state from reference object pose/velocity
else:
    hide object below floor
```

This is a major difference from legacy `InterMimicG1`, which still injected a hard-coded init pose and SMPL-X-shaped shims.

### 4.7 Observation construction

Inside [`InterMimicG1Native._compute_observations`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L368):

Pseudocode:

```text
current_state = gather simulator state:
    root pose
    root velocities
    dof pos/vel
    object pose/vel
    object mask

current_features =
    root height
    local root linear velocity
    local root angular velocity
    dof_pos
    dof_vel
    local object position
    local object rotation (tan-norm)
    local object linear velocity
    local object angular velocity
    object mask

for future offset in (1, 16):
    future_state = reference dataset state at progress + offset
    future_features =
        local root position delta
        root rotation delta
        local root linear velocity delta
        local root angular velocity delta
        dof_pos delta
        dof_vel delta
        local object position delta
        object rotation delta
        local object linear velocity delta
        local object angular velocity delta

obs = concat(current_features, future_features(offset=1), future_features(offset=16))
```

Observation size:

- current block = `81`
- each future block = `88`
- total = `81 + 2 * 88 = 257`

### 4.8 Reference refresh before reward

Inside [`InterMimicG1Native._compute_hoi_observations`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L313):

```text
curr_ref_state = motion_lib.get_state(data_id, progress_buf)
```

Despite the old method name, this is now just the native reference fetch.

### 4.9 Reward computation

Inside [`InterMimicG1Native._compute_reward`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L384):

Pseudocode:

```text
current_state = simulator state
ref_state = current reference frame

root_pos_err = ||ref_root_pos - cur_root_pos||^2
root_rot_err = angular_distance(ref_root_rot, cur_root_rot)
root_vel_err = mse(ref_root_lin_vel, cur_root_lin_vel)
root_ang_vel_err = mse(ref_root_ang_vel, cur_root_ang_vel)
dof_pos_err = mse(ref_dof_pos, cur_dof_pos)
dof_vel_err = mse(ref_dof_vel, cur_dof_vel)
action_rate_err = mse(action_t - action_{t-1})

reward =
    exp(-w_rootPos * root_pos_err)
    * exp(-w_rootRot * root_rot_err)
    * exp(-w_rootVel * (root_vel_err + root_ang_vel_err))
    * exp(-w_dofPos * dof_pos_err)
    * exp(-w_dofVel * dof_vel_err)
    * exp(-w_actionRate * action_rate_err)

if object tracking enabled:
    object_pos_err = ...
    object_rot_err = ...
    reward *= object_reward on object-valid frames

track_reset_mask =
    root_pos too far
    or root_rot too far
    or dof_rmse too far
    or object_pos too far

prev_actions = current_actions
```

### 4.10 Reset decision after step

Inside [`InterMimicG1Native._compute_reset`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1_native.py#L420):

```text
terminated = base humanoid early termination
terminated |= track_reset_mask

horizon_reset =
    progress reaches sequence end
    or rollout_length reached from sampled start

reset_buf = terminated or horizon_reset
terminate_buf = terminated
```

### 4.11 PPO rollout collection

Inside [`InterMimicAgent.play_steps`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/learning/intermimic_agent.py#L320):

Pseudocode:

```text
for t in 0 .. horizon_length - 1:
    obs = env_reset(done_indices)
    store obs

    policy output = model(obs)
        values
        actions
        neglogp
        mu
        sigma

    apply epsilon-greedy deterministic/stochastic action mixing
    env.step(actions)

    if obs contains NaN/Inf:
        zero those obs
        force done and terminate

    store:
        rewards
        next_obs
        done
        rand_action_mask
        next_values

compute GAE advantages
compute returns
flatten rollout tensors into batch_dict
```

### 4.12 PPO forward/backward/update

Inside [`InterMimicAgent.train_epoch`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/learning/intermimic_agent.py#L471) and [`calc_gradients`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/learning/intermimic_agent.py#L551):

Pseudocode:

```text
batch_dict = play_steps()
prepare_dataset(batch_dict)

for mini_epoch in range(mini_epochs):
    for minibatch in dataset:
        optimizer.zero_grad()

        res_dict = model(
            obs=minibatch.obs,
            prev_actions=minibatch.actions
        )

        action_log_probs = res_dict.prev_neglogp
        values = res_dict.values
        entropy = res_dict.entropy
        mu = res_dict.mus
        sigma = res_dict.sigmas

        actor_loss = PPO clipped surrogate loss
        critic_loss = value regression loss
        bounds_loss = action mean soft bound penalty

        total_loss =
            actor_loss
            + critic_coef * critic_loss
            + bounds_loss_coef * bounds_loss

        backward(total_loss)
        optional grad clip
        optimizer.step()

        compute KL for scheduler/logging
```

Important note:

- The native G1 branch does not require algorithm changes because the trainer only sees `obs_dim=257` and `act_dim=29`.

## 5. Comparison Against Legacy `InterMimicG1`

The old G1 path in [`intermimic_g1.py`](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/isaacgym/src/intermimic/env/tasks/intermimic_g1.py) still inherits the old SMPL-X supervision structure:

- pads G1 DoFs into a fake `153`-channel layout
- pads G1 body/contact state into fake `52`-body SMPL-X slots
- computes legacy `ig`
- consumes legacy `body_pos`, `body_rot`, and contact targets from `.pt`

The new native path intentionally does not do that.

This is the key architectural fork:

- `InterMimicG1`: compatibility shim into SMPL-X-shaped supervision
- `InterMimicG1Native`: actual G1-native supervision contract

## 6. Inheritance Plan To Reach a Stronger G1-to-G1 Training Setup

This section is the recommended path that inherits from the current work rather than rewriting the whole stack.

### 6.1 Keep the current trainer unchanged

Do not rewrite PPO first.

Reason:

- The current algorithm stack is already morphology-agnostic.
- The real mismatch problem is in dataset semantics and reward targets, not in PPO.

Keep inheriting:

- `run.py`
- `InterMimicAgent`
- `InterMimicBuilder`

### 6.2 Keep `g1_native_spec.py` as the single contract source

This file should remain the authority for:

- joint order
- body order
- observation layout
- quaternion convention
- schema version
- reset thresholds

If G1 body supervision is added, extend this file first.

### 6.3 Extend the converter, not the trainer, for richer supervision

Next major step should be:

```text
Holosoma output
    -> native converter
    -> native dataset with body/frame supervision
    -> native task reward/obs consume those fields
```

Recommended new dataset fields:

- `body_pos`
- `body_rot`
- `body_lin_vel`
- `body_ang_vel`
- `contact_body`
- `contact_object`
- optional `interaction_graph`

If Holosoma can export FK rigid-body transforms directly, use that.
If not, add an offline FK pass during conversion from `root_pos/root_rot/dof_pos` using the exact Holosoma G1 URDF.

### 6.4 Add a second native reward tier

Current reward tier:

- root + joints + object

Recommended next reward tier:

- dense rigid-body pose tracking
- dense rigid-body velocity tracking
- contact consistency
- optional hand/object relative pose terms

This should be implemented by extending `InterMimicG1Native._compute_reward`, not by reviving the legacy SMPL-X padding trick.

### 6.5 Add native contact semantics before native IG

Order of work:

1. add body FK targets
2. add body contact targets
3. then add interaction graph if still needed

Reason:

- Native body and contact targets are foundational.
- `ig` is useful, but it is secondary if the body/object/contact geometry is already supervised directly.

### 6.6 Split “provenance” from “training target”

Current converter already stores:

- `source_human_joints`
- `source_cost`

Keep this separation:

- human joints are provenance from the retarget process
- G1 rigid-body targets should become the actual training supervision

Do not make the runtime task depend directly on the original 52 human joints unless you are explicitly training cross-morphology alignment again.

## 7. Recommended Next Concrete Tasks

If the goal is practical progress toward a real G1-to-G1 teacher, the next tasks should be:

1. Expand `InterAct/OMOMO_holosoma_G1` beyond the current 2-file smoke set.
2. Extend the converter to export per-frame G1 body transforms and velocities.
3. Extend `G1NativeMotionDataset` schema validation to those new body fields.
4. Extend `InterMimicG1Native` reward with dense body tracking terms.
5. Add optional native contact labels and contact reward/reset logic.
6. Only after that, decide whether native `ig` is still necessary.

## 8. Bottom-Line Conclusion

The fork is already beyond “SMPL-X only” in one important sense:

- there is now a true native `G1 -> G1` teacher path.

But it is only a first native version:

- native data contract exists
- native runtime task exists
- native PPO training path exists
- dense body/contact/interaction supervision does not yet exist

So the current implementation is best described as:

`G1-native teacher PPO with root/joint/object tracking`, not yet `full OMOMO-equivalent G1-native imitation`.
