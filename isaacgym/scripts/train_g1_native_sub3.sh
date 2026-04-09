#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CFG_ENV="isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml"
DEFAULT_CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_native.yaml"
DEFAULT_RUN_NAME="g1_native_sub3"
DEFAULT_OUTPUT_PATH="output/sub3/"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

FAKE_REWARD=0
HDMI_REWARD=0
PASSTHRU_ARGS=""
append_passthru_arg() {
    if [ -z "$PASSTHRU_ARGS" ]; then
        PASSTHRU_ARGS="$1"
    else
        PASSTHRU_ARGS="${PASSTHRU_ARGS}
$1"
    fi
}
while [ $# -gt 0 ]; do
    case "$1" in
        --fake-reward)
            FAKE_REWARD=1
            ;;
        --hdmi-reward)
            HDMI_REWARD=1
            ;;
        *)
            append_passthru_arg "$1"
            ;;
    esac
    shift
done
if [ -n "$PASSTHRU_ARGS" ]; then
    OLD_IFS=$IFS
    IFS='
'
    set -- $PASSTHRU_ARGS
    IFS=$OLD_IFS
else
    set --
fi

CHECKPOINT_SET=0
for arg in "$@"; do
    if [ "$arg" = "--checkpoint" ]; then
        CHECKPOINT_SET=1
        break
    fi
done

RUN_NAME="$DEFAULT_RUN_NAME"
if [ $# -gt 0 ]; then
    case "$1" in
        -*) ;;
        *)
            RUN_NAME="$1"
            shift
            ;;
    esac
fi

HEADLESS_SET=0
for arg in "$@"; do
    if [ "$arg" = "--headless" ]; then
        HEADLESS_SET=1
        break
    fi
done

AUTO_HEADLESS=""
if [ "$HEADLESS_SET" -eq 0 ] && [ -z "${DISPLAY:-}" ]; then
    AUTO_HEADLESS="--headless"
    echo "[train_g1_native_sub3] DISPLAY is not set; forcing --headless to avoid Isaac Gym viewer segfaults"
fi

TMP_CFG_ENV=""
TMP_CFG_TRAIN=""
RESUME_PATH="$REPO_ROOT/$DEFAULT_OUTPUT_PATH$RUN_NAME/nn/$RUN_NAME.pth"
AUTO_RESUME_ARGS=""
is_valid_checkpoint() {
    checkpoint_path="$1"
    [ -f "$checkpoint_path" ] || return 1
    python - "$checkpoint_path" <<'PY' >/dev/null 2>&1
import sys
import torch

path = sys.argv[1]
torch.load(path, map_location='cpu')
PY
}
cleanup() {
    if [ -n "$TMP_CFG_ENV" ] && [ -f "$TMP_CFG_ENV" ]; then
        rm -f "$TMP_CFG_ENV"
    fi
    if [ -n "$TMP_CFG_TRAIN" ] && [ -f "$TMP_CFG_TRAIN" ]; then
        rm -f "$TMP_CFG_TRAIN"
    fi
}
trap cleanup EXIT INT TERM

TMP_CFG_ENV="$(mktemp "${TMPDIR:-/tmp}/omomo_g1_native.sub3.XXXXXX.yaml")"
python - "$REPO_ROOT/$DEFAULT_CFG_ENV" "$TMP_CFG_ENV" "$FAKE_REWARD" "$HDMI_REWARD" <<'PY'
import sys
import yaml

src_path, dst_path, fake_reward, hdmi_reward = sys.argv[1:5]
with open(src_path, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['env']['dataSub'] = ['sub3']
cfg['env']['fakeReward'] = bool(int(fake_reward))
cfg['env']['hdmiReward'] = bool(int(hdmi_reward))

with open(dst_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

TMP_CFG_TRAIN="$(mktemp "${TMPDIR:-/tmp}/omomo_g1_native.sub3.${RUN_NAME}.XXXXXX.yaml")"
python - "$REPO_ROOT/$DEFAULT_CFG_TRAIN" "$TMP_CFG_TRAIN" "$RUN_NAME" <<'PY'
import sys
import yaml

src_path, dst_path, run_name = sys.argv[1:4]
with open(src_path, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['params']['config']['name'] = run_name
cfg['params']['config']['full_experiment_name'] = run_name
cfg['params']['load_checkpoint'] = False
cfg['params']['config']['resume_from'] = 'None'

with open(dst_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "[train_g1_native_sub3] run_name=$RUN_NAME"
echo "[train_g1_native_sub3] dataSub=['sub3']"
echo "[train_g1_native_sub3] fakeReward=$FAKE_REWARD"
echo "[train_g1_native_sub3] hdmiReward=$HDMI_REWARD"
echo "[train_g1_native_sub3] checkpoint_dir=$DEFAULT_OUTPUT_PATH$RUN_NAME/nn"
if [ "$CHECKPOINT_SET" -eq 0 ] && is_valid_checkpoint "$RESUME_PATH"; then
    AUTO_RESUME_ARGS="--checkpoint $RESUME_PATH --resume 1"
    echo "[train_g1_native_sub3] auto_resume=$RESUME_PATH"
elif [ "$CHECKPOINT_SET" -eq 0 ] && [ -f "$RESUME_PATH" ]; then
    echo "[train_g1_native_sub3] skip_corrupt_resume=$RESUME_PATH"
fi

python -m intermimic.run \
    --task InterMimicG1Native \
    --cfg_env "$TMP_CFG_ENV" \
    --cfg_train "$TMP_CFG_TRAIN" \
    --output_path "$DEFAULT_OUTPUT_PATH" \
    $AUTO_RESUME_ARGS \
    $AUTO_HEADLESS \
    "$@"
