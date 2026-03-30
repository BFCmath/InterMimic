#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_native.yaml"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

RUN_NAME=""
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
    echo "[train_g1_native] DISPLAY is not set; forcing --headless to avoid Isaac Gym viewer segfaults"
fi

CFG_TRAIN_PATH="$DEFAULT_CFG_TRAIN"
TMP_CFG_TRAIN=""
cleanup() {
    if [ -n "$TMP_CFG_TRAIN" ] && [ -f "$TMP_CFG_TRAIN" ]; then
        rm -f "$TMP_CFG_TRAIN"
    fi
}
trap cleanup EXIT INT TERM

if [ -n "$RUN_NAME" ]; then
    TMP_CFG_TRAIN="$(mktemp "${TMPDIR:-/tmp}/omomo_g1_native.${RUN_NAME}.XXXXXX.yaml")"
    python - "$REPO_ROOT/$DEFAULT_CFG_TRAIN" "$TMP_CFG_TRAIN" "$RUN_NAME" <<'PY'
import sys
import yaml

src_path, dst_path, run_name = sys.argv[1:4]
with open(src_path, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['params']['config']['name'] = run_name
cfg['params']['config']['full_experiment_name'] = run_name

with open(dst_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    CFG_TRAIN_PATH="$TMP_CFG_TRAIN"
    echo "[train_g1_native] run_name=$RUN_NAME"
    echo "[train_g1_native] checkpoint_dir=output/$RUN_NAME/nn"
fi

python -m intermimic.run \
    --task InterMimicG1Native \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_g1_native.yaml \
    --cfg_train "$CFG_TRAIN_PATH" \
    $AUTO_HEADLESS \
    "$@"
