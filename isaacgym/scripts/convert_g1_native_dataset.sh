#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

INPUT_DIR="${1:-InterAct/OMOMO_holosoma_G1}"
OUTPUT_DIR="${2:-InterAct/OMOMO_holosoma_G1_native}"

if [ $# -gt 0 ]; then
    shift
fi
if [ $# -gt 0 ]; then
    shift
fi

python -m intermimic.utils.convert_g1_native_dataset \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    "$@"
