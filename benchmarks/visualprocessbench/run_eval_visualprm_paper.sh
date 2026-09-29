#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VISUALPRM_MODEL_PATH="${VISUALPRM_MODEL_PATH:-VisualPRM/VisualPRM-8B}"
export VPB_LIMIT="${VPB_LIMIT:-0}"
export VPB_THRESHOLD="${VPB_THRESHOLD:-0.85}"
export VPB_DTYPE="${VPB_DTYPE:-bfloat16}"
export VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/visualprm_predictions.jsonl}"

python "${SCRIPT_DIR}/visualprm_paper_eval.py" \
  --model-path "${VISUALPRM_MODEL_PATH}" \
  --threshold "${VPB_THRESHOLD}" \
  --dtype "${VPB_DTYPE}" \
  --limit "${VPB_LIMIT}" \
  --output "${VPB_OUTPUT}" \
  --overwrite
