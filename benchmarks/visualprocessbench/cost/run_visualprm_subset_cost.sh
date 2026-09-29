#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

VPB_LIMIT="${VPB_LIMIT:-32}"
VPB_SAMPLE_STRATEGY="${VPB_SAMPLE_STRATEGY:-random}"
VPB_SEED="${VPB_SEED:-42}"
VPB_SELECTED_IDS_FILE="${VPB_SELECTED_IDS_FILE:-}"
VPB_COST_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compute_cost}"
VPB_RUN_NAME="${VPB_RUN_NAME:-think_ablation_cost_$(date +%Y%m%d_%H%M%S)}"
VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
VPB_LOG_EVERY="${VPB_LOG_EVERY:-10}"

VISUALPRM_MODEL_PATH="${VISUALPRM_MODEL_PATH:-VisualPRM/VisualPRM-8B}"
VISUALPRM_DTYPE="${VISUALPRM_DTYPE:-${VPB_DTYPE:-bfloat16}}"
VISUALPRM_THRESHOLD="${VISUALPRM_THRESHOLD:-${VPB_THRESHOLD:-0.85}}"

ARGS=(
  --methods "visualprm"
  --limit "${VPB_LIMIT}"
  --sample-strategy "${VPB_SAMPLE_STRATEGY}"
  --seed "${VPB_SEED}"
  --output-dir "${VPB_COST_OUTPUT_DIR}"
  --run-name "${VPB_RUN_NAME}"
  --log-every "${VPB_LOG_EVERY}"
  --visualprm-model-path "${VISUALPRM_MODEL_PATH}"
  --visualprm-dtype "${VISUALPRM_DTYPE}"
  --visualprm-threshold "${VISUALPRM_THRESHOLD}"
)

if [[ -n "${VPB_BENCHMARK_DIR:-}" ]]; then
  ARGS+=(--benchmark-dir "${VPB_BENCHMARK_DIR}")
fi

if [[ -n "${VPB_SELECTED_IDS_FILE}" ]]; then
  ARGS+=(--selected-ids-file "${VPB_SELECTED_IDS_FILE}")
fi

if [[ "${VPB_OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "${VPB_NO_PROGRESS}" == "1" ]]; then
  ARGS+=(--no-progress)
fi

python "${SCRIPT_DIR}/benchmark_think_ablation_cost.py" "${ARGS[@]}"
