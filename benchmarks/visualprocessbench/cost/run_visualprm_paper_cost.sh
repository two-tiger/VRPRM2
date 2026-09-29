#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

VPB_LIMIT="${VPB_LIMIT:-0}"
VPB_THRESHOLD="${VPB_THRESHOLD:-${VISUALPRM_THRESHOLD:-0.85}}"
VPB_DTYPE="${VPB_DTYPE:-${VISUALPRM_DTYPE:-bfloat16}}"
VPB_RUN_NAME="${VPB_RUN_NAME:-visualprm_paper_cost_$(date +%Y%m%d_%H%M%S)}"
VPB_COST_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compute_cost}"
VPB_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR}/${VPB_RUN_NAME}"
VPB_OUTPUT="${VPB_OUTPUT:-${VPB_OUTPUT_DIR}/visualprm_paper_predictions.jsonl}"

export VPB_LIMIT VPB_THRESHOLD VPB_DTYPE VPB_OUTPUT
export VISUALPRM_MODEL_PATH="${VISUALPRM_MODEL_PATH:-VisualPRM/VisualPRM-8B}"

mkdir -p "${VPB_OUTPUT_DIR}"

start_time="$(date +%s.%N)"
bash "${SCRIPT_DIR}/../run_eval_visualprm.sh"
end_time="$(date +%s.%N)"
wall_time="$(awk -v start="${start_time}" -v end="${end_time}" 'BEGIN { printf "%.6f", end - start }')"

python "${SCRIPT_DIR}/summarize_compute_cost.py" \
  --predictions "${VPB_OUTPUT}" \
  --backend visualprm \
  --mode paper_soft_score \
  --output-dir "${VPB_OUTPUT_DIR}" \
  --prefix visualprm_paper \
  --wall-time-sec "${wall_time}"

echo "VisualPRM paper compute-cost outputs: ${VPB_OUTPUT_DIR}"
