#!/usr/bin/env bash
# J1-style entropy data selection: score the 40K RL pool with the frozen SFT
# checkpoint and reorder it uncertainty-first for train_vrprm_rl_v2.sh.
#
# 1) Serve the merged SFT checkpoint (LoRA already merged via sft/merge_lora.sh):
#      bash benchmarks/visualprocessbench/serve_vllm.sh /path/to/checkpoint-513-merge
#    For multi-replica serving use serve_vllm_dp.sh and pass comma-separated
#    URLs in SELECT_BASE_URL below.
# 2) Run this script (a few hours on a single 8B vLLM instance for 40K x 4).
# 3) Start RL v2; it picks up the entropy-ordered split automatically:
#      bash easyr1/examples/vrprm/train_vrprm_rl_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VRPRM_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"

SELECT_BASE_URL="${SELECT_BASE_URL:-http://127.0.0.1:8000/v1}"
SELECT_API_KEY="${SELECT_API_KEY:-EMPTY}"
SELECT_MODEL="${SELECT_MODEL:-auto}"
SELECT_NUM_SAMPLES="${SELECT_NUM_SAMPLES:-4}"
SELECT_TEMPERATURE="${SELECT_TEMPERATURE:-0.7}"
SELECT_CONCURRENCY="${SELECT_CONCURRENCY:-32}"

SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k}"
ENTROPY_DATASET_DIR="${ENTROPY_DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k_entropy}"

if [[ ! -f "${SOURCE_DATASET_DIR}/train.jsonl" ]]; then
  echo "Source RL split not found: ${SOURCE_DATASET_DIR}/train.jsonl" >&2
  echo "It is auto-prepared by train_vrprm_rl.sh / train_vrprm_rl_v2.sh on first RL run," >&2
  echo "or build it manually with the prepare step in those scripts." >&2
  exit 1
fi

python3 "${EASYR1_ROOT}/scripts/select_rl_data_by_entropy.py" \
  --input "${SOURCE_DATASET_DIR}/train.jsonl" \
  --output-dir "${ENTROPY_DATASET_DIR}" \
  --image-dir "${VRPRM_ROOT}/data" \
  --base-url "${SELECT_BASE_URL}" \
  --api-key "${SELECT_API_KEY}" \
  --model "${SELECT_MODEL}" \
  --num-samples "${SELECT_NUM_SAMPLES}" \
  --temperature "${SELECT_TEMPERATURE}" \
  --concurrency "${SELECT_CONCURRENCY}"

echo "Entropy-ordered split ready: ${ENTROPY_DATASET_DIR}"
