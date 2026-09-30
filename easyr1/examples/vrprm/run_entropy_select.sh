#!/usr/bin/env bash
# J1-style entropy data selection: score the 40K RL pool with the frozen SFT
# checkpoint and reorder it uncertainty-first for train_vrprm_rl_v2.sh.
#
# 1) Serve the merged SFT checkpoint (LoRA already merged via sft/merge_lora.sh).
#    8xH200 preset (one replica per GPU, ~1-1.5h for the whole pool):
#      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_REPLICAS=8 PORT_BASE=8000 \
#        MAX_NUM_SEQS=128 GPU_MEMORY_UTILIZATION=0.92 ENABLE_PREFIX_CACHING=1 \
#        ENFORCE_EAGER=0 \
#        bash benchmarks/visualprocessbench/serve_vllm_dp.sh start /path/to/checkpoint-513-merge
#      SELECT_BASE_URL="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1" \
#        SELECT_CONCURRENCY=128 bash easyr1/examples/vrprm/run_entropy_select.sh
#    Single-GPU fallback: bash benchmarks/visualprocessbench/serve_vllm.sh <model>
# 2) Run this script (a few hours on a single 8B vLLM instance for 40K x 4).
# 3) Start RL v2; it picks up the entropy-ordered split automatically:
#      bash easyr1/examples/vrprm/train_vrprm_rl_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VRPRM_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
BACKUP_DATA_ROOT="/mnt/shared-storage-user/evobox-share/chenxinquan/VRPRM_OpenSource/backup/VRPRM_v2.0_artifacts/data"
# RL rows reference images as VisualPRM400K-v1.1-Raw/images/..., so the image
# dir must be the directory CONTAINING VisualPRM400K-v1.1-Raw after the
# reorganization (missing images are silently skipped by the scorer, which
# would make entropy scores text-only).
SELECT_IMAGE_DIR="${SELECT_IMAGE_DIR:-/mnt/shared-storage-user/evobox-share/chenxinquan/VRPRM_OpenSource/backup/release_v1/huggingface_datasets}"

SELECT_BASE_URL="${SELECT_BASE_URL:-http://127.0.0.1:8000/v1}"
SELECT_API_KEY="${SELECT_API_KEY:-EMPTY}"
SELECT_MODEL="${SELECT_MODEL:-auto}"
SELECT_NUM_SAMPLES="${SELECT_NUM_SAMPLES:-4}"
SELECT_TEMPERATURE="${SELECT_TEMPERATURE:-0.7}"
SELECT_CONCURRENCY="${SELECT_CONCURRENCY:-32}"  # ~128 per 8 replicas (in-flight = concurrency x num_samples / replicas)

SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${BACKUP_DATA_ROOT}/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k}"
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
  --image-dir "${SELECT_IMAGE_DIR}" \
  --base-url "${SELECT_BASE_URL}" \
  --api-key "${SELECT_API_KEY}" \
  --model "${SELECT_MODEL}" \
  --num-samples "${SELECT_NUM_SAMPLES}" \
  --temperature "${SELECT_TEMPERATURE}" \
  --concurrency "${SELECT_CONCURRENCY}"

echo "Entropy-ordered split ready: ${ENTROPY_DATASET_DIR}"
