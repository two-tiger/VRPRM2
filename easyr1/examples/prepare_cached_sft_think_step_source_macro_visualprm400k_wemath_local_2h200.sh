#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VRPRM_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${VRPRM_ROOT}/.." && pwd)"
SERVE_SCRIPT="${VRPRM_ROOT}/benchmarks/visualprocessbench/serve_sft_vllm_dp.sh"

DEFAULT_SFT_MODEL="${VRPRM_ROOT}/sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft/v0-20260628-015250/checkpoint-513-merge"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_SFT_MODEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vrprm-v2-sft}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_REPLICAS="${NUM_REPLICAS:-2}"
export PORT_BASE="${PORT_BASE:-8000}"
export PORTS="${PORTS:-8000,8001}"
export CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
export HOST="${HOST:-0.0.0.0}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 8}}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export LOG_DIR="${LOG_DIR:-${VRPRM_ROOT}/benchmarks/visualprocessbench/outputs/sft_vllm_dp2_local_logs}"
export PID_FILE="${PID_FILE:-${LOG_DIR}/pids.txt}"

export DATA_ROOT="${DATA_ROOT:-${VRPRM_ROOT}/data/VisualPRM400K-v1.1-Raw}"
export IMAGE_DIR="${IMAGE_DIR:-${VRPRM_ROOT}/data}"
export SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_filtered_source_cases_clean_pos0875_pool150k_balanced}"
export DATASET_DIR="${DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_cached_sft_think_step_source_macro_wemath_60k}"

export SFT_ROLLOUT_BASE_URLS="${SFT_ROLLOUT_BASE_URLS:-http://${CLIENT_HOST}:8000/v1,http://${CLIENT_HOST}:8001/v1}"
export SFT_ROLLOUT_MODEL="${SFT_ROLLOUT_MODEL:-${SERVED_MODEL_NAME}}"
export SFT_ROLLOUT_CONCURRENCY="${SFT_ROLLOUT_CONCURRENCY:-128}"
export SFT_ROLLOUT_PER_ENDPOINT_CONCURRENCY="${SFT_ROLLOUT_PER_ENDPOINT_CONCURRENCY:-64}"
export SFT_ROLLOUT_REQUEST_TIMEOUT="${SFT_ROLLOUT_REQUEST_TIMEOUT:-300}"
export SFT_ROLLOUT_MAX_RETRIES="${SFT_ROLLOUT_MAX_RETRIES:-3}"
export CACHE_FLUSH_EVERY="${CACHE_FLUSH_EVERY:-20}"
export CACHE_FSYNC_EVERY="${CACHE_FSYNC_EVERY:-0}"

export MAX_TRAIN_SOURCE_ROLLOUT="${MAX_TRAIN_SOURCE_ROLLOUT:-16000}"
export MAX_VAL_SOURCE_ROLLOUT="${MAX_VAL_SOURCE_ROLLOUT:-2000}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-60000}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-3000}"
export THINKING_SAMPLES="${THINKING_SAMPLES:-3}"
export MAX_THINKING_VARIANTS_PER_SOURCE="${MAX_THINKING_VARIANTS_PER_SOURCE:-2}"
export THINKING_DROPOUT_RATIO="${THINKING_DROPOUT_RATIO:-0.25}"
export NO_THINKING_RATIO="${NO_THINKING_RATIO:-0.10}"

if [[ "${START_SFT_SERVER:-true}" == "true" || "${START_SFT_SERVER:-true}" == "1" ]]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  MODEL_PATH="${MODEL_PATH}" \
  bash "${SERVE_SCRIPT}" start "${MODEL_PATH}"
else
  echo "START_SFT_SERVER=false, assuming SFT endpoints are already ready: ${SFT_ROLLOUT_BASE_URLS}"
fi

echo "Preparing cached SFT-thinking stepwise RL data locally."
echo "SFT endpoints: ${SFT_ROLLOUT_BASE_URLS}"
echo "Dataset dir: ${DATASET_DIR}"

bash "${SCRIPT_DIR}/prepare_cached_sft_think_step_source_macro_visualprm400k_wemath.sh"

if [[ "${STOP_SFT_SERVER_AFTER_PREP:-false}" == "true" || "${STOP_SFT_SERVER_AFTER_PREP:-false}" == "1" ]]; then
  bash "${SERVE_SCRIPT}" stop
fi
