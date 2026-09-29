#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

VPB_BACKENDS="${VPB_BACKENDS:-sft,visualprm}"
VPB_LIMIT="${VPB_LIMIT:-128}"
VPB_SAMPLE_STRATEGY="${VPB_SAMPLE_STRATEGY:-random}"
VPB_SEED="${VPB_SEED:-42}"
VPB_RUN_NAME="${VPB_RUN_NAME:-cost_$(date +%Y%m%d_%H%M%S)}"
VPB_COST_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compute_cost}"
VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
VPB_LOG_EVERY="${VPB_LOG_EVERY:-10}"

SFT_BASE_URL="${SFT_BASE_URL:-${PRM_BASE_URL:-${VPB_BASE_URL:-http://127.0.0.1:8000/v1}}}"
SFT_API_KEY="${SFT_API_KEY:-${VPB_API_KEY:-EMPTY}}"
SFT_MODEL="${SFT_MODEL:-${PRM_MODEL:-${VPB_MODEL:-auto}}}"
SFT_MODE="${SFT_MODE:-global_think_stepwise}"
SFT_CONCURRENCY="${SFT_CONCURRENCY:-${VPB_CONCURRENCY:-1}}"
SFT_TEMPERATURE="${SFT_TEMPERATURE:-0.0}"
SFT_USE_REFERENCE_ANSWER="${SFT_USE_REFERENCE_ANSWER:-${VPB_USE_REFERENCE_ANSWER:-0}}"
SFT_THINK_MAX_TOKENS="${SFT_THINK_MAX_TOKENS:-${PRM_THINK_MAX_TOKENS:-1024}}"
SFT_WARMUP_MAX_TOKENS="${SFT_WARMUP_MAX_TOKENS:-1024}"
SFT_FULL_MAX_TOKENS="${SFT_FULL_MAX_TOKENS:-2048}"
SFT_MAX_RETRIES="${SFT_MAX_RETRIES:-3}"
SFT_REQUEST_TIMEOUT="${SFT_REQUEST_TIMEOUT:-300}"
SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-}"

VISUALPRM_MODEL_PATH="${VISUALPRM_MODEL_PATH:-VisualPRM/VisualPRM-8B}"
VISUALPRM_DTYPE="${VISUALPRM_DTYPE:-${VPB_DTYPE:-bfloat16}}"
VISUALPRM_THRESHOLD="${VISUALPRM_THRESHOLD:-${VPB_THRESHOLD:-0.85}}"

ARGS=(
  --backends "${VPB_BACKENDS}"
  --benchmark-dir "${VPB_BENCHMARK_DIR:-${SCRIPT_DIR}/VisualProcessBench}"
  --limit "${VPB_LIMIT}"
  --sample-strategy "${VPB_SAMPLE_STRATEGY}"
  --seed "${VPB_SEED}"
  --output-dir "${VPB_COST_OUTPUT_DIR}"
  --run-name "${VPB_RUN_NAME}"
  --log-every "${VPB_LOG_EVERY}"
  --sft-base-url "${SFT_BASE_URL}"
  --sft-api-key "${SFT_API_KEY}"
  --sft-model "${SFT_MODEL}"
  --sft-mode "${SFT_MODE}"
  --sft-concurrency "${SFT_CONCURRENCY}"
  --sft-temperature "${SFT_TEMPERATURE}"
  --sft-use-reference-answer "${SFT_USE_REFERENCE_ANSWER}"
  --sft-think-max-tokens "${SFT_THINK_MAX_TOKENS}"
  --sft-warmup-max-tokens "${SFT_WARMUP_MAX_TOKENS}"
  --sft-full-max-tokens "${SFT_FULL_MAX_TOKENS}"
  --sft-max-retries "${SFT_MAX_RETRIES}"
  --sft-request-timeout "${SFT_REQUEST_TIMEOUT}"
  --sft-tokenizer-path "${SFT_TOKENIZER_PATH}"
  --visualprm-model-path "${VISUALPRM_MODEL_PATH}"
  --visualprm-dtype "${VISUALPRM_DTYPE}"
  --visualprm-threshold "${VISUALPRM_THRESHOLD}"
)

if [[ "${VPB_NO_PROGRESS}" == "1" ]]; then
  ARGS+=(--no-progress)
fi

if [[ "${VPB_OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "${SCRIPT_DIR}/benchmark_compute_cost.py" "${ARGS[@]}"
