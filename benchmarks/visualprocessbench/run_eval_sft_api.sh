#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VPB_BASE_URL="${VPB_BASE_URL:-http://127.0.0.1:8000/v1}"
export VPB_API_KEY="${VPB_API_KEY:-EMPTY}"
export VPB_MODEL="${VPB_MODEL:-auto}"
export VPB_CONCURRENCY="${VPB_CONCURRENCY:-2}"
export VPB_LIMIT="${VPB_LIMIT:-0}"
export VPB_REWARD_MODE="${VPB_REWARD_MODE:-stepwise}" # stepwise or full
export VPB_MAX_TOKENS="${VPB_MAX_TOKENS:-2048}"
export VPB_WARMUP_MAX_TOKENS="${VPB_WARMUP_MAX_TOKENS:-1024}"
export VPB_MAX_RETRIES="${VPB_MAX_RETRIES:-3}"
export VPB_REQUEST_TIMEOUT="${VPB_REQUEST_TIMEOUT:-300}"
export VPB_STATUS_EVERY="${VPB_STATUS_EVERY:-10}"
export VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
export VPB_RESUME="${VPB_RESUME:-1}"
export VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
export VPB_RETRY_ERRORS="${VPB_RETRY_ERRORS:-0}"
export VPB_RETRY_INVALID="${VPB_RETRY_INVALID:-0}"
export VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/vrprm-v2-rl-clean-hard-150-${VPB_REWARD_MODE}_predictions.jsonl}"

ARGS=(
  --base-url "${VPB_BASE_URL}"
  --api-key "${VPB_API_KEY}"
  --model "${VPB_MODEL}"
  --concurrency "${VPB_CONCURRENCY}"
  --limit "${VPB_LIMIT}"
  --reward-mode "${VPB_REWARD_MODE}"
  --max-tokens "${VPB_MAX_TOKENS}"
  --warmup-max-tokens "${VPB_WARMUP_MAX_TOKENS}"
  --max-retries "${VPB_MAX_RETRIES}"
  --request-timeout "${VPB_REQUEST_TIMEOUT}"
  --status-every "${VPB_STATUS_EVERY}"
  --output "${VPB_OUTPUT}"
)

if [[ -n "${VPB_ERROR_LOG:-}" ]]; then
  ARGS+=(--error-log "${VPB_ERROR_LOG}")
fi

if [[ -n "${VPB_STATUS_FILE:-}" ]]; then
  ARGS+=(--status-file "${VPB_STATUS_FILE}")
fi

if [[ "${VPB_NO_PROGRESS}" == "1" ]]; then
  ARGS+=(--no-progress)
fi

if [[ "${VPB_RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ "${VPB_OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "${VPB_RETRY_ERRORS}" == "1" ]]; then
  ARGS+=(--retry-errors)
fi

if [[ "${VPB_RETRY_INVALID}" == "1" ]]; then
  ARGS+=(--retry-invalid)
fi

python "${SCRIPT_DIR}/api_eval_sft.py" "${ARGS[@]}"
