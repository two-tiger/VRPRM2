#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

VPB_BASE_URL="${VPB_BASE_URL:-http://127.0.0.1:8000/v1}"
VPB_API_KEY="${VPB_API_KEY:-EMPTY}"
VPB_MODEL="${VPB_MODEL:-auto}"
VPB_CONCURRENCY="${VPB_CONCURRENCY:-4}"
VPB_LIMIT="${VPB_LIMIT:-0}"
VPB_MAX_TOKENS="${VPB_MAX_TOKENS:-512}"
VPB_MAX_RETRIES="${VPB_MAX_RETRIES:-3}"
VPB_REQUEST_TIMEOUT="${VPB_REQUEST_TIMEOUT:-300}"
VPB_STATUS_EVERY="${VPB_STATUS_EVERY:-10}"
VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
VPB_RESUME="${VPB_RESUME:-1}"
VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
VPB_RETRY_ERRORS="${VPB_RETRY_ERRORS:-0}"
VPB_RETRY_INVALID="${VPB_RETRY_INVALID:-0}"
VPB_NO_THINK_MODE="${VPB_NO_THINK_MODE:-single_pass}"
VPB_USE_REFERENCE_ANSWER="${VPB_USE_REFERENCE_ANSWER:-0}"
VPB_INCLUDE_PREVIOUS_JUDGMENTS="${VPB_INCLUDE_PREVIOUS_JUDGMENTS:-1}"
VPB_STEP_IMAGE_MODE="${VPB_STEP_IMAGE_MODE:-never}"
VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/no_think_single_pass_no_ref_predictions.jsonl}"

ARGS=(
  --base-url "${VPB_BASE_URL}"
  --api-key "${VPB_API_KEY}"
  --model "${VPB_MODEL}"
  --concurrency "${VPB_CONCURRENCY}"
  --limit "${VPB_LIMIT}"
  --max-tokens "${VPB_MAX_TOKENS}"
  --max-retries "${VPB_MAX_RETRIES}"
  --request-timeout "${VPB_REQUEST_TIMEOUT}"
  --status-every "${VPB_STATUS_EVERY}"
  --no-think-mode "${VPB_NO_THINK_MODE}"
  --use-reference-answer "${VPB_USE_REFERENCE_ANSWER}"
  --include-previous-judgments "${VPB_INCLUDE_PREVIOUS_JUDGMENTS}"
  --step-image-mode "${VPB_STEP_IMAGE_MODE}"
  --output "${VPB_OUTPUT}"
)

if [[ -n "${VPB_BENCHMARK_DIR:-}" ]]; then
  ARGS+=(--benchmark-dir "${VPB_BENCHMARK_DIR}")
fi

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

python "${SCRIPT_DIR}/api_eval_no_think_stepwise.py" "${ARGS[@]}"
