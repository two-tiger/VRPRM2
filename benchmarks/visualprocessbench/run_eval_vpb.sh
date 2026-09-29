#!/usr/bin/env bash
# Single VisualProcessBench evaluation entry point for API-served VRPRM models.
# Consolidates run_eval_sft_api.sh and run_eval_no_think_stepwise.sh (preserved
# on archive/snapshot-202609).
#
# Modes (VPB_MODE):
#   global_think  two-stage: one global <think> block, then one guided 0/1
#                 single-token request per step        [official VRPRM protocol]
#   no_think      no thinking. VPB_NO_THINK_MODE=single_pass (one JSON response
#                 per sample, default) or stepwise (one request per step)
#   base_warmup   process-untrained thinking base model: short warmup think,
#                 then stepwise single tokens (the "Qwen3-VL-8B-Thinking" row)
#
# Protocol (docs/PROTOCOL.md): reference answers are NOT sent to the model
# (VPB_USE_REFERENCE_ANSWER=0, the official protocol); they are only used to
# compute metrics afterwards. Set VPB_USE_REFERENCE_ANSWER=1 only for
# oracle-style diagnostics. Metrics come from metrics.py, which reports both
# overall_step_macro_f1 (step-pooled, official) and mean_source_macro_f1
# (unweighted subset mean).
#
# Common environment variables (all optional):
#   VPB_BASE_URL, VPB_API_KEY, VPB_MODEL   endpoint config; MODEL=auto reads /v1/models
#   VPB_OUTPUT                              prediction JSONL path
#   VPB_LIMIT, VPB_CONCURRENCY, VPB_RESUME/VPB_OVERWRITE/VPB_RETRY_ERRORS/VPB_RETRY_INVALID
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

VPB_MODE="${VPB_MODE:-global_think}"

export VPB_BASE_URL="${VPB_BASE_URL:-http://127.0.0.1:8000/v1}"
export VPB_API_KEY="${VPB_API_KEY:-EMPTY}"
export VPB_MODEL="${VPB_MODEL:-auto}"
export VPB_CONCURRENCY="${VPB_CONCURRENCY:-4}"
export VPB_LIMIT="${VPB_LIMIT:-0}"
export VPB_MAX_RETRIES="${VPB_MAX_RETRIES:-3}"
export VPB_REQUEST_TIMEOUT="${VPB_REQUEST_TIMEOUT:-300}"
export VPB_STATUS_EVERY="${VPB_STATUS_EVERY:-10}"
export VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
export VPB_RESUME="${VPB_RESUME:-1}"
export VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
export VPB_RETRY_ERRORS="${VPB_RETRY_ERRORS:-0}"
export VPB_RETRY_INVALID="${VPB_RETRY_INVALID:-0}"
VPB_USE_REFERENCE_ANSWER="${VPB_USE_REFERENCE_ANSWER:-0}"

EVAL_PY=""
case "${VPB_MODE}" in
  global_think)
    EVAL_PY="eval_vpb_global_think.py"
    export VPB_THINK_MAX_TOKENS="${VPB_THINK_MAX_TOKENS:-1024}"
    export VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/global_think_stepwise_no_ref_predictions.jsonl}"
    ARGS=(
      --think-max-tokens "${VPB_THINK_MAX_TOKENS}"
      --use-reference-answer "${VPB_USE_REFERENCE_ANSWER}"
      --temperature "${VPB_TEMPERATURE:-0.0}"
    )
    ;;
  no_think)
    EVAL_PY="eval_vpb_no_think.py"
    export VPB_NO_THINK_MODE="${VPB_NO_THINK_MODE:-single_pass}"
    export VPB_MAX_TOKENS="${VPB_MAX_TOKENS:-512}"
    export VPB_INCLUDE_PREVIOUS_JUDGMENTS="${VPB_INCLUDE_PREVIOUS_JUDGMENTS:-1}"
    export VPB_STEP_IMAGE_MODE="${VPB_STEP_IMAGE_MODE:-never}"
    export VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/no_think_${VPB_NO_THINK_MODE}_no_ref_predictions.jsonl}"
    ARGS=(
      --no-think-mode "${VPB_NO_THINK_MODE}"
      --max-tokens "${VPB_MAX_TOKENS}"
      --use-reference-answer "${VPB_USE_REFERENCE_ANSWER}"
      --include-previous-judgments "${VPB_INCLUDE_PREVIOUS_JUDGMENTS}"
      --step-image-mode "${VPB_STEP_IMAGE_MODE}"
    )
    ;;
  base_warmup)
    EVAL_PY="vpb_common.py"
    export VPB_REWARD_MODE="${VPB_REWARD_MODE:-stepwise}"
    export VPB_MAX_TOKENS="${VPB_MAX_TOKENS:-2048}"
    export VPB_WARMUP_MAX_TOKENS="${VPB_WARMUP_MAX_TOKENS:-1024}"
    export VPB_OUTPUT="${VPB_OUTPUT:-${SCRIPT_DIR}/outputs/base_warmup_${VPB_REWARD_MODE}_predictions.jsonl}"
    ARGS=(
      --reward-mode "${VPB_REWARD_MODE}"
      --max-tokens "${VPB_MAX_TOKENS}"
      --warmup-max-tokens "${VPB_WARMUP_MAX_TOKENS}"
    )
    ;;
  *)
    echo "Unknown VPB_MODE '${VPB_MODE}'. Use global_think | no_think | base_warmup." >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --base-url "${VPB_BASE_URL}"
  --api-key "${VPB_API_KEY}"
  --model "${VPB_MODEL}"
  --concurrency "${VPB_CONCURRENCY}"
  --limit "${VPB_LIMIT}"
  --max-retries "${VPB_MAX_RETRIES}"
  --request-timeout "${VPB_REQUEST_TIMEOUT}"
  --status-every "${VPB_STATUS_EVERY}"
  --output "${VPB_OUTPUT}"
)

[[ -n "${VPB_BENCHMARK_DIR:-}" ]] && COMMON_ARGS+=(--benchmark-dir "${VPB_BENCHMARK_DIR}")
[[ -n "${VPB_ERROR_LOG:-}" ]] && COMMON_ARGS+=(--error-log "${VPB_ERROR_LOG}")
[[ -n "${VPB_STATUS_FILE:-}" ]] && COMMON_ARGS+=(--status-file "${VPB_STATUS_FILE}")
[[ "${VPB_NO_PROGRESS}" == "1" ]] && COMMON_ARGS+=(--no-progress)
[[ "${VPB_RESUME}" == "1" ]] && COMMON_ARGS+=(--resume)
[[ "${VPB_OVERWRITE}" == "1" ]] && COMMON_ARGS+=(--overwrite)
[[ "${VPB_RETRY_ERRORS}" == "1" ]] && COMMON_ARGS+=(--retry-errors)
[[ "${VPB_RETRY_INVALID}" == "1" ]] && COMMON_ARGS+=(--retry-invalid)

echo "[run_eval_vpb] mode=${VPB_MODE} use_reference_answer=${VPB_USE_REFERENCE_ANSWER}"
echo "[run_eval_vpb] evaluator=${EVAL_PY} output=${VPB_OUTPUT}"

python "${SCRIPT_DIR}/${EVAL_PY}" "${COMMON_ARGS[@]}" "${ARGS[@]}"

METRICS_OUTPUT="${VPB_OUTPUT%.jsonl}.metrics.json"
python "${SCRIPT_DIR}/metrics.py" --predictions "${VPB_OUTPUT}" --output "${METRICS_OUTPUT}"
echo "[run_eval_vpb] metrics written to ${METRICS_OUTPUT}"
echo "[run_eval_vpb] paper Overall = overall_step_macro_f1 (official pooled, single column); mean_source_macro_f1 = diagnostic only"
