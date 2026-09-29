#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${raw}"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

build_base_urls() {
  local client_host="$1"
  local port_base="$2"
  local ports_raw="$3"
  local replicas="$4"
  local ports=()
  local urls=()
  if [[ -n "${ports_raw}" ]]; then
    split_csv "${ports_raw}" ports
  else
    for ((i = 0; i < replicas; i += 1)); do
      ports+=("$((port_base + i))")
    done
  fi
  for port in "${ports[@]}"; do
    urls+=("http://${client_host}:${port}/v1")
  done
  join_by_comma "${urls[@]}"
}

infer_replicas() {
  local devices_raw="$1"
  local explicit_replicas="$2"
  if [[ -n "${explicit_replicas}" ]]; then
    printf '%s\n' "${explicit_replicas}"
    return
  fi
  local devices=()
  split_csv "${devices_raw}" devices
  printf '%s\n' "${#devices[@]}"
}

VPB_LIMIT="${VPB_LIMIT:-128}"
VPB_SAMPLE_STRATEGY="${VPB_SAMPLE_STRATEGY:-first}"
VPB_SEED="${VPB_SEED:-42}"
VPB_RUN_NAME="${VPB_RUN_NAME:-sft_vllm_cost_$(date +%Y%m%d_%H%M%S)}"
VPB_COST_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compute_cost}"
VPB_NO_PROGRESS="${VPB_NO_PROGRESS:-0}"
VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
VPB_LOG_EVERY="${VPB_LOG_EVERY:-10}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${MODEL_PATH:-}}"
SFT_START_VLLM="${SFT_START_VLLM:-}"
if [[ -z "${SFT_START_VLLM}" ]]; then
  if [[ -n "${SFT_MODEL_PATH}" ]]; then
    SFT_START_VLLM=1
  else
    SFT_START_VLLM=0
  fi
fi
SFT_STOP_VLLM_AFTER="${SFT_STOP_VLLM_AFTER:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8000}"
PORTS="${PORTS:-}"
NUM_REPLICAS="${NUM_REPLICAS:-}"
REPLICAS="$(infer_replicas "${CUDA_VISIBLE_DEVICES}" "${NUM_REPLICAS}")"
export CUDA_VISIBLE_DEVICES CLIENT_HOST PORT_BASE PORTS NUM_REPLICAS

SFT_BASE_URL="${SFT_BASE_URL:-${PRM_BASE_URL:-$(build_base_urls "${CLIENT_HOST}" "${PORT_BASE}" "${PORTS}" "${REPLICAS}")}}"
SFT_API_KEY="${SFT_API_KEY:-${VPB_API_KEY:-EMPTY}}"
SFT_MODEL="${SFT_MODEL:-${PRM_MODEL:-${VPB_MODEL:-auto}}}"
SFT_MODE="${SFT_MODE:-global_think_stepwise}"
SFT_CONCURRENCY="${SFT_CONCURRENCY:-${VPB_CONCURRENCY:-${PRM_CONCURRENCY:-${REPLICAS}}}}"
SFT_TEMPERATURE="${SFT_TEMPERATURE:-0.0}"
SFT_USE_REFERENCE_ANSWER="${SFT_USE_REFERENCE_ANSWER:-${VPB_USE_REFERENCE_ANSWER:-0}}"
SFT_THINK_MAX_TOKENS="${SFT_THINK_MAX_TOKENS:-${PRM_THINK_MAX_TOKENS:-1024}}"
SFT_WARMUP_MAX_TOKENS="${SFT_WARMUP_MAX_TOKENS:-1024}"
SFT_FULL_MAX_TOKENS="${SFT_FULL_MAX_TOKENS:-2048}"
SFT_DIRECT_MAX_TOKENS="${SFT_DIRECT_MAX_TOKENS:-512}"
SFT_MAX_RETRIES="${SFT_MAX_RETRIES:-3}"
SFT_REQUEST_TIMEOUT="${SFT_REQUEST_TIMEOUT:-300}"
SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-${SFT_MODEL_PATH}}"

stop_vllm_if_requested() {
  if [[ "${SFT_START_VLLM}" == "1" && "${SFT_STOP_VLLM_AFTER}" == "1" ]]; then
    bash "${SCRIPT_DIR}/serve_sft_vllm_dp.sh" stop || true
  fi
}

if [[ "${SFT_START_VLLM}" == "1" ]]; then
  if [[ -z "${SFT_MODEL_PATH}" ]]; then
    echo "ERROR: SFT_MODEL_PATH or MODEL_PATH is required when SFT_START_VLLM=1." >&2
    exit 1
  fi
  if [[ "${SFT_MODEL_PATH}" == /path/to/* ]]; then
    echo "ERROR: SFT_MODEL_PATH is still a placeholder: ${SFT_MODEL_PATH}" >&2
    echo "Please replace it with the real merged SFT model directory." >&2
    exit 1
  fi
  trap stop_vllm_if_requested EXIT
  bash "${SCRIPT_DIR}/serve_sft_vllm_dp.sh" start "${SFT_MODEL_PATH}"
else
  echo "Reuse existing SFT vLLM endpoint(s): ${SFT_BASE_URL}"
fi

ARGS=(
  --backends sft
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
  --sft-direct-max-tokens "${SFT_DIRECT_MAX_TOKENS}"
  --sft-max-retries "${SFT_MAX_RETRIES}"
  --sft-request-timeout "${SFT_REQUEST_TIMEOUT}"
  --sft-tokenizer-path "${SFT_TOKENIZER_PATH}"
)

if [[ "${VPB_NO_PROGRESS}" == "1" ]]; then
  ARGS+=(--no-progress)
fi

if [[ "${VPB_OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python "${SCRIPT_DIR}/benchmark_compute_cost.py" "${ARGS[@]}"
