#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${MODEL_PATH:-}}"
SFT_MODES="${SFT_MODES:-global_think_stepwise,direct_scores}"
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
export VPB_LIMIT="${VPB_LIMIT:-0}"
export VPB_SAMPLE_STRATEGY="${VPB_SAMPLE_STRATEGY:-first}"
export VPB_COST_OUTPUT_DIR="${VPB_COST_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/compute_cost}"
export VPB_OVERWRITE="${VPB_OVERWRITE:-0}"
export SFT_BASE_URL="${SFT_BASE_URL:-${PRM_BASE_URL:-$(build_base_urls "${CLIENT_HOST}" "${PORT_BASE}" "${PORTS}" "${REPLICAS}")}}"
export SFT_MODEL="${SFT_MODEL:-${PRM_MODEL:-${VPB_MODEL:-auto}}}"
export SFT_CONCURRENCY="${SFT_CONCURRENCY:-${VPB_CONCURRENCY:-${PRM_CONCURRENCY:-${REPLICAS}}}}"
export SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-${SFT_MODEL_PATH}}"
export SFT_USE_REFERENCE_ANSWER="${SFT_USE_REFERENCE_ANSWER:-${VPB_USE_REFERENCE_ANSWER:-0}}"

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

IFS=',' read -r -a MODE_ARRAY <<< "${SFT_MODES}"
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="${raw_mode#"${raw_mode%%[![:space:]]*}"}"
  mode="${mode%"${mode##*[![:space:]]}"}"
  [[ -z "${mode}" ]] && continue
  export SFT_MODE="${mode}"
  export SFT_START_VLLM=0
  export VPB_RUN_NAME="${VPB_RUN_PREFIX:-sft_vllm_full}_${mode}"
  echo "==> Run full VisualProcessBench SFT cost: mode=${mode}, output=${VPB_COST_OUTPUT_DIR}/${VPB_RUN_NAME}"
  bash "${SCRIPT_DIR}/run_benchmark_sft_vllm_compute_cost.sh"
done
