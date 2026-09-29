#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${ACTION:-start}"
if [[ "${1:-}" == "start" || "${1:-}" == "stop" || "${1:-}" == "status" ]]; then
  ACTION="$1"
  shift
fi

MODEL_PATH="${1:-${MODEL_PATH:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
HOST="${HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8000}"
PORTS="${PORTS:-}"
CUDA_VISIBLE_DEVICES_WAS_SET=0
if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
  CUDA_VISIBLE_DEVICES_WAS_SET=1
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_REPLICAS="${NUM_REPLICAS:-}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 8}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
WAIT_READY="${WAIT_READY:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

PRM_CONCURRENCY="${PRM_CONCURRENCY:-}"
PRM_MODEL="${PRM_MODEL:-}"
LOG_DIR="${LOG_DIR:-}"
PID_FILE="${PID_FILE:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

usage() {
  cat >&2 <<'EOF'
Usage:
  serve_sft_vllm_dp.sh [start] <model_path_or_hf_id>
  serve_sft_vllm_dp.sh stop
  serve_sft_vllm_dp.sh status

Environment overrides:
  MODEL_PATH                 Model directory or Hugging Face model id.
  SERVED_MODEL_NAME          Name exposed by each OpenAI-compatible API.
  CUDA_VISIBLE_DEVICES       GPUs used as independent replicas. Default: 0,1,2,3
  NUM_REPLICAS               Number of replicas. Default: inferred from CUDA_VISIBLE_DEVICES
  PORT_BASE                  First port when PORTS is unset. Default: 8000
  PORTS                      Comma-separated ports, e.g. 8000,8001
  MAX_MODEL_LEN              Max context length. Default: 16384
  MAX_NUM_SEQS               Per-replica concurrent sequences. Default: 16
  GPU_MEMORY_UTILIZATION     GPU memory fraction. Default: 0.90
  LIMIT_MM_PER_PROMPT        Multimodal limits. Default: {"image": 8}
  ENFORCE_EAGER              Disable torch.compile/cudagraph. Default: 1
  PRM_CONCURRENCY            Recommended scorer concurrency. Default: 32 for >=4 replicas, else 16
  LOG_DIR                    vLLM log directory. Default: outputs/sft_vllm_dp<N>_logs
  VLLM_EXTRA_ARGS            Extra raw args appended to vllm serve.
EOF
}

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${raw}"
}

build_default_devices() {
  local count="$1"
  local devices=()
  local i
  for ((i = 0; i < count; i += 1)); do
    devices+=("${i}")
  done
  join_by_comma "${devices[@]}"
}

configure_defaults() {
  local devices=()

  if [[ -n "${NUM_REPLICAS}" && ! "${NUM_REPLICAS}" =~ ^[0-9]+$ ]]; then
    echo "NUM_REPLICAS must be a positive integer: ${NUM_REPLICAS}" >&2
    exit 2
  fi
  if [[ -n "${NUM_REPLICAS}" && "${NUM_REPLICAS}" -lt 1 ]]; then
    echo "NUM_REPLICAS must be >= 1: ${NUM_REPLICAS}" >&2
    exit 2
  fi

  if [[ "${CUDA_VISIBLE_DEVICES_WAS_SET}" == "0" && -n "${NUM_REPLICAS}" ]]; then
    CUDA_VISIBLE_DEVICES="$(build_default_devices "${NUM_REPLICAS}")"
  fi

  split_csv "${CUDA_VISIBLE_DEVICES}" devices
  if [[ -z "${NUM_REPLICAS}" ]]; then
    NUM_REPLICAS="${#devices[@]}"
  fi
  if [[ "${NUM_REPLICAS}" -lt 1 ]]; then
    echo "No GPUs configured. Set CUDA_VISIBLE_DEVICES or NUM_REPLICAS." >&2
    exit 2
  fi
  if [[ -z "${PRM_CONCURRENCY}" ]]; then
    if [[ "${NUM_REPLICAS}" -ge 4 ]]; then
      PRM_CONCURRENCY=32
    else
      PRM_CONCURRENCY=16
    fi
  fi
  if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${SCRIPT_DIR}/outputs/sft_vllm_dp${NUM_REPLICAS}_logs"
  fi
  if [[ -z "${PID_FILE}" ]]; then
    PID_FILE="${LOG_DIR}/pids.txt"
  fi
}

resolve_model() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
    return
  fi
  if [[ -d "${input_path}" ]]; then
    printf '%s\n' "$(cd "${input_path}" && pwd)"
    return
  fi
  local project_relative_path
  project_relative_path="$(cd "${VRPRM_ROOT}/.." && pwd)/${input_path}"
  if [[ -d "${project_relative_path}" ]]; then
    printf '%s\n' "${project_relative_path}"
    return
  fi
  printf '%s\n' "${input_path}"
}

build_ports() {
  local -n out_ref="$1"
  if [[ -n "${PORTS}" ]]; then
    split_csv "${PORTS}" out_ref
    return
  fi
  out_ref=()
  for ((i = 0; i < NUM_REPLICAS; i += 1)); do
    out_ref+=("$((PORT_BASE + i))")
  done
}

build_base_urls() {
  local -n ports_ref="$1"
  local -n urls_ref="$2"
  urls_ref=()
  for port in "${ports_ref[@]}"; do
    urls_ref+=("http://${CLIENT_HOST}:${port}/v1")
  done
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

status() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "No pid file: ${PID_FILE}"
    return 0
  fi
  local any_running=0
  while read -r pid gpu port log_path; do
    [[ -z "${pid}" || "${pid}" == \#* ]] && continue
    if pid_running "${pid}"; then
      echo "running pid=${pid} gpu=${gpu} port=${port} log=${log_path}"
      any_running=1
    else
      echo "stopped pid=${pid} gpu=${gpu} port=${port} log=${log_path}"
    fi
  done < "${PID_FILE}"
  [[ "${any_running}" == "1" ]]
}

stop_servers() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "No pid file: ${PID_FILE}"
    return 0
  fi
  while read -r pid gpu port log_path; do
    [[ -z "${pid}" || "${pid}" == \#* ]] && continue
    if pid_running "${pid}"; then
      echo "Stopping pid=${pid} gpu=${gpu} port=${port}"
      kill "${pid}" || true
    fi
  done < "${PID_FILE}"
  rm -f "${PID_FILE}"
}

wait_for_ready() {
  local -n ports_ref="$1"
  if [[ "${WAIT_READY}" != "1" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; skip readiness check."
    return 0
  fi

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local ready
  while ((SECONDS < deadline)); do
    ready=0
    for port in "${ports_ref[@]}"; do
      if curl -fsS "http://${CLIENT_HOST}:${port}/v1/models" >/dev/null 2>&1; then
        ready=$((ready + 1))
      fi
    done
    if [[ "${ready}" -eq "${#ports_ref[@]}" ]]; then
      echo "All ${ready} SFT vLLM replicas are ready."
      return 0
    fi
    echo "Waiting for SFT vLLM replicas: ${ready}/${#ports_ref[@]} ready..."
    sleep "${POLL_INTERVAL}"
  done

  echo "Timed out waiting for vLLM replicas. Check logs under ${LOG_DIR}." >&2
  return 1
}

start_servers() {
  if [[ -z "${MODEL_PATH}" ]]; then
    usage
    exit 2
  fi

  local devices=()
  local ports=()
  local urls=()
  split_csv "${CUDA_VISIBLE_DEVICES}" devices
  build_ports ports
  build_base_urls ports urls

  if [[ "${#devices[@]}" -ne "${NUM_REPLICAS}" ]]; then
    echo "CUDA_VISIBLE_DEVICES exposes ${#devices[@]} GPUs, but NUM_REPLICAS=${NUM_REPLICAS}." >&2
    exit 1
  fi
  if [[ "${#ports[@]}" -ne "${NUM_REPLICAS}" ]]; then
    echo "PORTS exposes ${#ports[@]} ports, but NUM_REPLICAS=${NUM_REPLICAS}." >&2
    exit 1
  fi

  MODEL_PATH="$(resolve_model "${MODEL_PATH}")"
  if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    exit 1
  fi
  if [[ -z "${SERVED_MODEL_NAME}" ]]; then
    SERVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
  fi
  if [[ -z "${PRM_MODEL}" ]]; then
    PRM_MODEL="${SERVED_MODEL_NAME}"
  fi

  if [[ -f "${PID_FILE}" ]] && status >/dev/null 2>&1; then
    echo "Existing SFT vLLM replicas appear to be running. Use:"
    echo "  ${BASH_SOURCE[0]} stop"
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  : > "${PID_FILE}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

  local extra_args=()
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
  fi

  echo "Serving SFT model with ${NUM_REPLICAS} single-GPU vLLM replicas"
  echo "Model: ${MODEL_PATH}"
  echo "Served model name: ${SERVED_MODEL_NAME}"
  echo "GPUs: $(join_by_comma "${devices[@]}")"
  echo "Ports: $(join_by_comma "${ports[@]}")"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
  echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
  echo "Logs: ${LOG_DIR}"
  echo

  for ((i = 0; i < NUM_REPLICAS; i += 1)); do
    local gpu="${devices[$i]}"
    local port="${ports[$i]}"
    local log_path="${LOG_DIR}/sft_vllm_gpu${gpu}_port${port}.log"
    local args=(
      serve "${MODEL_PATH}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --host "${HOST}"
      --port "${port}"
      --tensor-parallel-size 1
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
      --trust-remote-code
      --mm-processor-cache-gb 0
    )
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
      args+=(--enable-prefix-caching)
    fi
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
      args+=(--enforce-eager)
    fi
    args+=("${extra_args[@]}")

    echo "Starting replica $((i + 1))/${NUM_REPLICAS}: gpu=${gpu} port=${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup vllm "${args[@]}" > "${log_path}" 2>&1 &
    echo "$! ${gpu} ${port} ${log_path}" >> "${PID_FILE}"
  done

  wait_for_ready ports

  local prm_base_url
  prm_base_url="$(join_by_comma "${urls[@]}")"
  cat <<EOF

Use these settings for VPB-style BoN scoring:

  export PRM_BASE_URL=${prm_base_url}
  export PRM_MODEL=${PRM_MODEL}
  export PRM_CONCURRENCY=${PRM_CONCURRENCY}
  bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_eval_qwen3_sft_global_stepwise_bon.sh

Stop replicas with:

  PID_FILE=${PID_FILE} bash VRPRM_v2.0/benchmarks/visualprocessbench/serve_sft_vllm_dp.sh stop

EOF
}

configure_defaults

case "${ACTION}" in
  start)
    start_servers
    ;;
  stop)
    stop_servers
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
