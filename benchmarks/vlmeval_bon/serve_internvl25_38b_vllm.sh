#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACTION="${ACTION:-start}"
if [[ "${1:-}" == "start" || "${1:-}" == "stop" || "${1:-}" == "status" ]]; then
  ACTION="$1"
  shift
fi

MODEL_PATH="${1:-${MODEL_PATH:-OpenGVLab/InternVL2_5-38B}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-internvl25-38b}"
HOST="${HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 8}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
WAIT_READY="${WAIT_READY:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

POLICY_CONCURRENCY="${POLICY_CONCURRENCY:-64}"
POLICY_ROLLOUT_CONCURRENCY="${POLICY_ROLLOUT_CONCURRENCY:-1}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-2048}"
POLICY_REQUEST_TIMEOUT="${POLICY_REQUEST_TIMEOUT:-600}"
POLICY_MAX_RETRIES="${POLICY_MAX_RETRIES:-3}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/internvl25_38b_vllm_logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pid.txt}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

usage() {
  cat >&2 <<'EOF'
Usage:
  serve_internvl25_38b_vllm.sh [start] [model_path_or_hf_id]
  serve_internvl25_38b_vllm.sh stop
  serve_internvl25_38b_vllm.sh status

Environment overrides:
  MODEL_PATH                 InternVL2.5-38B local directory or Hugging Face id.
                             Default: OpenGVLab/InternVL2_5-38B
  SERVED_MODEL_NAME          Name exposed by the OpenAI-compatible API.
                             Default: internvl25-38b
  CUDA_VISIBLE_DEVICES       GPUs used by this one vLLM server. Default: 0,1,2,3
  TENSOR_PARALLEL_SIZE       vLLM tensor parallel size. Default: 4
  PORT                       Bind port. Default: 8000
  MAX_MODEL_LEN              vLLM max context length. Default: 8192
  MAX_NUM_SEQS               Concurrent sequences. Default: 32
  GPU_MEMORY_UTILIZATION     GPU memory fraction. Default: 0.90
  LIMIT_MM_PER_PROMPT        Multimodal limits. Default: {"image": 8}
  LOG_DIR                    vLLM log directory.
  VLLM_EXTRA_ARGS            Extra raw args appended to vllm serve.
EOF
}

count_visible_devices() {
  local devices="$1"
  if [[ -z "${devices}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "${devices}"
}

resolve_model_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
    return
  fi
  if [[ -d "${input_path}" ]]; then
    printf '%s\n' "$(cd "${input_path}" && pwd)"
    return
  fi
  local repo_relative_path="${REPO_ROOT}/${input_path}"
  if [[ -d "${repo_relative_path}" ]]; then
    printf '%s\n' "$(cd "${repo_relative_path}" && pwd)"
    return
  fi
  printf '%s\n' "${input_path}"
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
  local pid
  pid="$(cat "${PID_FILE}")"
  if pid_running "${pid}"; then
    echo "running pid=${pid} endpoint=http://${CLIENT_HOST}:${PORT}/v1 log=${LOG_DIR}/vllm_internvl25_38b_port${PORT}.log"
    return 0
  fi
  echo "stopped pid=${pid} endpoint=http://${CLIENT_HOST}:${PORT}/v1 log=${LOG_DIR}/vllm_internvl25_38b_port${PORT}.log"
  return 1
}

stop_server() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "No pid file: ${PID_FILE}"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if pid_running "${pid}"; then
    echo "Stopping pid=${pid}"
    kill "${pid}" || true
  fi
  rm -f "${PID_FILE}"
}

wait_for_ready() {
  if [[ "${WAIT_READY}" != "1" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; skip readiness check."
    return 0
  fi

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://${CLIENT_HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "InternVL2.5-38B vLLM server is ready."
      return 0
    fi
    echo "Waiting for InternVL2.5-38B vLLM server..."
    sleep "${POLL_INTERVAL}"
  done

  echo "Timed out waiting for vLLM server. Check logs under ${LOG_DIR}." >&2
  return 1
}

start_server() {
  local visible_gpu_count
  visible_gpu_count="$(count_visible_devices "${CUDA_VISIBLE_DEVICES}")"
  if [[ "${visible_gpu_count}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
    echo "CUDA_VISIBLE_DEVICES exposes ${visible_gpu_count} GPUs, but TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}." >&2
    echo "For the default InternVL2.5-38B setup, use CUDA_VISIBLE_DEVICES=0,1,2,3 with TENSOR_PARALLEL_SIZE=4." >&2
    exit 1
  fi

  MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
  if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    exit 1
  fi

  if [[ -f "${PID_FILE}" ]] && status >/dev/null 2>&1; then
    echo "Existing InternVL2.5-38B vLLM server appears to be running. Use:"
    echo "  ${BASH_SOURCE[0]} stop"
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  export CUDA_VISIBLE_DEVICES
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  local extra_args=()
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
  fi

  local args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --trust-remote-code
    --dtype "${DTYPE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
  )
  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    args+=(--enable-prefix-caching)
  fi
  args+=("${extra_args[@]}")

  local log_path="${LOG_DIR}/vllm_internvl25_38b_port${PORT}.log"

  echo "Serving InternVL2.5-38B with vLLM"
  echo "Model path: ${MODEL_PATH}"
  echo "Served model name: ${SERVED_MODEL_NAME}"
  echo "Endpoint: http://${CLIENT_HOST}:${PORT}/v1"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
  echo "Logs: ${log_path}"
  echo

  nohup vllm "${args[@]}" > "${log_path}" 2>&1 &
  echo "$!" > "${PID_FILE}"

  wait_for_ready

  cat <<EOF

Use these settings for Bo128 rollout generation:

  export POLICY_BASE_URL=http://${CLIENT_HOST}:${PORT}/v1
  export POLICY_MODEL=${SERVED_MODEL_NAME}
  export POLICY_CONCURRENCY=${POLICY_CONCURRENCY}
  export POLICY_ROLLOUT_CONCURRENCY=${POLICY_ROLLOUT_CONCURRENCY}
  export POLICY_MAX_TOKENS=${POLICY_MAX_TOKENS}
  export POLICY_REQUEST_TIMEOUT=${POLICY_REQUEST_TIMEOUT}
  export POLICY_MAX_RETRIES=${POLICY_MAX_RETRIES}
  bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_generate_rollouts_cache.sh

Stop the server with:

  bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_vllm.sh stop

EOF
}

case "${ACTION}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
