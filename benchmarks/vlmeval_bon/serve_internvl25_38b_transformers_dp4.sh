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
PORT_BASE="${PORT_BASE:-8000}"
PORTS="${PORTS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"

MODEL_NAME_FOR_SPLIT="${MODEL_NAME_FOR_SPLIT:-InternVL2_5-38B}"
DEVICE_MAP="${DEVICE_MAP:-split}"
DTYPE="${DTYPE:-bf16}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
MAX_IMAGE_TILES="${MAX_IMAGE_TILES:-12}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"

POLICY_CONCURRENCY="${POLICY_CONCURRENCY:-4}"
POLICY_ROLLOUT_CONCURRENCY="${POLICY_ROLLOUT_CONCURRENCY:-1}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-2048}"
POLICY_REQUEST_TIMEOUT="${POLICY_REQUEST_TIMEOUT:-1800}"
POLICY_MAX_RETRIES="${POLICY_MAX_RETRIES:-3}"

WAIT_READY="${WAIT_READY:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/internvl25_38b_transformers_dp4_logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pids.txt}"

usage() {
  cat >&2 <<'EOF'
Usage:
  serve_internvl25_38b_transformers_dp4.sh [start] [model_path_or_hf_id]
  serve_internvl25_38b_transformers_dp4.sh stop
  serve_internvl25_38b_transformers_dp4.sh status

Environment overrides:
  MODEL_PATH                 InternVL2.5-38B local directory or Hugging Face id.
                             Default: OpenGVLab/InternVL2_5-38B
  SERVED_MODEL_NAME          Name exposed by each /v1/models. Default: internvl25-38b
  CUDA_VISIBLE_DEVICES       GPUs used as independent replicas. Default: 0,1,2,3
  NUM_REPLICAS               Number of model replicas. Default: 4
  PORT_BASE                  First port when PORTS is unset. Default: 8000
  PORTS                      Comma-separated ports, e.g. 8000,8001,8002,8003
  DEVICE_MAP                 split or auto inside each single-GPU process. Default: split
  DTYPE                      bf16 or fp16. Default: bf16
  USE_FLASH_ATTN             Pass use_flash_attn=True. Default: 1
  LOAD_IN_8BIT               Enable bitsandbytes 8-bit loading. Default: 0
  MAX_IMAGE_TILES            InternVL dynamic image tiles per image. Default: 12
  POLICY_CONCURRENCY         Suggested rollout concurrency. Default: 4
  LOG_DIR                    Server log directory.

Notes:
  This starts one transformers server per GPU. Each process sees only one GPU
  and serves the same OpenAI-compatible /v1/chat/completions API.
EOF
}

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${raw}"
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
      echo "All ${ready} InternVL2.5-38B transformers replicas are ready."
      return 0
    fi
    echo "Waiting for InternVL2.5-38B transformers replicas: ${ready}/${#ports_ref[@]} ready..."
    sleep "${POLL_INTERVAL}"
  done

  echo "Timed out waiting for transformers replicas. Check logs under ${LOG_DIR}." >&2
  return 1
}

start_servers() {
  local devices=()
  local ports=()
  local urls=()
  split_csv "${CUDA_VISIBLE_DEVICES}" devices
  build_ports ports
  build_base_urls ports urls

  if [[ "${#devices[@]}" -ne "${NUM_REPLICAS}" ]]; then
    echo "CUDA_VISIBLE_DEVICES exposes ${#devices[@]} GPUs, but NUM_REPLICAS=${NUM_REPLICAS}." >&2
    echo "Set CUDA_VISIBLE_DEVICES and NUM_REPLICAS consistently." >&2
    exit 1
  fi
  if [[ "${#ports[@]}" -ne "${NUM_REPLICAS}" ]]; then
    echo "PORTS exposes ${#ports[@]} ports, but NUM_REPLICAS=${NUM_REPLICAS}." >&2
    exit 1
  fi

  MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
  if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    exit 1
  fi

  if [[ -f "${PID_FILE}" ]] && status >/dev/null 2>&1; then
    echo "Existing InternVL2.5-38B transformers replicas appear to be running. Use:"
    echo "  ${BASH_SOURCE[0]} stop"
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  : > "${PID_FILE}"
  export USE_FLASH_ATTN
  export LOAD_IN_8BIT
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

  echo "Serving InternVL2.5-38B with ${NUM_REPLICAS} transformers data-parallel replicas"
  echo "Model path: ${MODEL_PATH}"
  echo "Served model name: ${SERVED_MODEL_NAME}"
  echo "GPUs: $(join_by_comma "${devices[@]}")"
  echo "Ports: $(join_by_comma "${ports[@]}")"
  echo "DEVICE_MAP=${DEVICE_MAP}"
  echo "DTYPE=${DTYPE}"
  echo "USE_FLASH_ATTN=${USE_FLASH_ATTN}"
  echo "LOAD_IN_8BIT=${LOAD_IN_8BIT}"
  echo "MAX_IMAGE_TILES=${MAX_IMAGE_TILES}"
  echo "Logs: ${LOG_DIR}"
  echo

  for ((i = 0; i < NUM_REPLICAS; i += 1)); do
    local gpu="${devices[$i]}"
    local port="${ports[$i]}"
    local log_path="${LOG_DIR}/internvl25_38b_transformers_gpu${gpu}_port${port}.log"
    echo "Starting replica $((i + 1))/${NUM_REPLICAS}: gpu=${gpu} port=${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python "${SCRIPT_DIR}/serve_internvl25_38b_transformers.py" \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${port}" \
      --model-name-for-split "${MODEL_NAME_FOR_SPLIT}" \
      --device-map "${DEVICE_MAP}" \
      --dtype "${DTYPE}" \
      --max-image-tiles "${MAX_IMAGE_TILES}" \
      --image-size "${IMAGE_SIZE}" \
      --default-max-new-tokens "${POLICY_MAX_TOKENS}" \
      > "${log_path}" 2>&1 &
    echo "$! ${gpu} ${port} ${log_path}" >> "${PID_FILE}"
  done

  wait_for_ready ports

  local policy_base_url
  policy_base_url="$(join_by_comma "${urls[@]}")"
  cat <<EOF

Use these settings for Bo128 rollout generation:

  export POLICY_BASE_URL=${policy_base_url}
  export POLICY_MODEL=${SERVED_MODEL_NAME}
  export POLICY_CONCURRENCY=${POLICY_CONCURRENCY}
  export POLICY_ROLLOUT_CONCURRENCY=${POLICY_ROLLOUT_CONCURRENCY}
  export POLICY_MAX_TOKENS=${POLICY_MAX_TOKENS}
  export POLICY_REQUEST_TIMEOUT=${POLICY_REQUEST_TIMEOUT}
  export POLICY_MAX_RETRIES=${POLICY_MAX_RETRIES}
  bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_generate_rollouts_cache.sh

Stop replicas with:

  bash VRPRM_v2.0/benchmarks/vlmeval_bon/serve_internvl25_38b_transformers_dp4.sh stop

EOF
}

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
