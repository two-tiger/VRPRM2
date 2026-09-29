#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-1}"
DP="${DP:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-1024}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

PRM_CONCURRENCY="${PRM_CONCURRENCY:-128}"
PRM_MODEL="${PRM_MODEL:-}"
PRM_REWARD_MODE="${PRM_REWARD_MODE:-stepwise}"
PRM_GUIDED_CHOICE="${PRM_GUIDED_CHOICE:-0}"
PRM_WARMUP_MAX_TOKENS="${PRM_WARMUP_MAX_TOKENS:-1024}"
PRM_REQUEST_TIMEOUT="${PRM_REQUEST_TIMEOUT:-300}"
PRM_MAX_RETRIES="${PRM_MAX_RETRIES:-3}"

usage() {
  cat >&2 <<'EOF'
Usage:
  serve_prm_sglang_dp4.sh <model_path_or_hf_id>

Environment overrides:
  MODEL_PATH                 PRM model directory or Hugging Face model id.
  SERVED_MODEL_NAME          Name exposed by the OpenAI-compatible API.
  HOST                       Bind host. Default: 0.0.0.0
  PORT                       Bind port. Default: 8001
  CLIENT_HOST                Host printed for client use. Default: 127.0.0.1
  CUDA_VISIBLE_DEVICES       Four GPUs to use. Default: 0,1,2,3
  TP                         SGLang tensor parallel size. Default: 1
  DP                         SGLang data parallel size. Default: 4
  MEM_FRACTION_STATIC        SGLang static memory fraction. Default: 0.88
  MAX_RUNNING_REQUESTS       Max active requests. Default: 128
  MAX_QUEUED_REQUESTS        Max queued requests. Default: 1024
  SGLANG_EXTRA_ARGS          Extra raw args appended to sglang.launch_server.
  PRM_CONCURRENCY            Suggested scorer concurrency. Default: 128
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
  printf '%s\n' "${input_path}"
}

if [[ -z "${MODEL_PATH}" ]]; then
  usage
  exit 2
fi

MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
VISIBLE_GPU_COUNT="$(count_visible_devices "${CUDA_VISIBLE_DEVICES}")"
REQUIRED_GPU_COUNT=$((TP * DP))

if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${VISIBLE_GPU_COUNT}" -ne "${REQUIRED_GPU_COUNT}" ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT} GPUs, but TP * DP = ${REQUIRED_GPU_COUNT}." >&2
  echo "For the requested DP=4 setup, use CUDA_VISIBLE_DEVICES=0,1,2,3 with TP=1 DP=4." >&2
  exit 1
fi

if [[ -z "${SERVED_MODEL_NAME}" ]]; then
  SERVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
fi
if [[ -z "${PRM_MODEL}" ]]; then
  PRM_MODEL="${SERVED_MODEL_NAME}"
fi

export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp "${TP}"
  --dp "${DP}"
  --port "${PORT}"
  --host "${HOST}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ -n "${SGLANG_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${SGLANG_EXTRA_ARGS}"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Serving PRM with SGLang DP=4"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TP=${TP}"
echo "DP=${DP}"
echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
echo "MAX_QUEUED_REQUESTS=${MAX_QUEUED_REQUESTS}"
echo
echo "Use this endpoint with the BoN scoring pipeline in another shell:"
echo "  export PRM_BASE_URL=http://${CLIENT_HOST}:${PORT}/v1"
echo "  export PRM_MODEL=${PRM_MODEL}"
echo "  export PRM_CONCURRENCY=${PRM_CONCURRENCY}"
echo "  export PRM_REWARD_MODE=${PRM_REWARD_MODE}"
echo "  export PRM_GUIDED_CHOICE=${PRM_GUIDED_CHOICE}"
echo "  export PRM_WARMUP_MAX_TOKENS=${PRM_WARMUP_MAX_TOKENS}"
echo "  export PRM_REQUEST_TIMEOUT=${PRM_REQUEST_TIMEOUT}"
echo "  export PRM_MAX_RETRIES=${PRM_MAX_RETRIES}"
echo "  bash VRPRM_v2.0/benchmarks/vlmeval_bon/run_score_rollouts_cache.sh"
echo

python "${ARGS[@]}"
