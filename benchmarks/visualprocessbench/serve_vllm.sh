#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 8}}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./serve_sft_vllm_manual.sh <model_path_or_hf_id>

Environment overrides:
  MODEL_PATH                 Model directory or Hugging Face model id.
  SERVED_MODEL_NAME          Name exposed by the OpenAI-compatible API.
  HOST                       Bind host. Default: 0.0.0.0
  PORT                       Bind port. Default: 8000
  CUDA_VISIBLE_DEVICES       GPUs to use. Default: 0,1
  TENSOR_PARALLEL_SIZE       vLLM tensor parallel size. Default: 2
  MAX_MODEL_LEN              Max context length. Default: 8192
  MAX_NUM_SEQS               Max concurrent sequences. Default: 8
  GPU_MEMORY_UTILIZATION     GPU memory fraction. Default: 0.90
  LIMIT_MM_PER_PROMPT        Multimodal limits. Default: {"image": 8}
EOF
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

if [[ -z "${MODEL_PATH}" ]]; then
  usage
  exit 2
fi

MODEL_PATH="$(resolve_model "${MODEL_PATH}")"

if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -z "${SERVED_MODEL_NAME}" ]]; then
  SERVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
fi

echo "Serving model with vLLM"
echo "Model: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --trust-remote-code \
  --mm-processor-cache-gb 0
