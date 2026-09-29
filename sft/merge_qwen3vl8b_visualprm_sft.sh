#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/env.list" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.list"
fi

VRPRM_ROOT="${VRPRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"
OUTPUT_DIR="${OUTPUT_DIR:-${VRPRM_ROOT}/sft/output/qwen3_vl_8b_thinking_visualprm_sft}"

find_latest_checkpoint() {
  local root="$1"
  find "${root}" -type f -name adapter_config.json ! -path '*-merged/*' -printf '%T@ %h\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

CKPT_DIR="${1:-${CKPT_DIR:-}}"
if [[ -z "${CKPT_DIR}" ]]; then
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory not found: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  CKPT_DIR="$(find_latest_checkpoint "${OUTPUT_DIR}")"
fi

if [[ -z "${CKPT_DIR}" ]]; then
  echo "No adapter checkpoint found under: ${OUTPUT_DIR}" >&2
  exit 1
fi

MERGED_OUTPUT_DIR="${2:-${MERGED_OUTPUT_DIR:-${CKPT_DIR}-merged}}"

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

CKPT_DIR="$(cd "${CKPT_DIR}" && pwd)"
MERGED_OUTPUT_DIR="$(readlink -m "${MERGED_OUTPUT_DIR}")"

if [[ ! -f "${CKPT_DIR}/adapter_config.json" ]]; then
  echo "Checkpoint is not a LoRA adapter checkpoint: ${CKPT_DIR}" >&2
  echo "Expected: ${CKPT_DIR}/adapter_config.json" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"
export MERGE_DEVICE_MAP="${MERGE_DEVICE_MAP:-auto}"

mkdir -p "$(dirname "${MERGED_OUTPUT_DIR}")"

cd "${SWIFT_ROOT}"

args=(
  export
  --adapters "${CKPT_DIR}"
  --merge_lora true
  --output_dir "${MERGED_OUTPUT_DIR}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --attn_impl "${ATTN_IMPL:-sdpa}"
  --safe_serialization "${SAFE_SERIALIZATION:-true}"
  --max_shard_size "${MAX_SHARD_SIZE:-5GB}"
)

if [[ "${EXIST_OK:-false}" == "true" ]]; then
  args+=(--exist_ok true)
fi

args+=(--device_map "${MERGE_DEVICE_MAP}")

if [[ -n "${MAX_MEMORY:-}" ]]; then
  args+=(--max_memory "${MAX_MEMORY}")
fi

echo "Merging checkpoint:"
echo "  adapter: ${CKPT_DIR}"
echo "  output : ${MERGED_OUTPUT_DIR}"
echo "  gpu    : ${CUDA_VISIBLE_DEVICES}"
echo "  device : ${MERGE_DEVICE_MAP}"
if [[ -n "${MAX_MEMORY:-}" ]]; then
  echo "  memory : ${MAX_MEMORY}"
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  printf 'Command:'
  printf ' %q' python "${SWIFT_ROOT}/swift/cli/main.py" "${args[@]}"
  printf '\n'
  exit 0
fi

python "${SWIFT_ROOT}/swift/cli/main.py" "${args[@]}"

echo "Merged Hugging Face model saved to: ${MERGED_OUTPUT_DIR}"
