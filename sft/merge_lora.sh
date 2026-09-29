#!/usr/bin/env bash
# Merge an ms-swift LoRA checkpoint into a Hugging Face model ready for vLLM.
#
# Consolidates the previous merge_qwen3vl8b_visualprm_sft.sh and
# benchmarks/visualprocessbench/export_merged_sft_for_vllm.sh (both preserved
# on archive/snapshot-202609).
#
# Usage:
#   bash sft/merge_lora.sh <adapter_checkpoint_or_run_dir> [merged_output_dir]
#
# If a run directory with several checkpoint-* folders is given, the latest
# checkpoint that contains a LoRA adapter is selected automatically.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/env.list" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.list"
fi

VRPRM_ROOT="${VRPRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
OVERWRITE_MERGED="${OVERWRITE_MERGED:-0}"
FIX_VLLM_TOKENIZER="${FIX_VLLM_TOKENIZER:-1}"

ADAPTER_PATH="${1:-${ADAPTER_PATH:-}}"
MERGED_MODEL_DIR_INPUT="${2:-${MERGED_MODEL_DIR:-}}"

if [[ -z "${ADAPTER_PATH}" ]]; then
  cat >&2 <<'EOF'
Usage: bash sft/merge_lora.sh <adapter_checkpoint_or_run_dir> [merged_output_dir]

Environment overrides:
  ADAPTER_PATH       ms-swift LoRA checkpoint (or run dir with checkpoint-* subdirs)
  BASE_MODEL         Base HF model for merge + tokenizer/processor files. Default: Qwen/Qwen3-VL-8B-Thinking
  MERGED_MODEL_DIR   Output directory. Default: <adapter>-merged
  OVERWRITE_MERGED   Set 1 to recreate an existing output directory
  FIX_VLLM_TOKENIZER Set 0 to skip copying tokenizer/processor files from BASE_MODEL
EOF
  exit 2
fi

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

has_lora_adapter() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/adapter_config.json" || -f "${checkpoint_dir}/adapter_model.safetensors" ]]
}

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapter directory not found: ${ADAPTER_PATH}" >&2
  exit 1
fi

if ! has_lora_adapter "${ADAPTER_PATH}"; then
  latest=""
  while IFS= read -r checkpoint_dir; do
    if has_lora_adapter "${checkpoint_dir}"; then
      latest="${checkpoint_dir}"
    fi
  done < <(find "${ADAPTER_PATH}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ -n "${latest}" ]]; then
    echo "Using latest checkpoint in run directory: ${latest}"
    ADAPTER_PATH="${latest}"
  fi
fi

if ! has_lora_adapter "${ADAPTER_PATH}" ]]; then
  echo "Not an ms-swift LoRA checkpoint: ${ADAPTER_PATH}" >&2
  exit 1
fi

MERGED_MODEL_DIR="${MERGED_MODEL_DIR_INPUT:-${ADAPTER_PATH}-merged}"

if [[ -e "${MERGED_MODEL_DIR}" ]]; then
  if [[ "${OVERWRITE_MERGED}" == "1" ]]; then
    echo "Removing existing merged model directory: ${MERGED_MODEL_DIR}"
    rm -rf "${MERGED_MODEL_DIR}"
  else
    echo "Merged model directory already exists: ${MERGED_MODEL_DIR}" >&2
    echo "Set OVERWRITE_MERGED=1 to recreate it." >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"

echo "Merging ms-swift LoRA checkpoint"
echo "  adapter: ${ADAPTER_PATH}"
echo "  base   : ${BASE_MODEL}"
echo "  output : ${MERGED_MODEL_DIR}"

cd "${SWIFT_ROOT}"

python "${SWIFT_ROOT}/swift/cli/main.py" export \
  --model "${BASE_MODEL}" \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true \
  --output_dir "${MERGED_MODEL_DIR}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl "${ATTN_IMPL:-sdpa}" \
  --max_shard_size "${MAX_SHARD_SIZE}" \
  --safe_serialization "${SAFE_SERIALIZATION:-true}" \
  --exist_ok true \
  --device_map "${MERGE_DEVICE_MAP:-auto}"

if [[ "${FIX_VLLM_TOKENIZER}" == "1" ]]; then
  python "${SCRIPT_DIR}/fix_vllm_merged_tokenizer.py" \
    --merged-model-dir "${MERGED_MODEL_DIR}" \
    --base-model "${BASE_MODEL}"
fi

[[ -f "${MERGED_MODEL_DIR}/config.json" ]] || { echo "Missing config.json in ${MERGED_MODEL_DIR}" >&2; exit 1; }
compgen -G "${MERGED_MODEL_DIR}/*.safetensors" >/dev/null || { echo "Missing safetensors weights in ${MERGED_MODEL_DIR}" >&2; exit 1; }

echo "Merged model ready: ${MERGED_MODEL_DIR}"
echo "Serve it with: bash benchmarks/visualprocessbench/serve_vllm.sh ${MERGED_MODEL_DIR}"
