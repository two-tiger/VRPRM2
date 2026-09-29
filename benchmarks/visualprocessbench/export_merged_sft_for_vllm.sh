#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"

export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

ADAPTER_PATH="${1:-${ADAPTER_PATH:-}}"
BASE_MODEL="${BASE_MODEL:-${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
MERGED_MODEL_DIR_INPUT="${MERGED_MODEL_DIR}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
OVERWRITE_MERGED="${OVERWRITE_MERGED:-0}"
FIX_VLLM_TOKENIZER="${FIX_VLLM_TOKENIZER:-1}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./export_merged_sft_for_vllm.sh <adapter_checkpoint_dir>

Environment overrides:
  ADAPTER_PATH         ms-swift LoRA checkpoint directory. Used when no argument is given.
                       If this points to a run directory, the latest checkpoint-* is selected.
  BASE_MODEL           Base HF model directory used for LoRA merge and tokenizer/processor files.
  MODEL_PATH           Alias fallback for BASE_MODEL.
  MERGED_MODEL_DIR     Output merged HF model directory. Default: <adapter_checkpoint_dir>-merged
  TORCH_DTYPE          Merge dtype passed to ms-swift. Default: bfloat16
  MAX_SHARD_SIZE       Output shard size passed to ms-swift. Default: 5GB
  OVERWRITE_MERGED     Set to 1 to delete and recreate MERGED_MODEL_DIR if it exists.
  FIX_VLLM_TOKENIZER   Set to 0 to skip copying tokenizer/processor files from BASE_MODEL. Default: 1
EOF
}

resolve_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  elif [[ -d "${input_path}" || -f "${input_path}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${input_path}")" && pwd)/$(basename "${input_path}")"
  else
    printf '%s\n' "$(cd "${VRPRM_ROOT}/.." && pwd)/${input_path}"
  fi
}

has_lora_adapter() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/adapter_config.json" || -f "${checkpoint_dir}/adapter_model.safetensors" ]]
}

latest_checkpoint_in() {
  local run_dir="$1"
  local checkpoint_dir
  local latest=""
  while IFS= read -r checkpoint_dir; do
    if has_lora_adapter "${checkpoint_dir}"; then
      latest="${checkpoint_dir}"
    fi
  done < <(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  printf '%s\n' "${latest}"
}

if [[ -z "${ADAPTER_PATH}" ]]; then
  usage
  exit 2
fi

ADAPTER_PATH="$(resolve_path "${ADAPTER_PATH}")"
BASE_MODEL="$(resolve_path "${BASE_MODEL}")"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapter directory not found: ${ADAPTER_PATH}" >&2
  exit 1
fi

if ! has_lora_adapter "${ADAPTER_PATH}"; then
  LATEST_CHECKPOINT="$(latest_checkpoint_in "${ADAPTER_PATH}")"
  if [[ -n "${LATEST_CHECKPOINT}" && -d "${LATEST_CHECKPOINT}" ]] && has_lora_adapter "${LATEST_CHECKPOINT}"; then
    echo "Using latest checkpoint in run directory: ${LATEST_CHECKPOINT}"
    ADAPTER_PATH="${LATEST_CHECKPOINT}"
  fi
fi

if ! has_lora_adapter "${ADAPTER_PATH}"; then
  echo "Adapter checkpoint does not look like an ms-swift LoRA checkpoint: ${ADAPTER_PATH}" >&2
  echo "Expected adapter_config.json or adapter_model.safetensors." >&2
  exit 1
fi

MERGED_MODEL_DIR="${MERGED_MODEL_DIR_INPUT:-${ADAPTER_PATH}-merged}"
MERGED_MODEL_DIR="$(resolve_path "${MERGED_MODEL_DIR}")"

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

if [[ -e "${MERGED_MODEL_DIR}" ]]; then
  if [[ "${OVERWRITE_MERGED}" == "1" ]]; then
    echo "Removing existing merged model directory: ${MERGED_MODEL_DIR}"
    rm -rf "${MERGED_MODEL_DIR}"
  else
    echo "Merged model directory already exists: ${MERGED_MODEL_DIR}" >&2
    echo "Set OVERWRITE_MERGED=1 to recreate it, or set MERGED_MODEL_DIR to a new output path." >&2
    exit 1
  fi
fi

echo "Merging ms-swift LoRA SFT checkpoint for vLLM"
echo "Adapter: ${ADAPTER_PATH}"
echo "Base:    ${BASE_MODEL}"
echo "Output:  ${MERGED_MODEL_DIR}"
echo "Dtype:   ${TORCH_DTYPE}"

cd "${SWIFT_ROOT}"

python "${SWIFT_ROOT}/swift/cli/main.py" export \
  --model "${BASE_MODEL}" \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true \
  --output_dir "${MERGED_MODEL_DIR}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --max_shard_size "${MAX_SHARD_SIZE}" \
  --exist_ok true

if [[ "${FIX_VLLM_TOKENIZER}" == "1" ]]; then
  python "${SCRIPT_DIR}/fix_vllm_merged_tokenizer.py" \
    --merged-model-dir "${MERGED_MODEL_DIR}" \
    --base-model "${BASE_MODEL}"
fi

if [[ ! -f "${MERGED_MODEL_DIR}/config.json" ]]; then
  echo "Merged output is missing config.json: ${MERGED_MODEL_DIR}" >&2
  exit 1
fi

if ! compgen -G "${MERGED_MODEL_DIR}/*.safetensors" >/dev/null; then
  echo "Merged output is missing safetensors weights: ${MERGED_MODEL_DIR}" >&2
  exit 1
fi

echo "Merged model is ready: ${MERGED_MODEL_DIR}"
echo "Serve it with:"
echo "  bash ${SCRIPT_DIR}/serve_sft_vllm_manual.sh ${MERGED_MODEL_DIR}"
