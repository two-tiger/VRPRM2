#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/env.list" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.list"
fi

VRPRM_ROOT="${VRPRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET_PATHS="${DATASET_PATHS:-${VRPRM_ROOT}/rollout_outputs/rollout_sft_success_0615.json,${VRPRM_ROOT}/rollout_outputs/rollout_sft_success_0614.json}"
MERGED_DATASET_PATH="${MERGED_DATASET_PATH:-${VRPRM_ROOT}/rollout_outputs/rollout_sft_success_0614_0615.json}"
DATASET_PATH="${DATASET_PATH:-${MERGED_DATASET_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${VRPRM_ROOT}/sft/output/qwen3_vl_8b_visualprm_sft}"

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" && -n "${DATASET_PATHS}" ]]; then
  echo "Merged dataset not found: ${DATASET_PATH}"
  echo "Creating it from: ${DATASET_PATHS}"
  DATASET_PATHS="${DATASET_PATHS}" DATASET_PATH="${DATASET_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

dataset_paths = [Path(item.strip()) for item in os.environ["DATASET_PATHS"].split(",") if item.strip()]
output_path = Path(os.environ["DATASET_PATH"])
merged = []
for path in dataset_paths:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list, got {type(data).__name__}")
    merged.extend(data)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote merged dataset: {output_path} ({len(merged)} samples)")
PY
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file not found: ${DATASET_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

cd "${SWIFT_ROOT}"

args=(
  sft
  --model "${MODEL_PATH}"
  --dataset "${DATASET_PATH}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE:-true}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.01}"
  --tuner_type "${TUNER_TYPE:-lora_llm}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
  --attn_impl "${ATTN_IMPL:-sdpa}"
  --packing "${PACKING:-false}"
  --padding_free "${PADDING_FREE:-false}"
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --lora_rank "${LORA_RANK:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --vit_gradient_checkpointing "${VIT_GRADIENT_CHECKPOINTING:-true}"
  --add_non_thinking_prefix false
  --eval_steps "${EVAL_STEPS:-100}"
  --save_steps "${SAVE_STEPS:-500}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --max_length "${MAX_LENGTH:-8192}"
  --output_dir "${OUTPUT_DIR}"
  --warmup_ratio "${WARMUP_RATIO:-0.05}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
  --dataset_num_proc "${DATASET_NUM_PROC:-8}"
  --save_only_model true
)

if [[ -n "${DEEPSPEED:-}" && "${DEEPSPEED}" != "false" && "${DEEPSPEED}" != "0" && "${DEEPSPEED}" != "none" ]]; then
  args+=(--deepspeed "${DEEPSPEED}")
fi

if [[ -n "${VIT_LR:-}" ]]; then
  args+=(--vit_lr "${VIT_LR}")
fi

if [[ -n "${ALIGNER_LR:-}" ]]; then
  args+=(--aligner_lr "${ALIGNER_LR}")
fi

python "${SWIFT_ROOT}/swift/cli/main.py" "${args[@]}"
