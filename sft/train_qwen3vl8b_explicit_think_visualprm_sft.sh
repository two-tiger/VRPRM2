#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ -f "${SCRIPT_DIR}/env.list" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.list"
fi

VRPRM_ROOT="${VRPRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"

# Dedicated variables keep this explicit-<think> run independent from env.list's
# default rollout_sft_success_0614_0615.json settings.
MODEL_PATH="${EXPLICIT_THINK_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}}"
DATASET_PATH="${EXPLICIT_THINK_DATASET_PATH:-${VRPRM_ROOT}/rollout_outputs/visualprm400k_pos_neg_rollout_sft_success_shuffled_with_system_think.json}"
OUTPUT_DIR="${EXPLICIT_THINK_OUTPUT_DIR:-${VRPRM_ROOT}/sft/output/qwen3_vl_8b_explicit_think_visualprm_sft}"
REQUIRE_THINK_PREFIX="${REQUIRE_THINK_PREFIX:-true}"
REQUIRE_SYSTEM_THINK_PROMPT="${REQUIRE_SYSTEM_THINK_PROMPT:-true}"

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "Qwen3-VL-8B model directory not found: ${MODEL_PATH}" >&2
  echo "Set EXPLICIT_THINK_MODEL_PATH=/path/to/Qwen3-VL-8B if needed." >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file not found: ${DATASET_PATH}" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH}" \
REQUIRE_THINK_PREFIX="${REQUIRE_THINK_PREFIX}" \
REQUIRE_SYSTEM_THINK_PROMPT="${REQUIRE_SYSTEM_THINK_PROMPT}" \
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DATASET_PATH"])
require_think_prefix = os.environ.get("REQUIRE_THINK_PREFIX", "true").lower() not in {"0", "false", "no"}
require_system_think_prompt = os.environ.get("REQUIRE_SYSTEM_THINK_PROMPT", "true").lower() not in {
    "0",
    "false",
    "no",
}

data = json.loads(path.read_text(encoding="utf-8"))
bad_assistant = []
bad_system = []
for idx, sample in enumerate(data):
    messages = sample.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    assistant = messages[-1].get("content", "") if messages else ""

    if require_system_think_prompt and ("<think>" not in system or "</think>" not in system):
        bad_system.append(idx)
        if len(bad_system) >= 10:
            break

    if require_think_prefix and not assistant.lstrip().startswith("<think>"):
        bad_assistant.append(idx)
        if len(bad_assistant) >= 10:
            break

if bad_system:
    raise SystemExit(
        "Dataset validation failed: system prompt must explicitly require '<think>...</think>'. "
        f"First bad sample indexes: {bad_system}"
    )

if bad_assistant:
    raise SystemExit(
        "Dataset validation failed: assistant content must start with '<think>'. "
        f"First bad sample indexes: {bad_assistant}"
    )

print(
    "Validated explicit-think SFT dataset: "
    f"{len(data)} samples, system_think_prompt={require_system_think_prompt}, "
    f"assistant_think_prefix={require_think_prefix}: {path}"
)
PY

export CUDA_VISIBLE_DEVICES="${EXPLICIT_THINK_CUDA_VISIBLE_DEVICES:-${USER_CUDA_VISIBLE_DEVICES:-1}}"
USE_TORCHRUN="${USE_TORCHRUN:-false}"
if [[ "${USE_TORCHRUN}" == "true" || "${USE_TORCHRUN}" == "1" ]]; then
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  export MASTER_PORT="${MASTER_PORT:-29503}"
else
  unset NPROC_PER_NODE
  unset NNODES
  unset NODE_RANK
  unset MASTER_ADDR
  unset RANK
  unset LOCAL_RANK
  unset WORLD_SIZE
  unset LOCAL_WORLD_SIZE
fi
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
  --preserve_thinking true
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
