#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/env.list" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.list"
fi

VRPRM_ROOT="${VRPRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-${VRPRM_ROOT}/ms-swift}"

MODEL_PATH="${THINKING_MODEL_PATH:-Qwen/Qwen3-VL-8B-Thinking}"
DATASET_PATH="${THINKING_MULTITURN_DATASET_PATH:-${VRPRM_ROOT}/rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35.json}"
OUTPUT_DIR="${THINKING_MULTITURN_OUTPUT_DIR:-${VRPRM_ROOT}/sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft}"

if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift directory not found: ${SWIFT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "Thinking model directory not found: ${MODEL_PATH}" >&2
  echo "Set THINKING_MODEL_PATH=/path/to/Qwen3-VL-8B-Thinking if needed." >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset file not found: ${DATASET_PATH}" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DATASET_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
no_think = any(sample.get("task_type") == "no_think_stepwise_multiturn" for sample in data)
bad = []
trainable_neg = trainable_pos = 0
for idx, sample in enumerate(data):
    messages = sample.get("messages") or []
    min_len = 3 if no_think else 5
    if len(messages) < min_len or messages[0].get("role") != "system":
        bad.append((idx, "bad_messages"))
        continue
    if not no_think:
        if messages[2].get("role") != "assistant" or not messages[2].get("content", "").lstrip().startswith("<think>"):
            bad.append((idx, "bad_global_thinking"))
    else:
        if messages[1].get("role") != "user" or "<think>" in messages[1].get("content", ""):
            bad.append((idx, "bad_no_think_first_turn"))
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").strip()
        if content in {"0", "1"} and msg.get("loss") is not False:
            if content == "0":
                trainable_neg += 1
            else:
                trainable_pos += 1
    if len(bad) >= 10:
        break
if bad:
    raise SystemExit(f"Dataset validation failed for {path}: {bad[:10]}")
total = trainable_neg + trainable_pos
ratio = trainable_neg / total if total else 0
mode = "no_think" if no_think else "global_think"
print(
    f"Validated multiturn dataset ({mode}): samples={len(data)}, "
    f"trainable_neg={trainable_neg}, trainable_pos={trainable_pos}, neg_ratio={ratio:.4f}"
)
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_TORCHRUN="${USE_TORCHRUN:-false}"
if [[ "${USE_TORCHRUN}" == "true" || "${USE_TORCHRUN}" == "1" ]]; then
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  export MASTER_PORT="${MASTER_PORT:-29504}"
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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export PYTHONPATH="${SWIFT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${SWIFT_ROOT}"

echo "Distributed launch:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  USE_TORCHRUN=${USE_TORCHRUN}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE:-unset}"
echo "  MASTER_PORT=${MASTER_PORT:-unset}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"

args=(
  sft
  --model "${MODEL_PATH}"
  --dataset "${DATASET_PATH}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE:-true}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.01}"
  --tuner_type "${TUNER_TYPE:-lora_llm}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --attn_impl "${ATTN_IMPL:-sdpa}"
  --packing "${PACKING:-false}"
  --padding_free "${PADDING_FREE:-false}"
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --lora_rank "${LORA_RANK:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --vit_gradient_checkpointing "${VIT_GRADIENT_CHECKPOINTING:-true}"
  --preserve_thinking true
  --add_non_thinking_prefix false
  --loss_scale "${LOSS_SCALE:-default}"
  --is_binary_loss_scale false
  --eval_steps "${EVAL_STEPS:-100}"
  --save_steps "${SAVE_STEPS:-500}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --max_length "${MAX_LENGTH:-12288}"
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
