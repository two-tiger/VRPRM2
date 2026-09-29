#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VRPRM_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${VRPRM_ROOT}/.." && pwd)"

DEFAULT_SFT_MODEL="${VRPRM_ROOT}/sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft/v0-20260628-015250/checkpoint-513-merge"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Thinking}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_SFT_MODEL}}"

IMAGE_DIR="${IMAGE_DIR:-${VRPRM_ROOT}/data}"
DATASET_DIR="${DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_cached_sft_think_step_source_macro_wemath_60k}"

export MAX_STEPS="${MAX_STEPS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export VAL_FREQ="${VAL_FREQ:-50}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export VAL_GENERATIONS_TO_LOG="${VAL_GENERATIONS_TO_LOG:-16}"
export TRAIN_ROLLOUTS_TO_LOG="${TRAIN_ROLLOUTS_TO_LOG:-16}"
export TRAIN_ROLLOUT_LOG_FREQ="${TRAIN_ROLLOUT_LOG_FREQ:-10}"
export GENERATION_LOG_MAX_CHARS="${GENERATION_LOG_MAX_CHARS:-6000}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

export LEARNING_RATE="${LEARNING_RATE:-1e-8}"
export KL_COEF="${KL_COEF:-2.0e-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4}"
export SOURCE_MACRO_WEIGHT="${SOURCE_MACRO_WEIGHT:-0.75}"
export STEP_WEIGHT="${STEP_WEIGHT:-0.20}"
export FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.05}"

export ROLLOUT_N="${ROLLOUT_N:-16}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export MINI_ROLLOUT_BATCH_SIZE="${MINI_ROLLOUT_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE_UPDATE="${MICRO_BATCH_SIZE_UPDATE:-4}"
export MICRO_BATCH_SIZE_EXPERIENCE="${MICRO_BATCH_SIZE_EXPERIENCE:-4}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"

# The original experiment name contains a bad-data run whose train split had only 307
# rows and a 92% negative ratio. Use a new default path so this script starts from
# the SFT model unless EXPERIMENT_NAME/SAVE_CHECKPOINT_PATH are explicitly set.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_8b_cached_sft_think_step_source_macro_visualprm400k_gspo_lora_wemath_300_fixdata}"
export SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${EASYR1_ROOT}/checkpoints/visualprm_easy_r1/${EXPERIMENT_NAME}}"

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != Qwen/* ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to a merged SFT Hugging Face directory or a Hub model id." >&2
  exit 1
fi

cd "${EASYR1_ROOT}"

if [[ -d "${MODEL_PATH}" && -f "${MODEL_PATH}/config.json" && -f "${BASE_MODEL_PATH}/config.json" ]]; then
  python3 scripts/fix_qwen3vl_config_rope.py \
    "${MODEL_PATH}" \
    --reference-config "${BASE_MODEL_PATH}/config.json"
fi

if [[ ! -f "${DATASET_DIR}/train.jsonl" || ! -f "${DATASET_DIR}/test.jsonl" || ! -f "${DATASET_DIR}/.visualprm_cached_think_step_paths_v1" ]]; then
  echo "Processed cached-thinking source-macro RL dataset not found: ${DATASET_DIR}" >&2
  echo "Prepare it first with:" >&2
  echo "  SFT_ROLLOUT_BASE_URLS=http://host:8000/v1,http://host:8001/v1 bash examples/prepare_cached_sft_think_step_source_macro_visualprm400k_wemath.sh" >&2
  exit 1
fi

python3 - "${DATASET_DIR}" "${MIN_TRAIN_ROWS:-50000}" "${MIN_VAL_ROWS:-1000}" "${TARGET_NEGATIVE_STEP_RATIO:-0.28}" "${NEGATIVE_RATIO_TOLERANCE:-0.03}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

dataset_dir = Path(sys.argv[1])
min_train_rows = int(sys.argv[2])
min_val_rows = int(sys.argv[3])
target_neg_ratio = float(sys.argv[4])
neg_ratio_tolerance = float(sys.argv[5])


def global_thinking(prompt: str) -> str:
    marker = "[Global Thinking]\n"
    if marker not in prompt:
        return ""
    section = prompt.split(marker, 1)[1]
    return section.split("\n\n[Previous Step Judgments]", 1)[0]


def inspect_split(split: str, min_rows: int) -> tuple[int, float, Counter]:
    path = dataset_dir / f"{split}.jsonl"
    labels = Counter()
    thinking_modes = Counter()
    score_artifacts = 0
    bad_think_blocks = 0
    rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ground_truth = row.get("ground_truth")
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            label = int(ground_truth["label"])
            labels[label] += 1
            thinking_modes[ground_truth.get("thinking_mode", "unknown")] += 1
            rows += 1

            think = global_thinking(row.get("prompt", ""))
            if think.count("<think>") != 1 or think.count("</think>") != 1:
                bad_think_blocks += 1
            if '"Score"' in think or "'Score'" in think or "Score:" in think:
                score_artifacts += 1

    if rows < min_rows:
        raise SystemExit(f"{split}.jsonl has only {rows} rows; expected at least {min_rows}. Regenerate data before training.")
    if bad_think_blocks:
        raise SystemExit(f"{split}.jsonl contains {bad_think_blocks} malformed global thinking blocks.")
    if score_artifacts:
        raise SystemExit(f"{split}.jsonl contains {score_artifacts} global thinking score artifacts.")

    neg_ratio = labels[0] / max(labels[0] + labels[1], 1)
    if abs(neg_ratio - target_neg_ratio) > neg_ratio_tolerance:
        raise SystemExit(
            f"{split}.jsonl negative ratio is {neg_ratio:.4f}, outside target "
            f"{target_neg_ratio:.4f} +/- {neg_ratio_tolerance:.4f}."
        )
    return rows, neg_ratio, thinking_modes


train_rows, train_neg_ratio, train_modes = inspect_split("train", min_train_rows)
val_rows, val_neg_ratio, val_modes = inspect_split("test", min_val_rows)
print(
    "Dataset sanity check passed: "
    f"train_rows={train_rows}, train_neg_ratio={train_neg_ratio:.4f}, train_modes={dict(train_modes)}, "
    f"val_rows={val_rows}, val_neg_ratio={val_neg_ratio:.4f}, val_modes={dict(val_modes)}"
)
PY

export PYTHONPATH="${EASYR1_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_vrprm_cached_sm}"
if (( ${#RAY_TMPDIR} > 50 )); then
  echo "RAY_TMPDIR is too long for Ray socket paths, overriding: ${RAY_TMPDIR}" >&2
  RAY_TMPDIR="/tmp/ray_vrprm_csm"
fi
export RAY_TMPDIR
mkdir -p "${RAY_TMPDIR}"

CACHE_ROOT="${CACHE_ROOT:-${VRPRM_ROOT}/cache/vrprm_rl}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_WORKSPACE_BASE}/cubins"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}" "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_CUBIN_DIR}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${VRPRM_ROOT}/wandb}"

ROLLOUT_EXTRA_SAMPLING_PARAMS="${ROLLOUT_EXTRA_SAMPLING_PARAMS:-{\"guided_regex\":\"[01]\"}}"

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${DATASET_DIR}/train.jsonl" \
  data.val_files="${DATASET_DIR}/test.jsonl" \
  data.prompt_key=prompt \
  data.answer_key=ground_truth \
  data.image_key=images \
  data.shuffle="${DATA_SHUFFLE:-false}" \
  data.image_dir="${IMAGE_DIR}" \
  data.format_prompt=./examples/format_prompt/visualprm_step_level.jinja \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
  data.mini_rollout_batch_size="${MINI_ROLLOUT_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-128}" \
  data.min_pixels="${MIN_PIXELS:-200704}" \
  data.max_pixels="${MAX_PIXELS:-1048576}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-true}" \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.trust_remote_code="${TRUST_REMOTE_CODE:-true}" \
  worker.actor.model.freeze_vision_tower="${FREEZE_VISION_TOWER:-true}" \
  worker.actor.model.lora.rank="${LORA_RANK:-16}" \
  worker.actor.model.lora.alpha="${LORA_ALPHA:-32}" \
  worker.actor.model.lora.target_modules="${LORA_TARGET_MODULES:-all-linear}" \
  worker.actor.model.lora.exclude_modules="${LORA_EXCLUDE_MODULES:-.*visual.*}" \
  worker.actor.fsdp.torch_dtype="${TORCH_DTYPE:-bf16}" \
  worker.actor.optim.strategy="${OPTIM_STRATEGY:-adamw_bf16}" \
  worker.actor.optim.lr="${LEARNING_RATE}" \
  worker.actor.optim.weight_decay="${WEIGHT_DECAY:-1e-2}" \
  worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE_UPDATE}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE_EXPERIENCE}" \
  worker.actor.loss_type="${ACTOR_LOSS_TYPE:-gspo_token}" \
  worker.actor.loss_avg_mode="${LOSS_AVG_MODE:-seq}" \
  worker.actor.clip_ratio_low="${CLIP_RATIO_LOW:-3e-4}" \
  worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-4e-4}" \
  worker.actor.offload.offload_params="${ACTOR_OFFLOAD_PARAMS:-false}" \
  worker.actor.offload.offload_optimizer="${ACTOR_OFFLOAD_OPTIMIZER:-false}" \
  worker.ref.fsdp.enable_cpu_offload="${REF_CPU_OFFLOAD:-false}" \
  worker.rollout.n="${ROLLOUT_N}" \
  worker.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.8}" \
  worker.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
  worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.75}" \
  worker.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  worker.rollout.limit_images="${ROLLOUT_LIMIT_IMAGES:-0}" \
  worker.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-true}" \
  worker.rollout.extra_sampling_params="${ROLLOUT_EXTRA_SAMPLING_PARAMS}" \
  worker.reward.reward_function=./examples/reward_function/visualprm_cached_think_step_source_macro.py:compute_score \
  worker.reward.reward_function_kwargs="{\"allow_fallback_parse\":${ALLOW_FALLBACK_PARSE:-false},\"source_macro_weight\":${SOURCE_MACRO_WEIGHT},\"step_weight\":${STEP_WEIGHT},\"format_weight\":${FORMAT_WEIGHT},\"invalid_reward\":${INVALID_REWARD:-0.0}}" \
  algorithm.adv_estimator=grpo \
  algorithm.disable_kl="${DISABLE_KL:-false}" \
  algorithm.use_kl_loss="${USE_KL_LOSS:-true}" \
  algorithm.kl_penalty="${KL_PENALTY:-low_var_kl}" \
  algorithm.kl_coef="${KL_COEF}" \
  algorithm.online_filtering="${ONLINE_FILTERING:-false}" \
  algorithm.filter_key="${FILTER_KEY:-overall}" \
  algorithm.filter_low="${FILTER_LOW:-0.0}" \
  algorithm.filter_high="${FILTER_HIGH:-1.0}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.project_name="${PROJECT_NAME:-visualprm_easy_r1}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${LOGGER:-[\"file\",\"wandb\"]}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-2}" \
  trainer.val_freq="${VAL_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.val_generations_to_log="${VAL_GENERATIONS_TO_LOG}" \
  trainer.train_rollouts_to_log="${TRAIN_ROLLOUTS_TO_LOG}" \
  trainer.train_rollout_log_freq="${TRAIN_ROLLOUT_LOG_FREQ}" \
  trainer.generation_log_max_chars="${GENERATION_LOG_MAX_CHARS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.save_limit="${SAVE_LIMIT:-5}" \
  trainer.save_model_only="${SAVE_MODEL_ONLY:-false}" \
  trainer.find_last_checkpoint="${FIND_LAST_CHECKPOINT:-true}" \
  trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH}"
