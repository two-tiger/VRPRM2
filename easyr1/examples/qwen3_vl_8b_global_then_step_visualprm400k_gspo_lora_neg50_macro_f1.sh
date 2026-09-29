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

DATA_ROOT="${DATA_ROOT:-${VRPRM_ROOT}/data/VisualPRM400K-v1.1-Raw}"
DATASET_DIR="${DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_global_then_step_rl_neg50_50k}"
IMAGE_DIR="${IMAGE_DIR:-${VRPRM_ROOT}/data}"

if [[ ! -d "${MODEL_PATH}" && "${MODEL_PATH}" != Qwen/* ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to a merged SFT Hugging Face directory or a Hub model id." >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "VisualPRM400K raw data directory not found: ${DATA_ROOT}" >&2
  echo "Download VisualPRM400K-v1.1-Raw from Hugging Face into VRPRM_v2.0/data/ or set DATA_ROOT." >&2
  exit 1
fi

cd "${EASYR1_ROOT}"

if [[ -d "${MODEL_PATH}" && -f "${MODEL_PATH}/config.json" && -f "${BASE_MODEL_PATH}/config.json" ]]; then
  python3 scripts/fix_qwen3vl_config_rope.py \
    "${MODEL_PATH}" \
    --reference-config "${BASE_MODEL_PATH}/config.json"
fi

if [[ ! -f "${DATASET_DIR}/train.jsonl" || ! -f "${DATASET_DIR}/test.jsonl" || ! -f "${DATASET_DIR}/.visualprm_global_stepwise_paths_v1" || "${REBUILD_DATASET:-false}" == "true" ]]; then
  python3 scripts/prepare_visualprm400k_global_stepwise_rl.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${DATASET_DIR}" \
    --negative-threshold "${NEGATIVE_THRESHOLD:-0.125}" \
    --positive-threshold "${POSITIVE_THRESHOLD:-0.75}" \
    --min-confident-steps "${MIN_CONFIDENT_STEPS:-1}" \
    --val-ratio "${VAL_RATIO:-0.01}" \
    --max-val-samples "${MAX_VAL_SAMPLES:-2000}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES:-50000}" \
    --target-negative-step-ratio "${TARGET_NEGATIVE_STEP_RATIO:-0.50}" \
    --max-all-positive-per-batch-ratio "${MAX_ALL_POSITIVE_PER_BATCH_RATIO:-0.25}" \
    --batch-size-for-ordering "${BATCH_SIZE_FOR_ORDERING:-${MINI_ROLLOUT_BATCH_SIZE:-16}}" \
    --seed "${DATA_SEED:-42}" \
    --limit "${DATA_LIMIT:-0}"
fi

export PYTHONPATH="${EASYR1_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_gts_mf1}"
if (( ${#RAY_TMPDIR} > 50 )); then
  echo "RAY_TMPDIR is too long for Ray socket paths, overriding: ${RAY_TMPDIR}" >&2
  RAY_TMPDIR="/tmp/ray_gts"
fi
export RAY_TMPDIR
mkdir -p "${RAY_TMPDIR}"

CACHE_ROOT="${CACHE_ROOT:-${VRPRM_ROOT}/cache/vrprm_rl}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_WORKSPACE_BASE}/cubins}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}" "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_CUBIN_DIR}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${VRPRM_ROOT}/wandb}"

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${DATASET_DIR}/train.jsonl" \
  data.val_files="${DATASET_DIR}/test.jsonl" \
  data.prompt_key=prompt \
  data.answer_key=ground_truth \
  data.image_key=images \
  data.shuffle="${DATA_SHUFFLE:-false}" \
  data.image_dir="${IMAGE_DIR}" \
  data.format_prompt=./examples/format_prompt/visualprm_global_then_step.jinja \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1536}" \
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE:-32}" \
  data.mini_rollout_batch_size="${MINI_ROLLOUT_BATCH_SIZE:-16}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-8}" \
  data.min_pixels="${MIN_PIXELS:-200704}" \
  data.max_pixels="${MAX_PIXELS:-1048576}" \
  data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS:-true}" \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.trust_remote_code="${TRUST_REMOTE_CODE:-true}" \
  worker.actor.model.freeze_vision_tower="${FREEZE_VISION_TOWER:-true}" \
  worker.actor.model.lora.rank="${LORA_RANK:-64}" \
  worker.actor.model.lora.alpha="${LORA_ALPHA:-64}" \
  worker.actor.model.lora.target_modules="${LORA_TARGET_MODULES:-all-linear}" \
  worker.actor.model.lora.exclude_modules="${LORA_EXCLUDE_MODULES:-.*visual.*}" \
  worker.actor.fsdp.torch_dtype="${TORCH_DTYPE:-bf16}" \
  worker.actor.optim.strategy="${OPTIM_STRATEGY:-adamw_bf16}" \
  worker.actor.optim.lr="${LEARNING_RATE:-5e-8}" \
  worker.actor.optim.weight_decay="${WEIGHT_DECAY:-1e-2}" \
  worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE:-16}" \
  worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE_UPDATE:-2}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE_EXPERIENCE:-2}" \
  worker.actor.loss_type="${ACTOR_LOSS_TYPE:-gspo_token}" \
  worker.actor.loss_avg_mode="${LOSS_AVG_MODE:-seq}" \
  worker.actor.clip_ratio_low="${CLIP_RATIO_LOW:-3e-4}" \
  worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-4e-4}" \
  worker.actor.offload.offload_params="${ACTOR_OFFLOAD_PARAMS:-false}" \
  worker.actor.offload.offload_optimizer="${ACTOR_OFFLOAD_OPTIMIZER:-false}" \
  worker.ref.fsdp.enable_cpu_offload="${REF_CPU_OFFLOAD:-false}" \
  worker.rollout.n="${ROLLOUT_N:-8}" \
  worker.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}" \
  worker.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
  worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.75}" \
  worker.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}" \
  worker.rollout.limit_images="${ROLLOUT_LIMIT_IMAGES:-0}" \
  worker.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-true}" \
  worker.reward.reward_function=./examples/reward_function/visualprm_global_then_step_macro_f1.py:compute_score \
  worker.reward.reward_function_kwargs="{\"step_weight\":${STEP_WEIGHT:-0.95},\"format_weight\":${FORMAT_WEIGHT:-0.03},\"think_weight\":${THINK_WEIGHT:-0.02},\"count_weight\":${COUNT_WEIGHT:-0.05},\"min_think_chars\":${MIN_THINK_CHARS:-80},\"max_think_chars\":${MAX_THINK_CHARS:-1800}}" \
  algorithm.adv_estimator=grpo \
  algorithm.disable_kl="${DISABLE_KL:-false}" \
  algorithm.use_kl_loss="${USE_KL_LOSS:-true}" \
  algorithm.kl_penalty="${KL_PENALTY:-low_var_kl}" \
  algorithm.kl_coef="${KL_COEF:-5.0e-2}" \
  algorithm.online_filtering="${ONLINE_FILTERING:-false}" \
  algorithm.filter_key="${FILTER_KEY:-overall}" \
  algorithm.filter_low="${FILTER_LOW:-0.0}" \
  algorithm.filter_high="${FILTER_HIGH:-1.0}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.max_steps="${MAX_STEPS:-400}" \
  trainer.project_name="${PROJECT_NAME:-visualprm_easy_r1}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_vl_8b_global_then_step_visualprm400k_gspo_lora_neg50_macro_f1}" \
  trainer.logger="${LOGGER:-[\"file\",\"wandb\"]}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-2}" \
  trainer.val_freq="${VAL_FREQ:-100}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-true}" \
  trainer.val_generations_to_log="${VAL_GENERATIONS_TO_LOG:-0}" \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.save_limit="${SAVE_LIMIT:-5}" \
  trainer.save_model_only="${SAVE_MODEL_ONLY:-false}" \
  trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${EASYR1_ROOT}/checkpoints/visualprm_easy_r1/qwen3_vl_8b_global_then_step_visualprm400k_gspo_lora_neg_50_50k_macro_f1}"
