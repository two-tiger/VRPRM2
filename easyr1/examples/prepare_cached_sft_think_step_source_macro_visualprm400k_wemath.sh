#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VRPRM_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
REPO_ROOT="$(cd "${VRPRM_ROOT}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${VRPRM_ROOT}/data/VisualPRM400K-v1.1-Raw}"
IMAGE_DIR="${IMAGE_DIR:-${VRPRM_ROOT}/data}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_filtered_source_cases_clean_pos0875_pool150k_balanced}"
DATASET_DIR="${DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_cached_sft_think_step_source_macro_wemath_60k}"

export NEGATIVE_THRESHOLD="${NEGATIVE_THRESHOLD:-0.125}"
export POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.875}"
export TARGET_NEGATIVE_SOURCE_RATIO="${TARGET_NEGATIVE_SOURCE_RATIO:-0.45}"
export TARGET_NEGATIVE_STEP_RATIO="${TARGET_NEGATIVE_STEP_RATIO:-0.28}"
export MAX_TRAIN_SOURCE_SAMPLES="${MAX_TRAIN_SOURCE_SAMPLES:-150000}"
export MAX_VAL_SOURCE_SAMPLES="${MAX_VAL_SOURCE_SAMPLES:-2000}"
export MAX_TRAIN_SOURCE_ROLLOUT="${MAX_TRAIN_SOURCE_ROLLOUT:-20000}"
export MAX_VAL_SOURCE_ROLLOUT="${MAX_VAL_SOURCE_ROLLOUT:-2000}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-60000}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-3000}"
export DATA_SEED="${DATA_SEED:-42}"

export THINKING_SAMPLES="${THINKING_SAMPLES:-3}"
export MAX_THINKING_VARIANTS_PER_SOURCE="${MAX_THINKING_VARIANTS_PER_SOURCE:-2}"
export THINKING_DROPOUT_RATIO="${THINKING_DROPOUT_RATIO:-0.25}"
export NO_THINKING_RATIO="${NO_THINKING_RATIO:-0.10}"
export THINKING_SENTENCE_KEEP_RATIO="${THINKING_SENTENCE_KEEP_RATIO:-0.65}"
export MIN_THINK_CHARS="${MIN_THINK_CHARS:-80}"
export MAX_THINK_CHARS="${MAX_THINK_CHARS:-1600}"
export SOURCE_BONUS_REGEX="${SOURCE_BONUS_REGEX:-(math|geo|geometry|mavis|unigeo|figureqa|chartqa|ai2d|MathV360K)}"
export SOURCE_BONUS="${SOURCE_BONUS:-0.05}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "VisualPRM400K raw data directory not found: ${DATA_ROOT}" >&2
  exit 1
fi

cd "${EASYR1_ROOT}"

if [[ ! -f "${SOURCE_DATASET_DIR}/train_source.jsonl" || ! -f "${SOURCE_DATASET_DIR}/test_source.jsonl" || ! -f "${SOURCE_DATASET_DIR}/.visualprm_filtered_source_cases_v1" || "${REBUILD_SOURCE_DATASET:-false}" == "true" ]]; then
  python3 scripts/prepare_visualprm400k_filtered_source_cases.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${SOURCE_DATASET_DIR}" \
    --negative-threshold "${NEGATIVE_THRESHOLD}" \
    --positive-threshold "${POSITIVE_THRESHOLD}" \
    --target-negative-source-ratio "${TARGET_NEGATIVE_SOURCE_RATIO}" \
    --target-negative-step-ratio "${TARGET_NEGATIVE_STEP_RATIO}" \
    --max-train-source-samples "${MAX_TRAIN_SOURCE_SAMPLES}" \
    --max-val-source-samples "${MAX_VAL_SOURCE_SAMPLES}" \
    --max-all-positive-per-batch-ratio "${MAX_ALL_POSITIVE_PER_BATCH_RATIO:-0.50}" \
    --batch-size-for-ordering "${BATCH_SIZE_FOR_ORDERING:-32}" \
    --val-ratio "${VAL_RATIO:-0.01}" \
    --seed "${DATA_SEED}" \
    --limit "${DATA_LIMIT:-0}"
fi

ROLLOUT_ARGS=()
if [[ "${SKIP_THINK_ROLLOUT:-false}" == "true" || "${SKIP_THINK_ROLLOUT:-false}" == "1" ]]; then
  ROLLOUT_ARGS+=(--skip-rollout)
fi
if [[ "${OVERWRITE_THINK_CACHE:-false}" == "true" || "${OVERWRITE_THINK_CACHE:-false}" == "1" ]]; then
  ROLLOUT_ARGS+=(--overwrite-cache)
fi

if [[ ! -f "${DATASET_DIR}/train.jsonl" || ! -f "${DATASET_DIR}/test.jsonl" || ! -f "${DATASET_DIR}/.visualprm_cached_think_step_paths_v1" || "${REBUILD_DATASET:-false}" == "true" ]]; then
  python3 scripts/rollout_sft_global_thinking_to_step_rl.py \
    --input-dir "${SOURCE_DATASET_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --base-url "${SFT_ROLLOUT_BASE_URL:-http://localhost:8000/v1}" \
    --base-urls "${SFT_ROLLOUT_BASE_URLS:-}" \
    --api-key "${SFT_ROLLOUT_API_KEY:-EMPTY}" \
    --model "${SFT_ROLLOUT_MODEL:-auto}" \
    --temperature "${SFT_ROLLOUT_TEMPERATURE:-0.7}" \
    --top-p "${SFT_ROLLOUT_TOP_P:-0.95}" \
    --max-tokens "${SFT_ROLLOUT_MAX_TOKENS:-1024}" \
    --min-think-chars "${MIN_THINK_CHARS}" \
    --max-think-chars "${MAX_THINK_CHARS}" \
    --thinking-samples "${THINKING_SAMPLES}" \
    --max-thinking-variants-per-source "${MAX_THINKING_VARIANTS_PER_SOURCE}" \
    --thinking-dropout-ratio "${THINKING_DROPOUT_RATIO}" \
    --no-thinking-ratio "${NO_THINKING_RATIO}" \
    --thinking-sentence-keep-ratio "${THINKING_SENTENCE_KEEP_RATIO}" \
    --require-think-block \
    --allow-missing-think-open \
    --reject-step-scores-in-thinking \
    --reject-all-correct-mismatch \
    --reject-error-mismatch \
    --reject-first-error-mismatch \
    --first-error-tolerance "${FIRST_ERROR_TOLERANCE:-1}" \
    --concurrency "${SFT_ROLLOUT_CONCURRENCY:-256}" \
    --per-endpoint-concurrency "${SFT_ROLLOUT_PER_ENDPOINT_CONCURRENCY:-64}" \
    --max-retries "${SFT_ROLLOUT_MAX_RETRIES:-3}" \
    --request-timeout "${SFT_ROLLOUT_REQUEST_TIMEOUT:-300}" \
    --cache-flush-every "${CACHE_FLUSH_EVERY:-20}" \
    --cache-fsync-every "${CACHE_FSYNC_EVERY:-0}" \
    --max-train-source-rollout "${MAX_TRAIN_SOURCE_ROLLOUT}" \
    --max-val-source-rollout "${MAX_VAL_SOURCE_ROLLOUT}" \
    --max-train-step-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-step-samples "${MAX_VAL_SAMPLES}" \
    --target-negative-source-ratio "${TARGET_NEGATIVE_SOURCE_RATIO}" \
    --target-negative-step-ratio "${TARGET_NEGATIVE_STEP_RATIO}" \
    --source-bonus-regex "${SOURCE_BONUS_REGEX}" \
    --source-bonus "${SOURCE_BONUS}" \
    --seed "${DATA_SEED}" \
    "${ROLLOUT_ARGS[@]}"
else
  echo "Cached-thinking RL dataset already exists: ${DATASET_DIR}"
fi

echo "Data preparation complete."
echo "Source dataset: ${SOURCE_DATASET_DIR}"
echo "RL dataset: ${DATASET_DIR}"
