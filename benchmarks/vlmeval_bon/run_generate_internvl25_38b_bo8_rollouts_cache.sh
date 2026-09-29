#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABEL="${MODEL_LABEL:-InternVL2.5-38B}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-internvl25-38b}"

DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-8}"
SEED_ROLLOUT_N="${SEED_ROLLOUT_N:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

POLICY_CONCURRENCY="${POLICY_CONCURRENCY:-32}"
POLICY_ROLLOUT_CONCURRENCY="${POLICY_ROLLOUT_CONCURRENCY:-1}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-2048}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.7}"
POLICY_TOP_P="${POLICY_TOP_P:-0.95}"
POLICY_MAX_RETRIES="${POLICY_MAX_RETRIES:-3}"
POLICY_REQUEST_TIMEOUT="${POLICY_REQUEST_TIMEOUT:-600}"
OVERWRITE="${OVERWRITE:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue

  OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
  mkdir -p "${OUT_DIR}"
  ROLLOUTS="${OUT_DIR}/rollouts_n${ROLLOUT_N}.jsonl"
  SEED_ROLLOUTS="${OUT_DIR}/rollouts_n${SEED_ROLLOUT_N}.jsonl"

  ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    ARGS+=(--no-progress)
  fi
  if [[ -f "${SEED_ROLLOUTS}" || -f "${SEED_ROLLOUTS%.jsonl}.candidates.jsonl" ]]; then
    ARGS+=(--seed-output "${SEED_ROLLOUTS}")
  fi

  echo "==> ${MODEL_LABEL}/${DATASET}: generate/resume Bo${ROLLOUT_N} rollouts"
  if [[ -f "${SEED_ROLLOUTS}" || -f "${SEED_ROLLOUTS%.jsonl}.candidates.jsonl" ]]; then
    echo "    seed: ${SEED_ROLLOUTS}"
  fi
  python "${SCRIPT_DIR}/generate_rollouts.py" \
    --dataset "${DATASET}" \
    --output "${ROLLOUTS}" \
    --policy-base-url "${POLICY_BASE_URL}" \
    --policy-api-key "${POLICY_API_KEY}" \
    --policy-model "${POLICY_MODEL}" \
    --bon "${ROLLOUT_N}" \
    --concurrency "${POLICY_CONCURRENCY}" \
    --rollout-concurrency "${POLICY_ROLLOUT_CONCURRENCY}" \
    --max-tokens "${POLICY_MAX_TOKENS}" \
    --temperature "${POLICY_TEMPERATURE}" \
    --top-p "${POLICY_TOP_P}" \
    --max-retries "${POLICY_MAX_RETRIES}" \
    --request-timeout "${POLICY_REQUEST_TIMEOUT}" \
    "${ARGS[@]}"
done
