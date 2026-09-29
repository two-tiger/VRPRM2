#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABEL="${MODEL_LABEL:-InternVL2.5-8B}"
PRM_BASE_URL="${PRM_BASE_URL:-http://127.0.0.1:8001/v1}"
PRM_API_KEY="${PRM_API_KEY:-EMPTY}"
PRM_MODEL="${PRM_MODEL:-auto}"
REWARD_LABEL="${REWARD_LABEL:-prm}"

DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

PRM_CONCURRENCY="${PRM_CONCURRENCY:-16}"
PRM_MAX_TOKENS="${PRM_MAX_TOKENS:-2048}"
PRM_WARMUP_MAX_TOKENS="${PRM_WARMUP_MAX_TOKENS:-1024}"
PRM_TEMPERATURE="${PRM_TEMPERATURE:-0.0}"
PRM_REWARD_MODE="${PRM_REWARD_MODE:-stepwise}"
PRM_GUIDED_CHOICE="${PRM_GUIDED_CHOICE:-1}"
PRM_MAX_RETRIES="${PRM_MAX_RETRIES:-3}"
PRM_REQUEST_TIMEOUT="${PRM_REQUEST_TIMEOUT:-300}"
OVERWRITE="${OVERWRITE:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue

  OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
  ROLLOUTS="${OUT_DIR}/rollouts_n${ROLLOUT_N}.jsonl"
  SCORES="${OUT_DIR}/rollouts_n${ROLLOUT_N}_${REWARD_LABEL}_scores.jsonl"

  if [[ ! -f "${ROLLOUTS}" ]]; then
    echo "ERROR: missing rollouts file: ${ROLLOUTS}" >&2
    echo "Expected an existing rollouts_n${ROLLOUT_N}.jsonl cache under outputs/<MODEL_LABEL>/<DATASET>." >&2
    exit 1
  fi

  ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    ARGS+=(--no-progress)
  fi

  echo "==> ${DATASET}: score first ${ROLLOUT_N} rollouts -> ${SCORES}"
  python "${SCRIPT_DIR}/score_rollouts_with_prm.py" \
    --dataset "${DATASET}" \
    --rollouts "${ROLLOUTS}" \
    --score-output "${SCORES}" \
    --prm-base-url "${PRM_BASE_URL}" \
    --prm-api-key "${PRM_API_KEY}" \
    --prm-model "${PRM_MODEL}" \
    --bon "${ROLLOUT_N}" \
    --concurrency "${PRM_CONCURRENCY}" \
    --max-tokens "${PRM_MAX_TOKENS}" \
    --warmup-max-tokens "${PRM_WARMUP_MAX_TOKENS}" \
    --temperature "${PRM_TEMPERATURE}" \
    --reward-mode "${PRM_REWARD_MODE}" \
    --guided-choice "${PRM_GUIDED_CHOICE}" \
    --max-retries "${PRM_MAX_RETRIES}" \
    --request-timeout "${PRM_REQUEST_TIMEOUT}" \
    "${ARGS[@]}"
done
