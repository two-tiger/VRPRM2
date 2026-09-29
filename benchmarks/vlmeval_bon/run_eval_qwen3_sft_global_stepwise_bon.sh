#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABELS="${MODEL_LABELS:-${MODEL_LABEL:-InternVL2.5-8B}}"
DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
BON_LIST="${BON_LIST:-2,4,8,16,32,64,128}"
SCORE_BON="${SCORE_BON:-}"

INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs_qwen3_vl_8b_thinking_global_stepwise_bon}"
REWARD_LABEL="${REWARD_LABEL:-qwen3_sft_global_stepwise_logprob_final_answer}"

PRM_BASE_URL="${PRM_BASE_URL:-http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1}"
PRM_API_KEY="${PRM_API_KEY:-EMPTY}"
PRM_MODEL="${PRM_MODEL:-auto}"
PRM_CONCURRENCY="${PRM_CONCURRENCY:-32}"
PRM_THINK_MAX_TOKENS="${PRM_THINK_MAX_TOKENS:-1024}"
PRM_TEMPERATURE="${PRM_TEMPERATURE:-0.0}"
PRM_GUIDED_CHOICE="${PRM_GUIDED_CHOICE:-1}"
PRM_USE_LOGPROB_SCORE="${PRM_USE_LOGPROB_SCORE:-1}"
PRM_TOP_LOGPROBS="${PRM_TOP_LOGPROBS:-5}"
PRM_REQUIRE_ANSWER_BOX="${PRM_REQUIRE_ANSWER_BOX:-0}"
PRM_FINAL_ANSWER_WEIGHT="${PRM_FINAL_ANSWER_WEIGHT:-0.0}"
PRM_FINAL_ANSWER_MISMATCH_PENALTY="${PRM_FINAL_ANSWER_MISMATCH_PENALTY:-0.0}"
PRM_MAX_RETRIES="${PRM_MAX_RETRIES:-3}"
PRM_API_ERROR_RETRIES="${PRM_API_ERROR_RETRIES:-3}"
PRM_REQUEST_TIMEOUT="${PRM_REQUEST_TIMEOUT:-300}"

RUN_SCORE="${RUN_SCORE:-1}"
RUN_SELECT="${RUN_SELECT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_API_NPROC="${EVAL_API_NPROC:-4}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
OVERWRITE="${OVERWRITE:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

IFS=',' read -r -a MODEL_LABEL_ARRAY <<< "${MODEL_LABELS}"
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a BON_ARRAY <<< "${BON_LIST}"

MAX_BON=0
for BON in "${BON_ARRAY[@]}"; do
  BON="$(echo "${BON}" | xargs)"
  [[ -z "${BON}" ]] && continue
  if (( BON > MAX_BON )); then
    MAX_BON="${BON}"
  fi
done

if (( MAX_BON < 1 )); then
  echo "ERROR: BON_LIST must contain at least one positive integer." >&2
  exit 1
fi

if [[ -z "${SCORE_BON}" ]]; then
  SCORE_BON="${MAX_BON}"
fi

if (( SCORE_BON < MAX_BON )); then
  echo "ERROR: SCORE_BON=${SCORE_BON} is smaller than max BON_LIST=${MAX_BON}" >&2
  exit 1
fi
if (( SCORE_BON > ROLLOUT_N )); then
  echo "ERROR: SCORE_BON=${SCORE_BON} is larger than ROLLOUT_N=${ROLLOUT_N}" >&2
  exit 1
fi

for MODEL_LABEL in "${MODEL_LABEL_ARRAY[@]}"; do
  MODEL_LABEL="$(echo "${MODEL_LABEL}" | xargs)"
  [[ -z "${MODEL_LABEL}" ]] && continue

  for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="$(echo "${DATASET}" | xargs)"
    [[ -z "${DATASET}" ]] && continue

    IN_DIR="${INPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    ROLLOUTS="${IN_DIR}/rollouts_n${ROLLOUT_N}.jsonl"
    SCORES="${OUT_DIR}/rollouts_n${ROLLOUT_N}_${REWARD_LABEL}_scores.jsonl"

    if [[ ! -f "${ROLLOUTS}" ]]; then
      echo "WARNING: skip missing rollouts file: ${ROLLOUTS}" >&2
      continue
    fi

    mkdir -p "${OUT_DIR}"

    SCORE_ARGS=()
    if [[ "${OVERWRITE}" == "1" ]]; then
      SCORE_ARGS+=(--overwrite)
    fi
    if [[ "${NO_PROGRESS}" == "1" ]]; then
      SCORE_ARGS+=(--no-progress)
    fi

    if [[ "${RUN_SCORE}" == "1" ]]; then
      echo "==> ${MODEL_LABEL}/${DATASET}: VPB-style global-think stepwise score first ${SCORE_BON}/${ROLLOUT_N} candidates"
      python "${SCRIPT_DIR}/score_rollouts_global_think_stepwise.py" \
        --dataset "${DATASET}" \
        --rollouts "${ROLLOUTS}" \
        --score-output "${SCORES}" \
        --prm-base-url "${PRM_BASE_URL}" \
        --prm-api-key "${PRM_API_KEY}" \
        --prm-model "${PRM_MODEL}" \
        --bon "${SCORE_BON}" \
        --concurrency "${PRM_CONCURRENCY}" \
        --think-max-tokens "${PRM_THINK_MAX_TOKENS}" \
        --temperature "${PRM_TEMPERATURE}" \
        --guided-choice "${PRM_GUIDED_CHOICE}" \
        --use-logprob-score "${PRM_USE_LOGPROB_SCORE}" \
        --top-logprobs "${PRM_TOP_LOGPROBS}" \
        --require-answer-box "${PRM_REQUIRE_ANSWER_BOX}" \
        --final-answer-weight "${PRM_FINAL_ANSWER_WEIGHT}" \
        --final-answer-mismatch-penalty "${PRM_FINAL_ANSWER_MISMATCH_PENALTY}" \
        --max-retries "${PRM_MAX_RETRIES}" \
        --api-error-retries "${PRM_API_ERROR_RETRIES}" \
        --request-timeout "${PRM_REQUEST_TIMEOUT}" \
        "${SCORE_ARGS[@]}"
    fi

    if [[ "${RUN_SELECT}" != "1" && "${RUN_EVAL}" != "1" ]]; then
      continue
    fi
    if [[ ! -f "${SCORES}" ]]; then
      echo "ERROR: missing score cache: ${SCORES}" >&2
      exit 1
    fi

    for BON in "${BON_ARRAY[@]}"; do
      BON="$(echo "${BON}" | xargs)"
      [[ -z "${BON}" ]] && continue
      if (( BON > ROLLOUT_N )); then
        echo "ERROR: BON=${BON} is larger than ROLLOUT_N=${ROLLOUT_N}" >&2
        exit 1
      fi

      SELECTED="${OUT_DIR}/bon${BON}_${REWARD_LABEL}_from_n${ROLLOUT_N}_selected.xlsx"
      SUMMARY="${OUT_DIR}/bon${BON}_${REWARD_LABEL}_from_n${ROLLOUT_N}_selection.json"
      EVAL_OUT="${OUT_DIR}/bon${BON}_${REWARD_LABEL}_from_n${ROLLOUT_N}_eval.json"

      SELECT_ARGS=()
      if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
        SELECT_ARGS+=(--allow-partial)
      fi

      if [[ "${RUN_SELECT}" == "1" ]]; then
        echo "==> ${MODEL_LABEL}/${DATASET}: select Bo${BON} best rollout"
        python "${SCRIPT_DIR}/select_bon_from_scores.py" \
          --dataset "${DATASET}" \
          --score-output "${SCORES}" \
          --output-xlsx "${SELECTED}" \
          --summary-output "${SUMMARY}" \
          --bon "${BON}" \
          "${SELECT_ARGS[@]}"
      fi

      if [[ "${RUN_EVAL}" == "1" ]]; then
        if [[ ! -f "${SELECTED}" ]]; then
          echo "ERROR: missing selected prediction file: ${SELECTED}" >&2
          exit 1
        fi
        echo "==> ${MODEL_LABEL}/${DATASET}: evaluate Bo${BON}"
        python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
          --dataset "${DATASET}" \
          --prediction "${SELECTED}" \
          --output "${EVAL_OUT}" \
          --api-nproc "${EVAL_API_NPROC}"
      fi
    done
  done
done

echo "Outputs written under: ${OUTPUT_ROOT}"
