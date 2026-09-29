#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABELS="${MODEL_LABELS:-${MODEL_LABEL:-InternVL2.5-8B}}"
DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
BON_LIST="${BON_LIST:-2,4,8,16,32,64,128}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs_visualprm_bon}"
REWARD_LABEL="${REWARD_LABEL:-visualprm_soft}"

EVAL_API_NPROC="${EVAL_API_NPROC:-4}"
EVAL_RETRY="${EVAL_RETRY:-3}"
EVAL_JUDGE="${EVAL_JUDGE:-}"
EVAL_JUDGE_ARGS="${EVAL_JUDGE_ARGS:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"

IFS=',' read -r -a MODEL_LABEL_ARRAY <<< "${MODEL_LABELS}"
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a BON_ARRAY <<< "${BON_LIST}"

for MODEL_LABEL in "${MODEL_LABEL_ARRAY[@]}"; do
  MODEL_LABEL="$(echo "${MODEL_LABEL}" | xargs)"
  [[ -z "${MODEL_LABEL}" ]] && continue

  for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="$(echo "${DATASET}" | xargs)"
    [[ -z "${DATASET}" ]] && continue

    OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"

    for BON in "${BON_ARRAY[@]}"; do
      BON="$(echo "${BON}" | xargs)"
      [[ -z "${BON}" ]] && continue
      if (( BON > ROLLOUT_N )); then
        echo "ERROR: BON=${BON} is larger than ROLLOUT_N=${ROLLOUT_N}" >&2
        exit 1
      fi

      SELECTED="${OUT_DIR}/bon${BON}_${REWARD_LABEL}_from_n${ROLLOUT_N}_selected.xlsx"
      EVAL_OUT="${OUT_DIR}/bon${BON}_${REWARD_LABEL}_from_n${ROLLOUT_N}_eval.json"

      if [[ ! -f "${SELECTED}" ]]; then
        echo "ERROR: missing selected prediction file: ${SELECTED}" >&2
        exit 1
      fi

      EVAL_ARGS=()
      if [[ -n "${EVAL_JUDGE}" ]]; then
        EVAL_ARGS+=(--judge "${EVAL_JUDGE}")
      fi
      if [[ -n "${EVAL_JUDGE_ARGS}" ]]; then
        EVAL_ARGS+=(--judge-args "${EVAL_JUDGE_ARGS}")
      fi
      if [[ "${NO_PROGRESS}" != "1" ]]; then
        echo "==> ${MODEL_LABEL}/${DATASET}: evaluate VisualPRM Bo${BON}"
      fi

      python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
        --dataset "${DATASET}" \
        --prediction "${SELECTED}" \
        --output "${EVAL_OUT}" \
        --api-nproc "${EVAL_API_NPROC}" \
        --retry "${EVAL_RETRY}" \
        "${EVAL_ARGS[@]}"
    done
  done
done

echo "VisualPRM BoN evaluation outputs written under: ${OUTPUT_ROOT}"
