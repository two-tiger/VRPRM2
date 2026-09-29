#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABEL="${MODEL_LABEL:-InternVL2.5-8B}"
REWARD_LABEL="${REWARD_LABEL:-prm}"

DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
BON_LIST="${BON_LIST:-1,2,4,8,16,32,64,128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_API_NPROC="${EVAL_API_NPROC:-4}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"
PROGRESS_WIDTH="${PROGRESS_WIDTH:-30}"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a BON_ARRAY <<< "${BON_LIST}"

TOTAL_TASKS=0
for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue
  for BON in "${BON_ARRAY[@]}"; do
    BON="$(echo "${BON}" | xargs)"
    [[ -z "${BON}" ]] && continue
    TOTAL_TASKS=$((TOTAL_TASKS + 1))
  done
done

COMPLETED_TASKS=0
print_progress() {
  if [[ "${NO_PROGRESS}" == "1" || "${TOTAL_TASKS}" -le 0 ]]; then
    return
  fi

  local label="${1:-}"
  local percent=$((COMPLETED_TASKS * 100 / TOTAL_TASKS))
  local filled=$((COMPLETED_TASKS * PROGRESS_WIDTH / TOTAL_TASKS))
  local empty=$((PROGRESS_WIDTH - filled))
  local bar=""
  local spaces=""

  printf -v bar "%${filled}s" ""
  printf -v spaces "%${empty}s" ""
  bar="${bar// /#}"

  printf 'Progress [%s%s] %d/%d (%d%%) %s\n' \
    "${bar}" "${spaces}" "${COMPLETED_TASKS}" "${TOTAL_TASKS}" "${percent}" "${label}"
}

print_progress "start"

for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue

  OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
  SCORES="${OUT_DIR}/rollouts_n${ROLLOUT_N}_${REWARD_LABEL}_scores.jsonl"

  if [[ ! -f "${SCORES}" ]]; then
    echo "ERROR: missing score cache: ${SCORES}" >&2
    echo "Run run_score_rollouts_cache.sh first." >&2
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

    echo "==> ${DATASET}: select Bo${BON} from cached ${ROLLOUT_N} scores"
    python "${SCRIPT_DIR}/select_bon_from_scores.py" \
      --dataset "${DATASET}" \
      --score-output "${SCORES}" \
      --output-xlsx "${SELECTED}" \
      --summary-output "${SUMMARY}" \
      --bon "${BON}" \
      "${SELECT_ARGS[@]}"

    if [[ "${RUN_EVAL}" == "1" ]]; then
      echo "==> ${DATASET}: evaluate Bo${BON}"
      python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
        --dataset "${DATASET}" \
        --prediction "${SELECTED}" \
        --output "${EVAL_OUT}" \
        --api-nproc "${EVAL_API_NPROC}"
    fi

    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    print_progress "${DATASET} Bo${BON} done"
  done
done
