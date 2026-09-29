#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABEL="${MODEL_LABEL:-InternVL2.5-8B}"
DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,DynaMath,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

CANDIDATE_IDX="${CANDIDATE_IDX:-0}"
PREDICTION_MODE="${PREDICTION_MODE:-raw}"
FALLBACK_FIRST_VALID="${FALLBACK_FIRST_VALID:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_API_NPROC="${EVAL_API_NPROC:-4}"
NO_PROGRESS="${NO_PROGRESS:-0}"
PROGRESS_WIDTH="${PROGRESS_WIDTH:-30}"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

TOTAL_TASKS=0
for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue
  TOTAL_TASKS=$((TOTAL_TASKS + 1))
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
  ROLLOUTS="${OUT_DIR}/rollouts_n${ROLLOUT_N}.jsonl"

  if [[ ! -f "${ROLLOUTS}" ]]; then
    echo "ERROR: missing rollouts file: ${ROLLOUTS}" >&2
    echo "Expected an existing rollouts_n${ROLLOUT_N}.jsonl cache under outputs/<MODEL_LABEL>/<DATASET>." >&2
    exit 1
  fi

  SELECTED="${OUT_DIR}/bon1_policy_from_n${ROLLOUT_N}_selected.xlsx"
  SUMMARY="${OUT_DIR}/bon1_policy_from_n${ROLLOUT_N}_selection.json"
  EVAL_OUT="${OUT_DIR}/bon1_policy_from_n${ROLLOUT_N}_eval.json"

  SELECT_ARGS=()
  if [[ "${FALLBACK_FIRST_VALID}" == "1" ]]; then
    SELECT_ARGS+=(--fallback-first-valid)
  fi

  echo "==> ${DATASET}: select Bo1 policy candidate ${CANDIDATE_IDX} from cached ${ROLLOUT_N} rollouts"
  python "${SCRIPT_DIR}/select_bo1_from_rollouts.py" \
    --dataset "${DATASET}" \
    --rollouts "${ROLLOUTS}" \
    --output-xlsx "${SELECTED}" \
    --summary-output "${SUMMARY}" \
    --candidate-idx "${CANDIDATE_IDX}" \
    --prediction-mode "${PREDICTION_MODE}" \
    "${SELECT_ARGS[@]}"

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "==> ${DATASET}: evaluate Bo1 policy"
    python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
      --dataset "${DATASET}" \
      --prediction "${SELECTED}" \
      --output "${EVAL_OUT}" \
      --api-nproc "${EVAL_API_NPROC}"
  fi

  COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
  print_progress "${DATASET} Bo1 done"
done
