#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_LABELS="${MODEL_LABELS:-${MODEL_LABEL:-InternVL2.5-8B}}"
DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
K_LIST="${K_LIST:-1,2,4,8,16,32,64,128}"

INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs_pass_major}"
SKIP_ERRORS="${SKIP_ERRORS:-1}"
NUMERIC_TOL="${NUMERIC_TOL:-1e-6}"

IFS=',' read -r -a MODEL_LABEL_ARRAY <<< "${MODEL_LABELS}"
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

COMBINED_CSV="${OUTPUT_ROOT}/pass_major_from_n${ROLLOUT_N}_summary.csv"
mkdir -p "${OUTPUT_ROOT}"
printf 'model_label,dataset,metric_mode,k,pass_at_k,major_at_k,num_samples\n' > "${COMBINED_CSV}"

for MODEL_LABEL in "${MODEL_LABEL_ARRAY[@]}"; do
  MODEL_LABEL="$(echo "${MODEL_LABEL}" | xargs)"
  [[ -z "${MODEL_LABEL}" ]] && continue

  for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="$(echo "${DATASET}" | xargs)"
    [[ -z "${DATASET}" ]] && continue

    IN_DIR="${INPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    ROLLOUTS="${IN_DIR}/rollouts_n${ROLLOUT_N}.jsonl"
    SUMMARY="${OUT_DIR}/pass_major_from_n${ROLLOUT_N}_metrics.json"
    DETAILS="${OUT_DIR}/pass_major_from_n${ROLLOUT_N}_details.jsonl"
    DATASET_CSV="${OUT_DIR}/pass_major_from_n${ROLLOUT_N}_metrics.csv"

    if [[ ! -f "${ROLLOUTS}" ]]; then
      echo "WARNING: skip missing rollouts file: ${ROLLOUTS}" >&2
      continue
    fi

    echo "==> ${MODEL_LABEL}/${DATASET}: evaluate Pass@K and Major@K from ${ROLLOUTS}"
    python "${SCRIPT_DIR}/evaluate_pass_major_from_rollouts.py" \
      --rollouts "${ROLLOUTS}" \
      --dataset "${DATASET}" \
      --model-label "${MODEL_LABEL}" \
      --k-list "${K_LIST}" \
      --summary-output "${SUMMARY}" \
      --details-output "${DETAILS}" \
      --csv-output "${DATASET_CSV}" \
      --max-candidates "${ROLLOUT_N}" \
      --skip-errors "${SKIP_ERRORS}" \
      --numeric-tol "${NUMERIC_TOL}"

    tail -n +2 "${DATASET_CSV}" >> "${COMBINED_CSV}"
  done
done

echo "Combined summary: ${COMBINED_CSV}"
