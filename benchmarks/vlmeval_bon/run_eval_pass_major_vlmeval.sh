#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${VRPRM_ROOT}/../third_party/VLMEvalKit:${PYTHONPATH:-}"

MODEL_LABELS="${MODEL_LABELS:-${MODEL_LABEL:-InternVL2.5-8B}}"
DATASETS="${DATASETS:-MMMU,MathVista,MathVision,MathVerse-VO,WeMath,LogicVista}"
ROLLOUT_N="${ROLLOUT_N:-128}"
K_LIST="${K_LIST:-1,2,4,8,16,32,64,128}"

INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs_pass_major_vlmeval}"

BUILD_PREDICTIONS="${BUILD_PREDICTIONS:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
AGGREGATE="${AGGREGATE:-0}"
OPTIMIZED="${OPTIMIZED:-1}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-0}"

CANDIDATE_PREDICTION_MODE="${CANDIDATE_PREDICTION_MODE:-raw}"
SKIP_ERRORS="${SKIP_ERRORS:-1}"
WEMATH_PRIMARY="${WEMATH_PRIMARY:-strict}"
NUMERIC_TOL="${NUMERIC_TOL:-1e-6}"

EVAL_API_NPROC="${EVAL_API_NPROC:-4}"
EVAL_RETRY="${EVAL_RETRY:-3}"
EVAL_JUDGE="${EVAL_JUDGE:-}"
EVAL_JUDGE_ARGS="${EVAL_JUDGE_ARGS:-}"
EVAL_VERBOSE="${EVAL_VERBOSE:-0}"

IFS=',' read -r -a MODEL_LABEL_ARRAY <<< "${MODEL_LABELS}"
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a K_ARRAY <<< "${K_LIST}"

MAX_K=0
for K in "${K_ARRAY[@]}"; do
  K="$(echo "${K}" | xargs)"
  [[ -z "${K}" ]] && continue
  if (( K > MAX_K )); then
    MAX_K="${K}"
  fi
done

if (( MAX_K < 1 )); then
  echo "ERROR: K_LIST must contain at least one positive integer." >&2
  exit 1
fi

COMBINED_CSV="${OUTPUT_ROOT}/pass_major_vlmeval_from_n${ROLLOUT_N}_summary.csv"
mkdir -p "${OUTPUT_ROOT}"
printf 'model_label,dataset,metric_mode,k,pass_at_k,major_at_k,pass_metric_name,major_metric_name,num_samples\n' > "${COMBINED_CSV}"

for MODEL_LABEL in "${MODEL_LABEL_ARRAY[@]}"; do
  MODEL_LABEL="$(echo "${MODEL_LABEL}" | xargs)"
  [[ -z "${MODEL_LABEL}" ]] && continue

  for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET="$(echo "${DATASET}" | xargs)"
    [[ -z "${DATASET}" ]] && continue

    IN_DIR="${INPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    OUT_DIR="${OUTPUT_ROOT}/${MODEL_LABEL}/${DATASET}"
    ROLLOUTS="${IN_DIR}/rollouts_n${ROLLOUT_N}.jsonl"
    MANIFEST="${OUT_DIR}/pass_major_vlmeval_from_n${ROLLOUT_N}_manifest.json"
    SUMMARY="${OUT_DIR}/pass_major_vlmeval_from_n${ROLLOUT_N}_metrics.json"
    DATASET_CSV="${OUT_DIR}/pass_major_vlmeval_from_n${ROLLOUT_N}_metrics.csv"

    if [[ "${OPTIMIZED}" == "1" && "${BUILD_PREDICTIONS}" == "1" && "${RUN_EVAL}" == "1" && "${AGGREGATE}" == "1" ]]; then
      if [[ ! -f "${ROLLOUTS}" ]]; then
        echo "WARNING: skip missing rollouts file: ${ROLLOUTS}" >&2
        continue
      fi
      OPT_ARGS=()
      if [[ -n "${EVAL_JUDGE}" ]]; then
        OPT_ARGS+=(--judge "${EVAL_JUDGE}")
      fi
      if [[ -n "${EVAL_JUDGE_ARGS}" ]]; then
        OPT_ARGS+=(--judge-args "${EVAL_JUDGE_ARGS}")
      fi
      if [[ "${EVAL_VERBOSE}" == "1" ]]; then
        OPT_ARGS+=(--verbose)
      fi
      if [[ "${OVERWRITE_EVAL}" == "1" ]]; then
        OPT_ARGS+=(--overwrite)
      fi

      echo "==> ${MODEL_LABEL}/${DATASET}: optimized official Pass@K with early stop and local Major@K vote"
      python "${SCRIPT_DIR}/evaluate_pass_major_vlmeval_optimized.py" \
        --rollouts "${ROLLOUTS}" \
        --dataset "${DATASET}" \
        --model-label "${MODEL_LABEL}" \
        --k-list "${K_LIST}" \
        --output-dir "${OUT_DIR}" \
        --summary-output "${SUMMARY}" \
        --csv-output "${DATASET_CSV}" \
        --rollout-n "${ROLLOUT_N}" \
        --candidate-prediction-mode "${CANDIDATE_PREDICTION_MODE}" \
        --skip-errors "${SKIP_ERRORS}" \
        --wemath-primary "${WEMATH_PRIMARY}" \
        --numeric-tol "${NUMERIC_TOL}" \
        --api-nproc "${EVAL_API_NPROC}" \
        --retry "${EVAL_RETRY}" \
        "${OPT_ARGS[@]}"
      tail -n +2 "${DATASET_CSV}" >> "${COMBINED_CSV}"
      continue
    fi

    if [[ "${BUILD_PREDICTIONS}" == "1" ]]; then
      if [[ ! -f "${ROLLOUTS}" ]]; then
        echo "WARNING: skip missing rollouts file: ${ROLLOUTS}" >&2
        continue
      fi
      echo "==> ${MODEL_LABEL}/${DATASET}: build VLMEvalKit Pass@K/Major@K prediction files"
      python "${SCRIPT_DIR}/build_pass_major_vlmeval_predictions.py" \
        --rollouts "${ROLLOUTS}" \
        --dataset "${DATASET}" \
        --model-label "${MODEL_LABEL}" \
        --k-list "${K_LIST}" \
        --output-dir "${OUT_DIR}" \
        --manifest-output "${MANIFEST}" \
        --rollout-n "${ROLLOUT_N}" \
        --candidate-prediction-mode "${CANDIDATE_PREDICTION_MODE}" \
        --skip-errors "${SKIP_ERRORS}"
    fi

    if [[ ! -f "${MANIFEST}" ]]; then
      echo "ERROR: missing manifest: ${MANIFEST}" >&2
      echo "Run with BUILD_PREDICTIONS=1 first." >&2
      exit 1
    fi

    if [[ "${RUN_EVAL}" == "1" ]]; then
      for (( CANDIDATE_IDX=0; CANDIDATE_IDX<MAX_K; CANDIDATE_IDX++ )); do
        SELECTED="${OUT_DIR}/pass_candidate$(printf '%03d' "${CANDIDATE_IDX}")_from_n${ROLLOUT_N}_selected.xlsx"
        EVAL_OUT="${OUT_DIR}/pass_candidate$(printf '%03d' "${CANDIDATE_IDX}")_from_n${ROLLOUT_N}_eval.json"
        if [[ ! -f "${SELECTED}" ]]; then
          echo "ERROR: missing candidate prediction file: ${SELECTED}" >&2
          exit 1
        fi
        if [[ -f "${EVAL_OUT}" && "${OVERWRITE_EVAL}" != "1" ]]; then
          echo "==> ${MODEL_LABEL}/${DATASET}: skip existing candidate ${CANDIDATE_IDX} eval"
          continue
        fi
        EVAL_ARGS=()
        if [[ -n "${EVAL_JUDGE}" ]]; then
          EVAL_ARGS+=(--judge "${EVAL_JUDGE}")
        fi
        if [[ -n "${EVAL_JUDGE_ARGS}" ]]; then
          EVAL_ARGS+=(--judge-args "${EVAL_JUDGE_ARGS}")
        fi
        if [[ "${EVAL_VERBOSE}" == "1" ]]; then
          EVAL_ARGS+=(--verbose)
        fi
        echo "==> ${MODEL_LABEL}/${DATASET}: evaluate Pass candidate ${CANDIDATE_IDX}"
        python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
          --dataset "${DATASET}" \
          --prediction "${SELECTED}" \
          --output "${EVAL_OUT}" \
          --api-nproc "${EVAL_API_NPROC}" \
          --retry "${EVAL_RETRY}" \
          "${EVAL_ARGS[@]}"
      done

      for K in "${K_ARRAY[@]}"; do
        K="$(echo "${K}" | xargs)"
        [[ -z "${K}" ]] && continue
        SELECTED="${OUT_DIR}/major${K}_from_n${ROLLOUT_N}_selected.xlsx"
        EVAL_OUT="${OUT_DIR}/major${K}_from_n${ROLLOUT_N}_eval.json"
        if [[ ! -f "${SELECTED}" ]]; then
          echo "ERROR: missing Major@${K} prediction file: ${SELECTED}" >&2
          exit 1
        fi
        if [[ -f "${EVAL_OUT}" && "${OVERWRITE_EVAL}" != "1" ]]; then
          echo "==> ${MODEL_LABEL}/${DATASET}: skip existing Major@${K} eval"
          continue
        fi
        EVAL_ARGS=()
        if [[ -n "${EVAL_JUDGE}" ]]; then
          EVAL_ARGS+=(--judge "${EVAL_JUDGE}")
        fi
        if [[ -n "${EVAL_JUDGE_ARGS}" ]]; then
          EVAL_ARGS+=(--judge-args "${EVAL_JUDGE_ARGS}")
        fi
        if [[ "${EVAL_VERBOSE}" == "1" ]]; then
          EVAL_ARGS+=(--verbose)
        fi
        echo "==> ${MODEL_LABEL}/${DATASET}: evaluate Major@${K}"
        python "${SCRIPT_DIR}/evaluate_vlmeval.py" \
          --dataset "${DATASET}" \
          --prediction "${SELECTED}" \
          --output "${EVAL_OUT}" \
          --api-nproc "${EVAL_API_NPROC}" \
          --retry "${EVAL_RETRY}" \
          "${EVAL_ARGS[@]}"
      done
    fi

    if [[ "${AGGREGATE}" == "1" ]]; then
      echo "==> ${MODEL_LABEL}/${DATASET}: aggregate official Pass@K and Major@K"
      python "${SCRIPT_DIR}/aggregate_pass_major_vlmeval.py" \
        --manifest "${MANIFEST}" \
        --summary-output "${SUMMARY}" \
        --csv-output "${DATASET_CSV}" \
        --wemath-primary "${WEMATH_PRIMARY}"
      tail -n +2 "${DATASET_CSV}" >> "${COMBINED_CSV}"
    fi
  done
done

echo "Combined summary: ${COMBINED_CSV}"
