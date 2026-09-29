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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs_visualprm_bon}"
REWARD_LABEL="${REWARD_LABEL:-visualprm_soft}"

VISUALPRM_MODEL_PATH="${VISUALPRM_MODEL_PATH:-VisualPRM/VisualPRM-8B}"
VISUALPRM_DTYPE="${VISUALPRM_DTYPE:-bfloat16}"
VISUALPRM_THRESHOLD="${VISUALPRM_THRESHOLD:-0.85}"
VISUALPRM_LIMIT="${VISUALPRM_LIMIT:-0}"
VISUALPRM_LOG_EVERY="${VISUALPRM_LOG_EVERY:-10}"
VISUALPRM_GPUS="${VISUALPRM_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
VISUALPRM_PARALLEL_SCORE="${VISUALPRM_PARALLEL_SCORE:-1}"

RUN_SCORE="${RUN_SCORE:-1}"
RUN_SELECT="${RUN_SELECT:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
OVERWRITE="${OVERWRITE:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

IFS=',' read -r -a MODEL_LABEL_ARRAY <<< "${MODEL_LABELS}"
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a BON_ARRAY <<< "${BON_LIST}"
IFS=',' read -r -a GPU_ARRAY_RAW <<< "${VISUALPRM_GPUS}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

dataset_priority_rank() {
  case "$1" in
    MathVision) echo 0 ;;
    WeMath) echo 1 ;;
    MMMU) echo 2 ;;
    MathVista) echo 3 ;;
    MathVerse-VO|MathVerse|MathVerse_MINI_Vision_Only) echo 4 ;;
    LogicVista) echo 5 ;;
    DynaMath) echo 6 ;;
    *) echo 100 ;;
  esac
}

MAX_BON=0
for raw_bon in "${BON_ARRAY[@]}"; do
  bon="$(trim "${raw_bon}")"
  [[ -z "${bon}" ]] && continue
  if [[ ! "${bon}" =~ ^[0-9]+$ || "${bon}" == "0" ]]; then
    echo "ERROR: BON_LIST contains an invalid positive integer: ${bon}" >&2
    exit 1
  fi
  if (( bon > MAX_BON )); then
    MAX_BON="${bon}"
  fi
done
if (( MAX_BON < 1 )); then
  echo "ERROR: BON_LIST must contain at least one positive integer." >&2
  exit 1
fi
if [[ -z "${SCORE_BON}" ]]; then
  SCORE_BON="${MAX_BON}"
fi
if [[ ! "${SCORE_BON}" =~ ^[0-9]+$ || "${SCORE_BON}" == "0" ]]; then
  echo "ERROR: SCORE_BON must be a positive integer: ${SCORE_BON}" >&2
  exit 1
fi
if (( SCORE_BON < MAX_BON )); then
  echo "ERROR: SCORE_BON=${SCORE_BON} is smaller than max BON_LIST=${MAX_BON}" >&2
  exit 1
fi
if (( SCORE_BON > ROLLOUT_N )); then
  echo "ERROR: SCORE_BON=${SCORE_BON} is larger than ROLLOUT_N=${ROLLOUT_N}" >&2
  exit 1
fi

score_one() {
  local model_label="$1"
  local dataset="$2"
  local in_dir="${INPUT_ROOT}/${model_label}/${dataset}"
  local out_dir="${OUTPUT_ROOT}/${model_label}/${dataset}"
  local rollouts="${in_dir}/rollouts_n${ROLLOUT_N}.jsonl"
  local scores="${out_dir}/rollouts_n${ROLLOUT_N}_${REWARD_LABEL}_scores.jsonl"

  if [[ ! -f "${rollouts}" ]]; then
    echo "WARNING: skip missing rollouts file: ${rollouts}" >&2
    return 0
  fi

  mkdir -p "${out_dir}"

  local score_args=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    score_args+=(--overwrite)
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    score_args+=(--no-progress)
  fi

  echo "==> ${model_label}/${dataset}: VisualPRM score first ${SCORE_BON}/${ROLLOUT_N} candidates on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  python "${SCRIPT_DIR}/score_rollouts_with_visualprm.py" \
    --dataset "${dataset}" \
    --rollouts "${rollouts}" \
    --score-output "${scores}" \
    --model-path "${VISUALPRM_MODEL_PATH}" \
    --bon "${SCORE_BON}" \
    --dtype "${VISUALPRM_DTYPE}" \
    --threshold "${VISUALPRM_THRESHOLD}" \
    --limit "${VISUALPRM_LIMIT}" \
    --log-every "${VISUALPRM_LOG_EVERY}" \
    "${score_args[@]}"
}

select_one() {
  local model_label="$1"
  local dataset="$2"
  local in_dir="${INPUT_ROOT}/${model_label}/${dataset}"
  local out_dir="${OUTPUT_ROOT}/${model_label}/${dataset}"
  local rollouts="${in_dir}/rollouts_n${ROLLOUT_N}.jsonl"
  local scores="${out_dir}/rollouts_n${ROLLOUT_N}_${REWARD_LABEL}_scores.jsonl"

  if [[ ! -f "${rollouts}" ]]; then
    echo "WARNING: skip missing rollouts file: ${rollouts}" >&2
    return 0
  fi
  if [[ ! -f "${scores}" ]]; then
    echo "ERROR: missing score cache: ${scores}" >&2
    return 1
  fi

  for bon in "${BON_ARRAY[@]}"; do
    bon="$(trim "${bon}")"
    [[ -z "${bon}" ]] && continue
    if (( bon > ROLLOUT_N )); then
      echo "ERROR: BON=${bon} is larger than ROLLOUT_N=${ROLLOUT_N}" >&2
      return 1
    fi

    local selected="${out_dir}/bon${bon}_${REWARD_LABEL}_from_n${ROLLOUT_N}_selected.xlsx"
    local summary="${out_dir}/bon${bon}_${REWARD_LABEL}_from_n${ROLLOUT_N}_selection.json"
    local select_args=()
    if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
      select_args+=(--allow-partial)
    fi

    echo "==> ${model_label}/${dataset}: select Bo${bon} by VisualPRM"
    python "${SCRIPT_DIR}/select_bon_from_scores.py" \
      --dataset "${dataset}" \
      --score-output "${scores}" \
      --output-xlsx "${selected}" \
      --summary-output "${summary}" \
      --bon "${bon}" \
      "${select_args[@]}"
  done
}

GPU_ARRAY=()
for gpu in "${GPU_ARRAY_RAW[@]}"; do
  gpu="$(trim "${gpu}")"
  [[ -z "${gpu}" ]] && continue
  GPU_ARRAY+=("${gpu}")
done

if [[ "${RUN_SCORE}" == "1" && "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: VISUALPRM_GPUS must contain at least one GPU id when RUN_SCORE=1." >&2
  exit 1
fi

TASKS=()
while IFS=$'\t' read -r _rank _seq model_label dataset; do
  [[ -z "${model_label:-}" || -z "${dataset:-}" ]] && continue
  TASKS+=("${model_label}|${dataset}")
done < <(
  seq_id=0
  for raw_model_label in "${MODEL_LABEL_ARRAY[@]}"; do
    model_label="$(trim "${raw_model_label}")"
    [[ -z "${model_label}" ]] && continue
    for raw_dataset in "${DATASET_ARRAY[@]}"; do
      dataset="$(trim "${raw_dataset}")"
      [[ -z "${dataset}" ]] && continue
      printf '%s\t%06d\t%s\t%s\n' "$(dataset_priority_rank "${dataset}")" "${seq_id}" "${model_label}" "${dataset}"
      seq_id=$((seq_id + 1))
    done
  done | sort -n -k1,1 -k2,2
)

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "ERROR: no model/dataset tasks to process." >&2
  exit 1
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
  if [[ "${VISUALPRM_PARALLEL_SCORE}" == "1" && "${#GPU_ARRAY[@]}" -gt 1 ]]; then
    echo "VisualPRM scoring uses ${#GPU_ARRAY[@]} GPU workers: ${GPU_ARRAY[*]}"
    pids=()
    num_gpus="${#GPU_ARRAY[@]}"
    for worker_idx in "${!GPU_ARRAY[@]}"; do
      (
        export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[worker_idx]}"
        for task_idx in "${!TASKS[@]}"; do
          if (( task_idx % num_gpus != worker_idx )); then
            continue
          fi
          IFS='|' read -r model_label dataset <<< "${TASKS[task_idx]}"
          score_one "${model_label}" "${dataset}"
        done
      ) &
      pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if [[ "${status}" != "0" ]]; then
      echo "ERROR: at least one VisualPRM scoring worker failed." >&2
      exit 1
    fi
  else
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"
    echo "VisualPRM scoring uses one GPU worker: ${CUDA_VISIBLE_DEVICES}"
    for task in "${TASKS[@]}"; do
      IFS='|' read -r model_label dataset <<< "${task}"
      score_one "${model_label}" "${dataset}"
    done
  fi
fi

if [[ "${RUN_SELECT}" == "1" ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r model_label dataset <<< "${task}"
    select_one "${model_label}" "${dataset}"
  done
fi

echo "VisualPRM BoN score/selection outputs written under: ${OUTPUT_ROOT}"
