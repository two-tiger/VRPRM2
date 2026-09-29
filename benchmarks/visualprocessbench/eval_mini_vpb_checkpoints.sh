#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${VRPRM_ROOT}/.." && pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${VRPRM_ROOT}/easyr1/checkpoints/visualprm_easy_r1/qwen3_vl_8b_source_macro_visualprm400k_gspo_lora_clean_balanced_40k_600}"
STEPS="${STEPS:-50,100,150,200,250,300,350,400,450,500,550,600}"
MODEL_SUBDIR="${MODEL_SUBDIR:-actor/huggingface}"
MINI_BENCHMARK_DIR="${MINI_BENCHMARK_DIR:-${SCRIPT_DIR}/VisualProcessBench_mini_dev}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/mini_vpb_checkpoint_selection}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/summary.jsonl}"

AUTO_SERVE="${AUTO_SERVE:-1}"
WAIT_FOR_CHECKPOINTS="${WAIT_FOR_CHECKPOINTS:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-21600}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"
PORT_BASE="${PORT_BASE:-8000}"
PORTS="${PORTS:-}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
VPB_BASE_URL="${VPB_BASE_URL:-}"
VPB_MODEL_PREFIX="${VPB_MODEL_PREFIX:-vrprm-v2-rl-clean-balanced-40k}"
VPB_API_KEY="${VPB_API_KEY:-EMPTY}"
VPB_CONCURRENCY="${VPB_CONCURRENCY:-64}"
VPB_THINK_MAX_TOKENS="${VPB_THINK_MAX_TOKENS:-1024}"
VPB_USE_REFERENCE_ANSWER="${VPB_USE_REFERENCE_ANSWER:-0}"
VPB_TEMPERATURE="${VPB_TEMPERATURE:-0.0}"
VPB_MAX_RETRIES="${VPB_MAX_RETRIES:-3}"
VPB_REQUEST_TIMEOUT="${VPB_REQUEST_TIMEOUT:-300}"
VPB_STATUS_EVERY="${VPB_STATUS_EVERY:-20}"
VPB_OVERWRITE="${VPB_OVERWRITE:-1}"
VPB_RESUME="${VPB_RESUME:-0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${raw}"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

build_ports() {
  local -n out_ref="$1"
  if [[ -n "${PORTS}" ]]; then
    split_csv "${PORTS}" out_ref
    return
  fi
  out_ref=()
  local i
  for ((i = 0; i < NUM_REPLICAS; i += 1)); do
    out_ref+=("$((PORT_BASE + i))")
  done
}

build_base_urls() {
  if [[ -n "${VPB_BASE_URL}" ]]; then
    printf '%s\n' "${VPB_BASE_URL}"
    return
  fi
  local ports=()
  local urls=()
  local port
  build_ports ports
  for port in "${ports[@]}"; do
    urls+=("http://${CLIENT_HOST}:${port}/v1")
  done
  join_by_comma "${urls[@]}"
}

wait_for_checkpoint() {
  local model_dir="$1"
  if [[ -d "${model_dir}" ]]; then
    return 0
  fi
  if [[ "${WAIT_FOR_CHECKPOINTS}" != "1" ]]; then
    echo "Missing checkpoint model dir: ${model_dir}" >&2
    return 1
  fi
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while ((SECONDS < deadline)); do
    if [[ -d "${model_dir}" ]]; then
      return 0
    fi
    echo "Waiting for checkpoint: ${model_dir}"
    sleep "${WAIT_INTERVAL}"
  done
  echo "Timed out waiting for checkpoint: ${model_dir}" >&2
  return 1
}

prepare_mini_benchmark() {
  if [[ -f "${MINI_BENCHMARK_DIR}/test.jsonl" && -e "${MINI_BENCHMARK_DIR}/images" ]]; then
    return 0
  fi
  python "${SCRIPT_DIR}/prepare_mini_vpb_dev.py" \
    --benchmark-dir "${SCRIPT_DIR}/VisualProcessBench" \
    --output-dir "${MINI_BENCHMARK_DIR}" \
    --samples-per-source "${MINI_VPB_SAMPLES_PER_SOURCE:-60}" \
    --target-negative-source-ratio "${MINI_VPB_TARGET_NEGATIVE_SOURCE_RATIO:--1}" \
    --seed "${MINI_VPB_SEED:-42}"
}

stop_server() {
  local pid_file="$1"
  PID_FILE="${pid_file}" bash "${SCRIPT_DIR}/serve_sft_vllm_dp.sh" stop || true
}

start_server() {
  local model_dir="$1"
  local served_name="$2"
  local log_dir="$3"
  local pid_file="${log_dir}/pids.txt"
  stop_server "${pid_file}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_REPLICAS="${NUM_REPLICAS}" \
  PORT_BASE="${PORT_BASE}" \
  PORTS="${PORTS}" \
  CLIENT_HOST="${CLIENT_HOST}" \
  SERVED_MODEL_NAME="${served_name}" \
  LOG_DIR="${log_dir}" \
  PID_FILE="${pid_file}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
  ENFORCE_EAGER="${ENFORCE_EAGER}" \
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" \
    bash "${SCRIPT_DIR}/serve_sft_vllm_dp.sh" start "${model_dir}"
}

eval_step() {
  local step="$1"
  local model_dir="${CHECKPOINT_ROOT}/global_step_${step}/${MODEL_SUBDIR}"
  local served_name="${VPB_MODEL_PREFIX}-step-${step}"
  local output="${OUTPUT_DIR}/${served_name}_mini_global_think_stepwise_predictions.jsonl"
  local log_dir="${OUTPUT_DIR}/vllm_logs/step_${step}"

  wait_for_checkpoint "${model_dir}"
  mkdir -p "${OUTPUT_DIR}"

  if [[ "${AUTO_SERVE}" == "1" ]]; then
    start_server "${model_dir}" "${served_name}" "${log_dir}"
  else
    served_name="${VPB_MODEL:-${served_name}}"
  fi

  local base_urls
  base_urls="$(build_base_urls)"
  local args=(
    --benchmark-dir "${MINI_BENCHMARK_DIR}"
    --base-url "${base_urls}"
    --api-key "${VPB_API_KEY}"
    --model "${served_name}"
    --concurrency "${VPB_CONCURRENCY}"
    --temperature "${VPB_TEMPERATURE}"
    --think-max-tokens "${VPB_THINK_MAX_TOKENS}"
    --use-reference-answer "${VPB_USE_REFERENCE_ANSWER}"
    --max-retries "${VPB_MAX_RETRIES}"
    --request-timeout "${VPB_REQUEST_TIMEOUT}"
    --status-every "${VPB_STATUS_EVERY}"
    --output "${output}"
    --no-progress
  )
  if [[ "${VPB_OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${VPB_RESUME}" == "1" ]]; then
    args+=(--resume)
  fi
  python "${SCRIPT_DIR}/api_eval_global_think_stepwise.py" "${args[@]}"

  if [[ "${AUTO_SERVE}" == "1" ]]; then
    stop_server "${log_dir}/pids.txt"
  fi
}

write_summary() {
  python - "$OUTPUT_DIR" "$SUMMARY_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for path in sorted(output_dir.glob("*_mini_global_think_stepwise_predictions.metrics.json")):
    match = re.search(r"step-(\d+)_mini_", path.name)
    step = int(match.group(1)) if match else -1
    metrics = json.loads(path.read_text(encoding="utf-8"))
    overall = metrics["overall_step_macro_f1"]
    rows.append(
        {
            "step": step,
            "metrics_path": str(path),
            "macro_f1": overall["macro_f1"],
            "correct_f1": overall["correct_f1"],
            "incorrect_f1": overall["incorrect_f1"],
            "mean_source_macro_f1": metrics["mean_source_macro_f1"],
            "by_source_macro_f1": {
                source: value["macro_f1"] for source, value in metrics["by_source"].items()
            },
        }
    )
rows.sort(key=lambda item: item["step"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote summary: {summary_path}")
print("Top checkpoints by mini VPB macro_f1:")
for row in sorted(rows, key=lambda item: (item["macro_f1"], item["mean_source_macro_f1"]), reverse=True)[:10]:
    print(
        f"step={row['step']} macro={row['macro_f1']:.6f} "
        f"incorrect={row['incorrect_f1']:.6f} mean_source={row['mean_source_macro_f1']:.6f}"
    )
PY
}

prepare_mini_benchmark
split_csv "${STEPS}" step_list
for step in "${step_list[@]}"; do
  eval_step "${step}"
  write_summary
done
