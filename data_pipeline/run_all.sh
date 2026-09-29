#!/usr/bin/env bash
# Build the VRPRM SFT dataset end to end with the paper's default parameters.
#
# Stage 1: teacher rollout (Kimi K2.6 behind an OpenAI-compatible endpoint),
#          negative and positive polarity pools with conservative MC labels.
# Stage 2: merge accepted rollouts into evaluation-aligned multi-turn
#          conversations with a ~35% trainable negative step ratio.
#
# Protocol note (docs/PROTOCOL.md): SFT labels use tau- = 0.125 / tau+ = 0.75.
# If the paper moves to a unified tau+ = 0.875, change POSITIVE_THRESHOLD here
# and rebuild; the builder masks newly uncertain steps as loss=false context.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

NEGATIVE_THRESHOLD="${NEGATIVE_THRESHOLD:-0.125}"
POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.75}"
SAMPLE_TOTAL="${SAMPLE_TOTAL:-10000}"
TARGET_NEGATIVE_STEP_RATIO="${TARGET_NEGATIVE_STEP_RATIO:-0.4}"
NUM_WORKERS="${NUM_WORKERS:-32}"
TARGET_NEGATIVE_RATIO="${TARGET_NEGATIVE_RATIO:-0.35}"
SEED="${SEED:-42}"

echo "Stage 1/2: teacher rollout (negative pool)"
python3 "${SCRIPT_DIR}/rollout_teacher.py" \
  --polarity negative \
  --negative-score-threshold "${NEGATIVE_THRESHOLD}" \
  --positive-score-threshold "${POSITIVE_THRESHOLD}" \
  --sample-total "${SAMPLE_TOTAL}" \
  --target-negative-step-ratio "${TARGET_NEGATIVE_STEP_RATIO}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  "${@:-}"

echo "Stage 1/2: teacher rollout (positive pool)"
python3 "${SCRIPT_DIR}/rollout_teacher.py" \
  --polarity positive \
  --negative-score-threshold "${NEGATIVE_THRESHOLD}" \
  --positive-score-threshold "${POSITIVE_THRESHOLD}" \
  --sample-total "${SAMPLE_TOTAL}" \
  --target-negative-step-ratio "${TARGET_NEGATIVE_STEP_RATIO}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  "${@:-}"

echo "Stage 2/2: build multi-turn SFT dataset (neg ratio ${TARGET_NEGATIVE_RATIO})"
python3 "${SCRIPT_DIR}/build_multiturn.py" \
  --target-negative-ratio "${TARGET_NEGATIVE_RATIO}" \
  --seed "${SEED}"

echo "Done. Outputs are under rollout_outputs/ (see data_pipeline stats JSON)."
