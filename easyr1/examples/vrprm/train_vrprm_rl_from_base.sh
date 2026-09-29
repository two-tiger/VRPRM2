#!/usr/bin/env bash
# Ablation (revision plan M3): RL directly from the BASE model, skipping the
# CoT cold-start SFT stage. Justifies (or falsifies) the necessity of stage 1:
# if RL-from-base matches the two-stage pipeline, the cold start is redundant;
# if it degrades or diverges, this run is the evidence for the two-stage design.
#
# All RL hyperparameters and data preparation are identical to
# train_vrprm_rl.sh; only MODEL_PATH defaults to the base model.
#
# NOTE: the base model has not been format-aligned by SFT, so expect weaker
# initial format compliance (the guided_regex still constrains rollouts) and
# watch the format reward in the logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Thinking}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vrprm_rl_from_base_40k_600}"

echo "[RL-from-base ablation] model: ${MODEL_PATH}"
echo "[RL-from-base ablation] experiment: ${EXPERIMENT_NAME}"

exec bash "${SCRIPT_DIR}/train_vrprm_rl.sh"
