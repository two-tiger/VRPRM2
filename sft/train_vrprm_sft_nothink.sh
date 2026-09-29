#!/usr/bin/env bash
# Ablation (revision plan M5): retrain the SFT model WITHOUT the global-thinking
# turn but with the SAME 5,592 trainable step labels and the SAME stepwise
# single-token interface. This isolates the CoT *training signal* from
# inference-time thinking (the "w/o Thinking" rows in the paper skip thinking
# at inference on a think-trained checkpoint and therefore measure something
# different).
#
# 1) Build the no-think dataset from the SAME accepted teacher rollouts:
#      python data_pipeline/build_multiturn.py --no-think
#    (writes rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35_nothink.json)
# 2) Run this script.
#
# Training hyperparameters are inherited from train_vrprm_sft.sh; only the
# dataset, output dir, and system-prompt validation differ.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${NOTHINK_DATASET_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)/rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35_nothink.json}"
export THINKING_MULTITURN_DATASET_PATH="${DATASET_PATH}"
export THINKING_MULTITURN_OUTPUT_DIR="${NOTHINK_OUTPUT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft_nothink}"

echo "[no-think ablation] dataset: ${DATASET_PATH}"
echo "[no-think ablation] output : ${THINKING_MULTITURN_OUTPUT_DIR}"

exec bash "${SCRIPT_DIR}/train_vrprm_sft.sh"
