#!/usr/bin/env bash
# Control (revision plan M3/M2): SFT from the BASE model directly on the SAME
# 40K fully confident non-CoT examples used for RL — no CoT cold start, no RL.
# Together with VRPRM-SFT (5.5K CoT) and VRPRM-RL it separates the
# contributions of (a) extra non-CoT data and (b) the RL algorithm.
#
# 1) Build the dataset from the RL split (after prepare_rl_data.sh):
#      python data_pipeline/build_noncot_sft_from_rl.py \
#        --input easyr1/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k/train.jsonl \
#        --output rollout_outputs/visualprm400k_noncot40k_stepwise_sft.json
# 2) Run this script.
#
# Hyperparameters follow train_vrprm_sft.sh except: 1 epoch (the dataset is
# ~29x larger than the 1,378 CoT conversations) and a lighter save cadence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRPRM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_PATH="${NONCOT_DATASET_PATH:-${VRPRM_ROOT}/rollout_outputs/visualprm400k_noncot40k_stepwise_sft.json}"
export THINKING_MULTITURN_DATASET_PATH="${DATASET_PATH}"
export THINKING_MULTITURN_OUTPUT_DIR="${NONCOT_OUTPUT_DIR:-${VRPRM_ROOT}/sft/output/qwen3_vl_8b_noncot40k_stepwise_sft}"

# One pass over 40K conversations; three epochs would cost 3x for a control run.
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"

echo "[40K non-CoT SFT control] dataset: ${DATASET_PATH}"
echo "[40K non-CoT SFT control] output : ${THINKING_MULTITURN_OUTPUT_DIR}"

exec bash "${SCRIPT_DIR}/train_vrprm_sft.sh"
