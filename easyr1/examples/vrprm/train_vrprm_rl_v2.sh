#!/usr/bin/env bash
# VRPRM RL v2 (exploratory): unlock the optimization + FirstError localization
# reward + entropy-ordered data. The SFT checkpoint stays FROZEN.
#
# Differences vs. the paper run (train_vrprm_rl.sh), all RL-side:
#
#   optimization   lr 5e-9 -> 1e-7, KL coef 0.2 -> 1e-2, 600 -> 1200 steps,
#                  rollout temperature 0.8 -> 1.0 (top-p 1.0). The paper
#                  setting was intentionally conservative and barely moved
#                  the policy; these values release that headroom.
#   filtering      online_filtering on [0.05, 0.95]: degenerate groups (all
#                  rollouts fully right or fully wrong) are skipped instead
#                  of wasting rollout compute (J1/DAPO/MJ1 practice).
#   reward         reward_source_macro_localize.py: <answer> gains a
#                  "FirstError: j" line (guided-decoding enforced, learnable
#                  without SFT). R_cons = min(macro-F1, localization) couples
#                  the rationale to the judgments; think band widened to
#                  [80, 2400] chars (hard cap stays max_response_length).
#   data           entropy-ordered RL split (uncertainty-first curriculum):
#                  run run_entropy_select.sh once against the merged SFT
#                  checkpoint; it reorders the 40K pool so the samples the
#                  SFT model is least consistent on are trained first.
#                  DATA_SHUFFLE=false preserves the order.
#
# Every knob remains overridable; this overlay only changes defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- optimization headroom -------------------------------------------------
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"
export KL_COEF="${KL_COEF:-1.0e-2}"
export MAX_STEPS="${MAX_STEPS:-1200}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"

# ---- degenerate-group filtering --------------------------------------------
export ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
export FILTER_LOW="${FILTER_LOW:-0.05}"
export FILTER_HIGH="${FILTER_HIGH:-0.95}"

# ---- FirstError localization reward ----------------------------------------
export REWARD_FUNCTION="${REWARD_FUNCTION:-./examples/vrprm/reward_source_macro_localize.py:compute_score}"
export FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/vrprm/vrprm_source_macro_localize.jinja}"
export REWARD_KWARGS_JSON="${REWARD_KWARGS_JSON:-{\"step_weight\":${STEP_WEIGHT:-0.75},\"cons_weight\":${CONS_WEIGHT:-0.20},\"format_weight\":${FORMAT_WEIGHT:-0.04},\"think_weight\":${THINK_WEIGHT:-0.01},\"count_weight\":${COUNT_WEIGHT:-0.15},\"min_think_chars\":${MIN_THINK_CHARS:-80},\"max_think_chars\":${MAX_THINK_CHARS:-2400}}}"
# Guided grammar now REQUIRES the FirstError line (JSON-escaped backslashes).
export GUIDED_REGEX="${GUIDED_REGEX:-<think>[\\\\s\\\\S]*</think>\\\\s*<answer>(\\\\s*Step\\\\s*[0-9]+\\\\s*:\\\\s*[01]\\\\s*)+(\\\\s*FirstError\\\\s*:\\\\s*-?[0-9]+\\\\s*)</answer>}"

# ---- entropy-ordered data ---------------------------------------------------
export DATASET_DIR="${DATASET_DIR:-${EASYR1_ROOT}/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k_entropy}"
if [[ ! -f "${DATASET_DIR}/.entropy_selected" ]]; then
  echo "Entropy-ordered dataset not found: ${DATASET_DIR}" >&2
  echo "Run: bash ${SCRIPT_DIR}/run_entropy_select.sh   (serves/queries the merged SFT checkpoint)" >&2
  echo "Or set DATASET_DIR to the base split to train without entropy ordering." >&2
  exit 1
fi
export DATA_SHUFFLE="${DATA_SHUFFLE:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vrprm_rl_v2_localize_entropy_40k_1200}"

echo "[v2] lr=${LEARNING_RATE} kl=${KL_COEF} steps=${MAX_STEPS} temp=${ROLLOUT_TEMPERATURE}"
echo "[v2] reward=${REWARD_FUNCTION}"
echo "[v2] dataset=${DATASET_DIR} (entropy-ordered, shuffle=${DATA_SHUFFLE})"
echo "[v2] experiment=${EXPERIMENT_NAME}"

exec bash "${SCRIPT_DIR}/train_vrprm_rl.sh"
