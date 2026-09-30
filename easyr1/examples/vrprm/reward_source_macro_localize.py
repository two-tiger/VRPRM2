# Copyright 2024 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0.
"""VRPRM RL reward v2: step scoring + FIRST-ERROR LOCALIZATION.

Extends reward_source_macro.py (SFT checkpoint and RL data stay unchanged)
with a localization term that couples the reasoning trace to the judgments:

- The <answer> block additionally ends with one line ``FirstError: j`` where
  j is the index of the first incorrect step, or -1 when every step is
  correct. The guided regex in train_vrprm_rl_v2.sh enforces this grammar,
  so the format is learnable by RL alone (no SFT patch needed).
- ``R_localize = 1`` iff the declared index equals the first ground-truth
  negative step (from the existing step_labels; no data changes).
- ``R_cons = min(macro_f1, R_localize)``: localization only pays when the
  step judgments are also right, which blocks "guess the error index"
  shortcuts and gives gradient only to trajectories whose reasoning actually
  supports the judgments.
- The think-length band is widened ([80, 2400] chars by default) so RL can
  grow the rationale when localization pressure demands it, while the
  rollout token budget (2048) stays the hard cap against runaway thinking.

Default weights: R = (0.75 * step + 0.20 * cons + 0.04 * format
+ 0.01 * think) / 1.00.
"""

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_source_macro import (  # noqa: E402
    REWARD_TYPE,
    _labels_from_target,
    _load_ground_truth,
    _parse_blocks,
    _parse_step_scores,
    _source_macro_metrics,
    _think_score,
)

REWARD_NAME = "visualprm_source_macro_localize"

_FIRST_ERROR_PATTERN = re.compile(r"FirstError\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def _expected_first_error(labels: list[int]) -> int:
    for idx, label in enumerate(labels):
        if label == 0:
            return idx
    return -1


def _localize_score(
    answer: str,
    labels: list[int],
) -> tuple[float, int | None, int]:
    """Return (localize_score, predicted_index, expected_index).

    Unparseable or out-of-range declarations score 0. Out-of-range means the
    declared index cannot point at any candidate step (and is not -1).
    """
    expected = _expected_first_error(labels)
    match = _FIRST_ERROR_PATTERN.search(answer or "")
    if match is None:
        return 0.0, None, expected
    predicted = int(match.group(1))
    if predicted != -1 and not 0 <= predicted < len(labels):
        return 0.0, predicted, expected
    return (1.0 if predicted == expected else 0.0), predicted, expected


def compute_score(
    reward_inputs: list[dict[str, Any]],
    step_weight: float = 0.75,
    cons_weight: float = 0.20,
    format_weight: float = 0.04,
    think_weight: float = 0.01,
    count_weight: float = 0.15,
    min_think_chars: int = 80,
    max_think_chars: int = 2400,
) -> list[dict[str, float]]:
    scores = []
    total_weight = step_weight + cons_weight + format_weight + think_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    for reward_input in reward_inputs:
        think, answer, format_score = _parse_blocks(reward_input.get("response", ""))
        predictions = _parse_step_scores(answer)
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        labels = _labels_from_target(target)
        step_metrics = _source_macro_metrics(predictions, labels, count_weight=count_weight)
        think_quality = _think_score(
            think, min_think_chars=min_think_chars, max_think_chars=max_think_chars
        )

        localize, predicted_error, expected_error = _localize_score(answer, labels)
        # Localization only counts when the step judgments support it.
        cons = min(step_metrics["macro_f1"], localize)

        # Small format bonus once the FirstError line appears (the guided
        # regex enforces it structurally; this rewards the transition).
        declared = _FIRST_ERROR_PATTERN.search(answer or "") is not None
        if declared:
            format_score = min(1.0, format_score + 0.05)

        overall = (
            step_weight * step_metrics["step"]
            + cons_weight * cons
            + format_weight * format_score
            + think_weight * think_quality
        ) / total_weight

        scores.append(
            {
                "overall": float(overall),
                "format": format_score,
                "think": think_quality,
                "cons": cons,
                "localize": localize,
                "first_error_pred": -1.0 if predicted_error is None else float(predicted_error),
                "first_error_target": float(expected_error),
                **step_metrics,
            }
        )
    return scores
