# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from typing import Any


REWARD_NAME = "visualprm_step_level"
REWARD_TYPE = "batch"

_SCORE_PATTERN = re.compile(r"^\s*([01])\s*$")
_FALLBACK_SCORE_PATTERN = re.compile(r"([01])")


def _load_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8")
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _target_label(target: dict[str, Any]) -> int:
    label = target.get("label")
    try:
        label = int(label)
    except (TypeError, ValueError):
        return -1
    return label if label in (0, 1) else -1


def _parse_score(response: str, allow_fallback_parse: bool) -> tuple[int, float]:
    response = response or ""
    strict = _SCORE_PATTERN.fullmatch(response)
    if strict:
        return int(strict.group(1)), 1.0

    if allow_fallback_parse:
        fallback = _FALLBACK_SCORE_PATTERN.search(response)
        if fallback:
            return int(fallback.group(1)), 0.3
    return -1, 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    allow_fallback_parse: bool = False,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    invalid_reward: float = 0.0,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        pred, format_score = _parse_score(
            reward_input.get("response", ""),
            allow_fallback_parse=allow_fallback_parse,
        )
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        label = _target_label(target)

        if pred not in (0, 1) or label not in (0, 1):
            overall = invalid_reward
            accuracy = 0.0
        elif pred == label:
            overall = correct_reward
            accuracy = 1.0
        else:
            overall = incorrect_reward
            accuracy = 0.0

        scores.append(
            {
                "overall": float(overall),
                "accuracy": accuracy,
                "format": format_score,
                "pred": float(pred),
                "label": float(label),
                "negative_accuracy": float(pred == label == 0),
                "positive_accuracy": float(pred == label == 1),
                "is_negative": float(label == 0),
                "is_positive": float(label == 1),
            }
        )
    return scores
