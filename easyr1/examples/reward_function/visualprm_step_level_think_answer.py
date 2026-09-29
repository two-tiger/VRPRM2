# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from typing import Any


REWARD_NAME = "visualprm_step_level_think_answer"
REWARD_TYPE = "batch"

_STRICT_PATTERN = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
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
    try:
        label = int(target.get("label"))
    except (TypeError, ValueError):
        return -1
    return label if label in (0, 1) else -1


def _parse_response(response: str, allow_fallback_parse: bool) -> tuple[int, str, float, float]:
    response = response or ""
    strict = _STRICT_PATTERN.fullmatch(response)
    if strict:
        think = strict.group("think").strip()
        answer = strict.group("answer").strip()
        score_match = _SCORE_PATTERN.fullmatch(answer)
        pred = int(score_match.group(1)) if score_match else -1
        format_score = 1.0 if think and pred in (0, 1) else 0.5
        return pred, think, format_score, 1.0 if score_match else 0.0

    think_match = _THINK_PATTERN.search(response)
    answer_match = _ANSWER_PATTERN.search(response)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else response.strip()

    score_match = _SCORE_PATTERN.fullmatch(answer)
    if score_match:
        pred = int(score_match.group(1))
        score_format = 1.0
    elif allow_fallback_parse:
        fallback = _FALLBACK_SCORE_PATTERN.search(answer)
        pred = int(fallback.group(1)) if fallback else -1
        score_format = 0.3 if fallback else 0.0
    else:
        pred = -1
        score_format = 0.0

    format_score = 0.0
    if think:
        format_score += 0.35
    if answer_match:
        format_score += 0.35
    if response.strip().startswith("<think>"):
        format_score += 0.15
    if "</answer>" in response.lower():
        format_score += 0.15
    return pred, think, min(format_score, 0.8), score_format


def _think_score(think: str, min_think_chars: int, max_think_chars: int) -> float:
    if not think:
        return 0.0
    length = len(think)
    if length < min_think_chars:
        return max(0.0, length / max(min_think_chars, 1))
    if length > max_think_chars:
        return max(0.0, 1.0 - (length - max_think_chars) / max(max_think_chars, 1))
    return 1.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    allow_fallback_parse: bool = False,
    correctness_weight: float = 0.90,
    format_weight: float = 0.05,
    think_weight: float = 0.05,
    min_think_chars: int = 40,
    max_think_chars: int = 800,
    invalid_reward: float = 0.0,
) -> list[dict[str, float]]:
    scores = []
    total_weight = correctness_weight + format_weight + think_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    for reward_input in reward_inputs:
        pred, think, format_score, score_format = _parse_response(
            reward_input.get("response", ""),
            allow_fallback_parse=allow_fallback_parse,
        )
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        label = _target_label(target)

        if pred not in (0, 1) or label not in (0, 1):
            correctness = invalid_reward
        else:
            correctness = float(pred == label)
        think_quality = _think_score(think, min_think_chars=min_think_chars, max_think_chars=max_think_chars)

        overall = (
            correctness_weight * correctness
            + format_weight * format_score
            + think_weight * think_quality
        ) / total_weight

        scores.append(
            {
                "overall": float(overall),
                "accuracy": correctness,
                "format": format_score,
                "think": think_quality,
                "score_format": score_format,
                "pred": float(pred),
                "label": float(label),
                "negative_accuracy": float(pred == label == 0),
                "positive_accuracy": float(pred == label == 1),
                "is_negative": float(label == 0),
                "is_positive": float(label == 1),
                "think_chars": float(len(think)),
            }
        )
    return scores
