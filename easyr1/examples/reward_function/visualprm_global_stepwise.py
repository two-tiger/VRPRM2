# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from typing import Any


REWARD_NAME = "visualprm_global_stepwise"
REWARD_TYPE = "batch"


_TAG_PATTERN = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_STEP_SCORE_PATTERN = re.compile(
    r"(?:^|\n)[ \t]*(?:[-*][ \t]*)?(?:Step[ \t]*)?(\d+)[ \t]*(?:[:.)\-]|=>|[ \t])[ \t]*(?:\\boxed\{)?[ \t]*([01])[ \t]*(?:\})?",
    re.IGNORECASE,
)


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


def _labels_from_target(target: dict[str, Any]) -> list[int]:
    labels = target.get("step_labels")
    if isinstance(labels, list):
        return [int(x) if x in (0, 1, -1) else -1 for x in labels]

    labels = []
    neg_thr = float(target.get("negative_threshold", 0.125))
    pos_thr = float(target.get("positive_threshold", 0.75))
    for step in target.get("steps_with_score") or []:
        try:
            score = float(step.get("score"))
        except (TypeError, ValueError):
            labels.append(-1)
            continue
        if score <= neg_thr:
            labels.append(0)
        elif score >= pos_thr:
            labels.append(1)
        else:
            labels.append(-1)
    return labels


def _parse_blocks(response: str) -> tuple[float, str, str]:
    response = response or ""
    strict = _TAG_PATTERN.fullmatch(response)
    if strict:
        think = strict.group("think").strip()
        answer = strict.group("answer").strip()
        return (1.0 if think and answer else 0.5), think, answer

    think_match = _THINK_PATTERN.search(response)
    answer_match = _ANSWER_PATTERN.search(response)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    partial = 0.0
    if think:
        partial += 0.35
    if answer:
        partial += 0.35
    if response.strip().startswith("<think>"):
        partial += 0.15
    if "</answer>" in response.lower():
        partial += 0.15
    return min(partial, 0.8), think, answer


def _parse_step_scores(text: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    if not text:
        return scores

    stripped = text.strip()
    if stripped.startswith("["):
        try:
            values = json.loads(stripped)
            if isinstance(values, list):
                for idx, value in enumerate(values):
                    if value in (0, 1):
                        scores[idx] = int(value)
                return scores
        except json.JSONDecodeError:
            pass

    if stripped.startswith("{"):
        try:
            values = json.loads(stripped)
            if isinstance(values, dict):
                for key, value in values.items():
                    match = re.search(r"\d+", str(key))
                    if match and value in (0, 1):
                        scores[int(match.group(0))] = int(value)
                return scores
        except json.JSONDecodeError:
            pass

    for match in _STEP_SCORE_PATTERN.finditer(text):
        scores[int(match.group(1))] = int(match.group(2))
    return scores


def _balanced_process_score(
    predictions: dict[int, int],
    labels: list[int],
    negative_step_weight: float,
) -> tuple[float, float, float, float, float]:
    confident = [(idx, label) for idx, label in enumerate(labels) if label in (0, 1)]
    if not confident:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    correct = sum(1 for idx, label in confident if predictions.get(idx) == label)
    micro_acc = correct / len(confident)
    weighted_correct = 0.0
    weighted_total = 0.0
    for idx, label in confident:
        weight = negative_step_weight if label == 0 else 1.0
        weighted_total += weight
        if predictions.get(idx) == label:
            weighted_correct += weight
    weighted_acc = weighted_correct / weighted_total if weighted_total else 0.0

    class_scores = []
    pos_acc = neg_acc = 0.0
    for label in (0, 1):
        indices = [idx for idx, value in confident if value == label]
        if not indices:
            continue
        acc = sum(1 for idx in indices if predictions.get(idx) == label) / len(indices)
        class_scores.append(acc)
        if label == 0:
            neg_acc = acc
        else:
            pos_acc = acc
    balanced_acc = sum(class_scores) / len(class_scores) if class_scores else 0.0

    expected_count = len(labels)
    count_error = abs(len(predictions) - expected_count)
    count_score = max(0.0, 1.0 - count_error / max(expected_count, 1))

    process = 0.75 * balanced_acc + 0.15 * weighted_acc + 0.10 * count_score
    return process, micro_acc, balanced_acc, neg_acc, pos_acc


def _thinking_alignment_score(think: str, labels: list[int], min_chars: int, max_chars: int) -> tuple[float, float, float, float]:
    if not think:
        return 0.0, 0.0, 0.0, 0.0

    length = len(think)
    if length < min_chars:
        length_score = max(0.0, length / max(min_chars, 1))
    elif length > max_chars:
        length_score = max(0.0, 1.0 - (length - max_chars) / max(max_chars, 1))
    else:
        length_score = 1.0

    confident_indices = [idx for idx, label in enumerate(labels) if label in (0, 1)]
    mentioned = [
        idx for idx in confident_indices
        if re.search(rf"\bStep\s*{idx}\b", think, flags=re.IGNORECASE)
    ]
    coverage = len(mentioned) / len(confident_indices) if confident_indices else 0.0
    score_leak = bool(
        re.search(r"\\boxed\s*\{?\s*\[", think)
        or re.search(r"\bscore\s*[:=]\s*\[", think, flags=re.IGNORECASE)
        or re.search(r"\{[^\n{}]*score[^\n{}]*\[[^\n{}]*\}", think, flags=re.IGNORECASE)
    )
    no_leak_score = 0.0 if score_leak else 1.0
    return 0.50 * length_score + 0.35 * coverage + 0.15 * no_leak_score, coverage, length_score, no_leak_score


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.15,
    process_weight: float = 0.75,
    think_weight: float = 0.10,
    min_think_chars: int = 80,
    max_think_chars: int = 2500,
    negative_step_weight: float = 2.0,
) -> list[dict[str, float]]:
    scores = []
    total_weight = format_weight + process_weight + think_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        format_score, think, answer = _parse_blocks(response)
        predictions = _parse_step_scores(answer)
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        labels = _labels_from_target(target)

        process_score, micro_acc, balanced_acc, neg_acc, pos_acc = _balanced_process_score(
            predictions,
            labels,
            negative_step_weight=negative_step_weight,
        )
        think_score, think_coverage, think_length, think_no_score_leak = _thinking_alignment_score(
            think,
            labels,
            min_chars=min_think_chars,
            max_chars=max_think_chars,
        )

        overall = (
            format_weight * format_score
            + process_weight * process_score
            + think_weight * think_score
        ) / total_weight

        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "process": process_score,
                "accuracy": micro_acc,
                "balanced_accuracy": balanced_acc,
                "negative_accuracy": neg_acc,
                "positive_accuracy": pos_acc,
                "think": think_score,
                "think_step_coverage": think_coverage,
                "think_length": think_length,
                "think_no_score_leak": think_no_score_leak,
                "parsed_steps": float(len(predictions)),
                "target_steps": float(len(labels)),
            }
        )

    return scores
