# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from typing import Any


REWARD_NAME = "visualprm_global_then_step_macro_f1"
REWARD_TYPE = "batch"

_STRICT_PATTERN = re.compile(
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


def _parse_blocks(response: str) -> tuple[str, str, float]:
    response = response or ""
    strict = _STRICT_PATTERN.fullmatch(response)
    if strict:
        think = strict.group("think").strip()
        answer = strict.group("answer").strip()
        return think, answer, 1.0 if think and answer else 0.5

    think_match = _THINK_PATTERN.search(response)
    answer_match = _ANSWER_PATTERN.search(response)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    format_score = 0.0
    if think:
        format_score += 0.35
    if answer:
        format_score += 0.35
    if response.strip().startswith("<think>"):
        format_score += 0.15
    if "</answer>" in response.lower():
        format_score += 0.15
    return think, answer, min(format_score, 0.8)


def _parse_step_scores(answer: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    if not answer:
        return scores

    stripped = answer.strip()
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

    for match in _STEP_SCORE_PATTERN.finditer(answer):
        scores[int(match.group(1))] = int(match.group(2))
    return scores


def _binary_f1(labels: list[int], preds: list[int], positive_label: int) -> float:
    tp = fp = fn = 0
    for label, pred in zip(labels, preds):
        label_pos = label == positive_label
        pred_pos = pred == positive_label
        if label_pos and pred_pos:
            tp += 1
        elif not label_pos and pred_pos:
            fp += 1
        elif label_pos and not pred_pos:
            fn += 1
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _aggregate_step_reward(
    predictions: dict[int, int],
    labels: list[int],
    count_weight: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    confident = [(idx, label) for idx, label in enumerate(labels) if label in (0, 1)]
    if not confident:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    label_values = [label for _, label in confident]
    pred_values = [predictions.get(idx, -2) for idx, _ in confident]
    correct = sum(pred == label for label, pred in zip(label_values, pred_values))
    micro_acc = correct / len(confident)

    neg_indices = [idx for idx, label in confident if label == 0]
    pos_indices = [idx for idx, label in confident if label == 1]
    neg_acc = sum(predictions.get(idx) == 0 for idx in neg_indices) / len(neg_indices) if neg_indices else 0.0
    pos_acc = sum(predictions.get(idx) == 1 for idx in pos_indices) / len(pos_indices) if pos_indices else 0.0
    class_accs = []
    if neg_indices:
        class_accs.append(neg_acc)
    if pos_indices:
        class_accs.append(pos_acc)
    balanced_acc = sum(class_accs) / len(class_accs) if class_accs else 0.0

    correct_f1 = _binary_f1(label_values, pred_values, positive_label=1)
    incorrect_f1 = _binary_f1(label_values, pred_values, positive_label=0)
    macro_f1 = (correct_f1 + incorrect_f1) / 2

    count_error = abs(len(predictions) - len(labels))
    count_score = max(0.0, 1.0 - count_error / max(len(labels), 1))
    step_reward = (1.0 - count_weight) * macro_f1 + count_weight * count_score
    return step_reward, micro_acc, balanced_acc, neg_acc, pos_acc, count_score, correct_f1, incorrect_f1


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
    step_weight: float = 0.95,
    format_weight: float = 0.03,
    think_weight: float = 0.02,
    count_weight: float = 0.05,
    min_think_chars: int = 80,
    max_think_chars: int = 1800,
) -> list[dict[str, float]]:
    scores = []
    total_weight = step_weight + format_weight + think_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    for reward_input in reward_inputs:
        think, answer, format_score = _parse_blocks(reward_input.get("response", ""))
        predictions = _parse_step_scores(answer)
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        labels = _labels_from_target(target)

        step_reward, micro_acc, balanced_acc, neg_acc, pos_acc, count_score, correct_f1, incorrect_f1 = _aggregate_step_reward(
            predictions,
            labels,
            count_weight=count_weight,
        )
        think_quality = _think_score(think, min_think_chars=min_think_chars, max_think_chars=max_think_chars)
        overall = (
            step_weight * step_reward
            + format_weight * format_score
            + think_weight * think_quality
        ) / total_weight

        scores.append(
            {
                "overall": float(overall),
                "step": step_reward,
                "macro_f1": (correct_f1 + incorrect_f1) / 2,
                "correct_f1": correct_f1,
                "incorrect_f1": incorrect_f1,
                "accuracy": micro_acc,
                "balanced_accuracy": balanced_acc,
                "negative_accuracy": neg_acc,
                "positive_accuracy": pos_acc,
                "format": format_score,
                "think": think_quality,
                "count": count_score,
                "parsed_steps": float(len(predictions)),
                "target_steps": float(len(labels)),
            }
        )
    return scores
