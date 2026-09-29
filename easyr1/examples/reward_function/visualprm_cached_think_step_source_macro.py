# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from collections import defaultdict
from typing import Any


REWARD_NAME = "visualprm_cached_think_step_source_macro"
REWARD_TYPE = "batch"

_STRICT_SCORE_PATTERN = re.compile(r"^\s*([01])\s*$")
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


def _parse_score(response: str, allow_fallback_parse: bool) -> tuple[int, float]:
    response = response or ""
    strict = _STRICT_SCORE_PATTERN.fullmatch(response)
    if strict:
        return int(strict.group(1)), 1.0

    if allow_fallback_parse:
        fallback = _FALLBACK_SCORE_PATTERN.search(response)
        if fallback:
            return int(fallback.group(1)), 0.3
    return -1, 0.0


def _class_f1(labels: list[int], preds: list[int], klass: int) -> float | None:
    if not any(label == klass for label in labels):
        return None
    tp = sum(label == klass and pred == klass for label, pred in zip(labels, preds))
    fp = sum(label != klass and pred == klass for label, pred in zip(labels, preds))
    fn = sum(label == klass and pred != klass for label, pred in zip(labels, preds))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _source_macro_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    labels = [item["label"] for item in items if item["label"] in (0, 1)]
    preds = [item["pred"] for item in items if item["label"] in (0, 1)]
    if not labels:
        return {
            "source_macro_f1": 0.0,
            "source_negative_f1": 0.0,
            "source_positive_f1": 0.0,
            "source_accuracy": 0.0,
            "source_pred_negative_ratio": 0.0,
            "source_num_steps": 0.0,
        }

    neg_f1 = _class_f1(labels, preds, 0)
    pos_f1 = _class_f1(labels, preds, 1)
    f1_values = [value for value in (neg_f1, pos_f1) if value is not None]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    accuracy = sum(label == pred for label, pred in zip(labels, preds)) / len(labels)
    pred_negative_ratio = sum(pred == 0 for pred in preds) / len(preds)
    return {
        "source_macro_f1": float(macro_f1),
        "source_negative_f1": float(0.0 if neg_f1 is None else neg_f1),
        "source_positive_f1": float(0.0 if pos_f1 is None else pos_f1),
        "source_accuracy": float(accuracy),
        "source_pred_negative_ratio": float(pred_negative_ratio),
        "source_num_steps": float(len(labels)),
    }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    allow_fallback_parse: bool = False,
    source_macro_weight: float = 0.75,
    step_weight: float = 0.20,
    format_weight: float = 0.05,
    invalid_reward: float = 0.0,
) -> list[dict[str, float]]:
    total_weight = source_macro_weight + step_weight + format_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    parsed_items = []
    occurrence_by_step: dict[tuple[str, int], int] = defaultdict(int)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row_idx, reward_input in enumerate(reward_inputs):
        pred, format_score = _parse_score(
            reward_input.get("response", ""),
            allow_fallback_parse=allow_fallback_parse,
        )
        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        label = _target_label(target)
        step_index = int(target.get("step_index", row_idx))
        source_group_id = str(target.get("source_group_id") or target.get("source_id") or row_idx)
        rollout_slot = occurrence_by_step[(source_group_id, step_index)]
        occurrence_by_step[(source_group_id, step_index)] += 1
        item = {
            "row_idx": row_idx,
            "pred": pred,
            "label": label,
            "format": format_score,
            "source_group_id": source_group_id,
            "rollout_slot": rollout_slot,
            "step_index": step_index,
            "thinking_mode": target.get("thinking_mode", "cached"),
        }
        parsed_items.append(item)
        grouped[(source_group_id, rollout_slot)].append(item)

    group_metrics = {
        group_key: _source_macro_metrics(group_items)
        for group_key, group_items in grouped.items()
    }

    scores = []
    for item in parsed_items:
        valid = item["pred"] in (0, 1) and item["label"] in (0, 1)
        step_correct = float(item["pred"] == item["label"]) if valid else float(invalid_reward)
        metrics = group_metrics[(item["source_group_id"], item["rollout_slot"])]
        overall = (
            source_macro_weight * metrics["source_macro_f1"]
            + step_weight * step_correct
            + format_weight * item["format"]
        ) / total_weight
        scores.append(
            {
                "overall": float(overall),
                "accuracy": step_correct,
                "format": float(item["format"]),
                "pred": float(item["pred"]),
                "label": float(item["label"]),
                "is_negative": float(item["label"] == 0),
                "is_positive": float(item["label"] == 1),
                "pred_negative": float(item["pred"] == 0),
                "pred_positive": float(item["pred"] == 1),
                "negative_accuracy": float(item["pred"] == item["label"] == 0),
                "positive_accuracy": float(item["pred"] == item["label"] == 1),
                "valid": float(valid),
                "rollout_slot": float(item["rollout_slot"]),
                "thinking_mode_cached": float(item["thinking_mode"] == "cached"),
                "thinking_mode_dropout": float(item["thinking_mode"] == "dropout"),
                "thinking_mode_no_thinking": float(item["thinking_mode"] == "no_thinking"),
                **metrics,
            }
        )
    return scores
