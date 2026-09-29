# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

import json
import re
from difflib import SequenceMatcher
from typing import Any


REWARD_NAME = "visualprm_process"
REWARD_TYPE = "batch"


_TAG_PATTERN = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL,
)


def _normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _extract_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"final\s+answer\s*:\s*(.*)$", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = text.strip().strip(".。").strip()
    return text


def _parse_response(response: str) -> tuple[float, str, str]:
    match = _TAG_PATTERN.fullmatch(response or "")
    if not match:
        return 0.0, response or "", ""

    think = match.group("think").strip()
    answer = match.group("answer").strip()
    if not think or not answer:
        return 0.0, think, answer

    answer_text = _extract_answer(answer)
    format_score = 1.0 if answer_text else 0.0
    return format_score, think, answer_text


def _split_steps(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = re.split(r"\n\s*\n+", text)
    if len(chunks) == 1:
        chunks = re.split(r"\n(?=\s*(?:[-*]|\d+[\).\:]|Step\s+\d+))", text, flags=re.IGNORECASE)

    steps = []
    for chunk in chunks:
        chunk = re.sub(r"^\s*(?:[-*]|\d+[\).\:]|Step\s+\d+[\).\:]?)\s*", "", chunk, flags=re.IGNORECASE)
        chunk = chunk.strip()
        if chunk:
            steps.append(chunk)

    return steps or [text]


def _similarity(left: str, right: str) -> float:
    left = _normalize_text(left)
    right = _normalize_text(right)
    if not left or not right:
        return 0.0

    left_tokens = set(re.findall(r"\w+", left))
    right_tokens = set(re.findall(r"\w+", right))
    token_f1 = 0.0
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens)
        precision = overlap / len(left_tokens)
        recall = overlap / len(right_tokens)
        if precision + recall:
            token_f1 = 2 * precision * recall / (precision + recall)

    seq_ratio = SequenceMatcher(None, left, right).ratio()
    return max(token_f1, seq_ratio)


def _load_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth

    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8")

    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            return parsed if isinstance(parsed, dict) else {"answer": ground_truth}
        except json.JSONDecodeError:
            return {"answer": ground_truth}

    return {"answer": str(ground_truth)}


def _process_reward(generated_steps: list[str], reference_steps: list[dict[str, Any]], min_similarity: float) -> float:
    if not generated_steps or not reference_steps:
        return 0.0

    total_weight = 0.0
    total_score = 0.0
    for ref in reference_steps:
        ref_step = ref.get("step", "")
        ref_score = float(ref.get("score", 0.0))
        ref_score = max(0.0, min(1.0, ref_score))
        best_similarity = max((_similarity(gen_step, ref_step) for gen_step in generated_steps), default=0.0)
        if best_similarity < min_similarity:
            best_similarity = 0.0

        total_weight += 1.0
        total_score += ref_score * best_similarity

    return total_score / total_weight if total_weight else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.2,
    process_weight: float = 0.8,
    answer_weight: float = 0.0,
    min_similarity: float = 0.25,
) -> list[dict[str, float]]:
    scores = []
    total_weight = format_weight + process_weight + answer_weight
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive.")

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        format_score, think, predicted_answer = _parse_response(response)
        generated_steps = _split_steps(think)

        target = _load_ground_truth(reward_input.get("ground_truth", ""))
        reference_steps = target.get("steps_with_score") or target.get("steps") or []
        process_score = _process_reward(generated_steps, reference_steps, min_similarity=min_similarity)

        gold_answer = _extract_answer(str(target.get("answer", "")))
        answer_score = 1.0 if gold_answer and _normalize_text(predicted_answer) == _normalize_text(gold_answer) else 0.0

        overall = (
            format_weight * format_score + process_weight * process_score + answer_weight * answer_score
        ) / total_weight
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "process": process_score,
                "answer": answer_score,
            }
        )

    return scores
