#!/usr/bin/env python3
"""Build the 40K non-CoT SFT control dataset from the RL training split.

Revision-plan control (M3/M2): supervised fine-tuning directly on the SAME
40K fully confident non-CoT examples used for RL, with the SAME stepwise
single-token interface as VRPRM (no global-thinking turn). Comparing
"base -> 40K non-CoT SFT" against "base -> 5.5K CoT SFT -> RL(40K)" isolates
how much of the gain comes from the extra 40K data versus the RL algorithm.

The RL split (easyr1 prepare_visualprm400k_source_macro_rl.py) stores rows as
{prompt, images, ground_truth}; this script re-serializes them into the
ms-swift multi-turn format used by data_pipeline/build_multiturn.py --no-think,
so the control differs from VRPRM-SFT only in data content and scale.

Example:
    python data_pipeline/build_noncot_sft_from_rl.py \
        --input easyr1/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k/train.jsonl \
        --output rollout_outputs/visualprm400k_noncot40k_stepwise_sft.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a visual process reward model. Given image(s), a problem, a reference "
    "answer, and a complete candidate solution, judge each candidate solution step.\n\n"
    "Answer every step-scoring turn with exactly one token: 1 or 0. Do not reason out "
    "loud and do not output <think>...</think>."
)

SECTION_SPLIT = re.compile(
    r"\[Question\]\n(?P<question>.*?)\n\n\[Reference Answer\]\n(?P<answer>.*?)\n\n"
    r"\[Candidate Solution Steps\]\n(?P<steps>.*?)\n\n\[Task\]",
    re.DOTALL,
)


def parse_prompt(prompt: str) -> dict | None:
    match = SECTION_SPLIT.search(prompt)
    if not match:
        return None
    steps = [line for line in match.group("steps").splitlines() if line.strip()]
    return {
        "question": match.group("question").strip(),
        "answer": match.group("answer").strip(),
        "steps": steps,
    }


def build_step_turn(question: str, answer: str, steps: list[str], idx: int, previous: list[int]) -> str:
    previous_text = (
        "\n".join(f"Step {i}: \\boxed{{{s}}}" for i, s in enumerate(previous)) or "None"
    )
    return (
        f"[Question]\n{question}\n"
        f"[Reference Answer]\n{answer}\n"
        f"[Candidate Solution]\n{chr(10).join(steps)}\n"
        f"[Previous Step Judgments]\n{previous_text}\n"
        f"[Current Step]\n{steps[idx]}\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )


def convert(row: dict, step_loss_scale: float) -> dict | None:
    parsed = parse_prompt(row["prompt"])
    ground_truth = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
    labels = ground_truth.get("step_labels", [])
    if parsed is None or not parsed["steps"] or len(labels) != len(parsed["steps"]):
        return None

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    previous: list[int] = []
    trainable = 0
    for idx, (step_text, label) in enumerate(zip(parsed["steps"], labels)):
        if int(label) not in (0, 1):
            break
        messages.append({
            "role": "user",
            "content": build_step_turn(parsed["question"], parsed["answer"], parsed["steps"], idx, previous),
        })
        messages.append({"role": "assistant", "content": str(int(label)), "loss_scale": step_loss_scale})
        previous.append(int(label))
        trainable += 1
    if trainable == 0:
        return None

    return {
        "messages": messages,
        "images": row.get("images", []),
        "source_id": ground_truth.get("source_id"),
        "source_annotation": ground_truth.get("source_annotation"),
        "source_sample_id": ground_truth.get("source_sample_id"),
        "task_type": "noncot_stepwise_multiturn",
        "num_steps": trainable,
        "num_trainable_steps": trainable,
        "num_ignored_steps": 0,
        "negative_trainable_steps": sum(1 for m in messages if m.get("role") == "assistant" and m["content"] == "0"),
        "positive_trainable_steps": sum(1 for m in messages if m.get("role") == "assistant" and m["content"] == "1"),
        "step_loss_scale": step_loss_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="RL train.jsonl (prompt/images/ground_truth rows).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-loss-scale", type=float, default=2.0,
                        help="Assistant loss scale for step turns. Default 2.0 as in the paper.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap.")
    args = parser.parse_args()

    rows, skipped = [], 0
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if args.limit and len(rows) >= args.limit:
                break
            converted = convert(json.loads(line), args.step_loss_scale)
            if converted is None:
                skipped += 1
                continue
            rows.append(converted)

    neg = sum(r["negative_trainable_steps"] for r in rows)
    pos = sum(r["positive_trainable_steps"] for r in rows)
    stats = {
        "input": args.input,
        "output": args.output,
        "samples": len(rows),
        "skipped_rows": skipped,
        "trainable_steps": neg + pos,
        "negative_trainable_steps": neg,
        "positive_trainable_steps": pos,
        "negative_ratio": round(neg / (neg + pos), 4) if neg + pos else 0.0,
        "step_loss_scale": args.step_loss_scale,
        "task_type": "noncot_stepwise_multiturn",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
