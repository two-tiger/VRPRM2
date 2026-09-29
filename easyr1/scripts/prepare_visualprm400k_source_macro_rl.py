#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def step_counts(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("negative_steps", 0)), int(row.get("positive_steps", 0))


def row_category(row: dict[str, Any]) -> str:
    neg, pos = step_counts(row)
    total = neg + pos
    if neg > 0 and pos > 0:
        return "hard_mixed"
    if neg > 0:
        return "medium_negative"
    if total >= 4:
        return "medium_positive"
    return "easy_positive"


def numbered_steps(steps: list[str]) -> list[str]:
    result = []
    for idx, step in enumerate(steps):
        text = str(step).strip()
        if text.lower().startswith(f"step {idx}:"):
            result.append(text)
        else:
            result.append(f"Step {idx}: {text}")
    return result


def build_prompt(row: dict[str, Any]) -> str:
    return (
        "[Question]\n"
        f"{row['question']}\n\n"
        "[Reference Answer]\n"
        f"{str(row.get('answer', '')).strip()}\n\n"
        "[Candidate Solution Steps]\n"
        f"{chr(10).join(numbered_steps(row['steps']))}\n\n"
        "[Task]\n"
        "First write one concise <think>...</think> block with global reasoning for process scoring. "
        "Then write one <answer>...</answer> block with exactly one binary score for every candidate step.\n\n"
        "Use this exact answer format:\n"
        "<answer>\n"
        "Step 0: 1\n"
        "Step 1: 0\n"
        "...\n"
        "</answer>\n\n"
        "Scoring policy: output 1 only if the step is fully supported and correct in context; "
        "output 0 if it is unsupported, visually mistaken, logically invalid, computationally wrong, "
        "contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery."
    )


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = [int(x) if int(x) in (0, 1) else -1 for x in row["step_labels"]]
    ground_truth = {
        "source_id": row.get("id"),
        "source_annotation": row.get("source_annotation"),
        "source_sample_id": row.get("source_sample_id"),
        "answer": row.get("answer", ""),
        "source_scores": row.get("source_scores", []),
        "step_labels": labels,
        "negative_steps": sum(label == 0 for label in labels),
        "positive_steps": sum(label == 1 for label in labels),
        "category": row_category(row),
        "negative_threshold": row.get("negative_threshold"),
        "positive_threshold": row.get("positive_threshold"),
    }
    return {
        "prompt": build_prompt(row),
        "images": row.get("images", []),
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
        "_category": row_category(row),
    }


def ratio_after(
    selected_count: int,
    selected_negative_sources: int,
    selected_negative_steps: int,
    selected_positive_steps: int,
    row: dict[str, Any],
) -> tuple[float, float]:
    row_negative_steps, row_positive_steps = step_counts(row)
    neg_sources = selected_negative_sources + int(row_negative_steps > 0)
    total_sources = selected_count + 1
    neg_steps = selected_negative_steps + row_negative_steps
    pos_steps = selected_positive_steps + row_positive_steps
    return neg_sources / total_sources, neg_steps / max(neg_steps + pos_steps, 1)


def select_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_source_ratio: float,
    target_negative_step_ratio: float,
    hard_mixed_bonus: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "hard_mixed": [],
        "medium_negative": [],
        "medium_positive": [],
        "easy_positive": [],
    }
    for row in rows:
        pools[row_category(row)].append(row)
    for pool in pools.values():
        rng.shuffle(pool)

    selected: list[dict[str, Any]] = []
    cursor = {key: 0 for key in pools}
    target_size = min(max_samples if max_samples > 0 else len(rows), len(rows))
    selected_negative_sources = 0
    selected_negative_steps = 0
    selected_positive_steps = 0

    while len(selected) < target_size:
        candidates = []
        for category, pool in pools.items():
            idx = cursor[category]
            if idx >= len(pool):
                continue
            row = pool[idx]
            source_ratio, step_ratio = ratio_after(
                selected_count=len(selected),
                selected_negative_sources=selected_negative_sources,
                selected_negative_steps=selected_negative_steps,
                selected_positive_steps=selected_positive_steps,
                row=row,
            )
            cost = (
                abs(source_ratio - target_negative_source_ratio)
                + abs(step_ratio - target_negative_step_ratio)
            )
            if category == "hard_mixed":
                cost -= hard_mixed_bonus
            elif category in {"medium_negative", "medium_positive"}:
                cost -= hard_mixed_bonus * 0.25
            candidates.append((cost, category, row))
        if not candidates:
            break
        _, category, row = min(candidates, key=lambda item: item[0])
        cursor[category] += 1
        selected.append(row)
        row_negative_steps, row_positive_steps = step_counts(row)
        selected_negative_sources += int(row_negative_steps > 0)
        selected_negative_steps += row_negative_steps
        selected_positive_steps += row_positive_steps

    rng.shuffle(selected)
    stats = summarize_source_rows(selected)
    stats["available_categories"] = {key: len(value) for key, value in pools.items()}
    stats["selected_categories"] = dict(Counter(row_category(row) for row in selected))
    stats["target_negative_source_ratio"] = target_negative_source_ratio
    stats["target_negative_step_ratio"] = target_negative_step_ratio
    stats["hard_mixed_bonus"] = hard_mixed_bonus
    return selected, stats


def summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg_sources = sum(step_counts(row)[0] > 0 for row in rows)
    neg_steps = sum(step_counts(row)[0] for row in rows)
    pos_steps = sum(step_counts(row)[1] for row in rows)
    lengths = [step_counts(row)[0] + step_counts(row)[1] for row in rows]
    return {
        "samples": len(rows),
        "negative_source_samples": neg_sources,
        "positive_source_samples": len(rows) - neg_sources,
        "negative_source_ratio": neg_sources / len(rows) if rows else 0.0,
        "negative_steps": neg_steps,
        "positive_steps": pos_steps,
        "negative_step_ratio": neg_steps / (neg_steps + pos_steps) if neg_steps + pos_steps else 0.0,
        "min_steps": min(lengths) if lengths else 0,
        "max_steps": max(lengths) if lengths else 0,
    }


def strip_private(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def process_split(
    source_path: Path,
    output_path: Path,
    max_samples: int,
    target_negative_source_ratio: float,
    target_negative_step_ratio: float,
    hard_mixed_bonus: float,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_rows = load_jsonl(source_path)
    selected, stats = select_rows(
        source_rows,
        max_samples=max_samples,
        target_negative_source_ratio=target_negative_source_ratio,
        target_negative_step_ratio=target_negative_step_ratio,
        hard_mixed_bonus=hard_mixed_bonus,
        rng=rng,
    )
    converted = [convert_row(row) for row in selected]
    write_jsonl(output_path, strip_private(converted))
    stats["source_rows"] = len(source_rows)
    stats["converted_rows"] = len(converted)
    stats["converted_categories"] = dict(Counter(row["_category"] for row in converted))
    return stats, converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-level hard/medium VisualPRM RL data for source-macro reward."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train-samples", default=30000, type=int)
    parser.add_argument("--max-val-samples", default=2000, type=int)
    parser.add_argument("--target-negative-source-ratio", default=0.50, type=float)
    parser.add_argument("--target-negative-step-ratio", default=0.35, type=float)
    parser.add_argument("--hard-mixed-bonus", default=0.08, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_stats, _ = process_split(
        args.source_dir / "train_source.jsonl",
        args.output_dir / "train.jsonl",
        max_samples=args.max_train_samples,
        target_negative_source_ratio=args.target_negative_source_ratio,
        target_negative_step_ratio=args.target_negative_step_ratio,
        hard_mixed_bonus=args.hard_mixed_bonus,
        rng=rng,
    )
    val_stats, _ = process_split(
        args.source_dir / "test_source.jsonl",
        args.output_dir / "test.jsonl",
        max_samples=args.max_val_samples,
        target_negative_source_ratio=args.target_negative_source_ratio,
        target_negative_step_ratio=args.target_negative_step_ratio,
        hard_mixed_bonus=args.hard_mixed_bonus,
        rng=rng,
    )
    metadata = {
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "target_negative_source_ratio": args.target_negative_source_ratio,
        "target_negative_step_ratio": args.target_negative_step_ratio,
        "hard_mixed_bonus": args.hard_mixed_bonus,
        "train": train_stats,
        "val": val_stats,
        "note": "Source-level RL rows. Reward should parse one <think> block and one <answer> block with all step scores.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".visualprm_source_macro_paths_v1").write_text("ok\n", encoding="utf-8")
    write_json(args.output_dir / "dataset_meta.json", metadata)
    print(f"Wrote train: {args.output_dir / 'train.jsonl'}")
    print(f"Wrote val: {args.output_dir / 'test.jsonl'}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
