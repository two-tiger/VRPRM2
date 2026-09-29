#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                yield item


def normalize_image_path(image: Any, data_root: Path) -> str | None:
    image_path = str(image or "").strip()
    if not image_path:
        return None

    if Path(image_path).is_absolute():
        return image_path if Path(image_path).is_file() else None

    dataset_name = data_root.name
    parts = image_path.split("/")
    candidates: list[tuple[Path, str]] = [(data_root.parent / image_path, image_path)]
    candidates.append((data_root / image_path, f"{dataset_name}/{image_path}"))

    if parts and parts[0] == dataset_name and (len(parts) < 2 or parts[1] != "images"):
        fixed = "/".join([parts[0], "images", *parts[1:]])
        candidates.insert(0, (data_root.parent / fixed, fixed))

    for full_path, returned_path in candidates:
        if full_path.is_file():
            return returned_path

    return None


def normalize_image_paths(image: Any, data_root: Path) -> list[str]:
    if isinstance(image, list):
        raw_images = image
    else:
        raw_images = [image]

    images = []
    for raw_image in raw_images:
        image_path = normalize_image_path(raw_image, data_root)
        if image_path:
            images.append(image_path)
    return images


def normalize_question(item: dict[str, Any], num_images: int) -> str:
    question = str(item.get("question_orig") or item.get("question") or "").strip()
    question = canonicalize_question_image_tags(question, num_images)
    return question.strip()


def sanitize_prompt_text(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"<\s*/?\s*image\s*>", " image ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*video\s*>", " video ", text, flags=re.IGNORECASE)
    return text.strip()


def canonicalize_question_image_tags(text: Any, num_images: int) -> str:
    text = str(text or "")
    image_tag_pattern = re.compile(r"<\s*image\s*>", flags=re.IGNORECASE)
    image_tag_count = len(image_tag_pattern.findall(text))

    if image_tag_count == 0:
        prefix = " ".join(["<image>"] * num_images)
        return f"{prefix}\n{text}" if prefix else text

    kept = 0

    def replace_image_tag(_: re.Match) -> str:
        nonlocal kept
        if kept < num_images:
            kept += 1
            return "<image>"
        return " image "

    text = image_tag_pattern.sub(replace_image_tag, text)
    if kept < num_images:
        prefix = " ".join(["<image>"] * (num_images - kept))
        text = f"{prefix}\n{text}"
    return text


def label_from_score(score: Any, negative_threshold: float, positive_threshold: float) -> int:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return -1
    if value <= negative_threshold:
        return 0
    if value >= positive_threshold:
        return 1
    return -1


def build_prompt(question: str, answer: Any, steps: list[dict[str, Any]]) -> str:
    numbered_steps = []
    for idx, step in enumerate(steps):
        step_text = sanitize_prompt_text(step.get("step", ""))
        if not step_text:
            continue
        numbered_steps.append(f"Step {idx}: {step_text}")

    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Reference Answer]\n"
        f"{str(answer).strip()}\n\n"
        "[Candidate Solution Steps]\n"
        f"{chr(10).join(numbered_steps)}\n\n"
        "[Task]\n"
        "Use the global-thinking stepwise format: first output one <think>...</think> block, "
        "then output one <answer>...</answer> block with exactly one 0/1 score for every candidate step."
    )


def convert_item(
    item: dict[str, Any],
    data_root: Path,
    negative_threshold: float,
    positive_threshold: float,
    min_confident_steps: int,
) -> dict[str, Any] | None:
    images = normalize_image_paths(item.get("image"), data_root)
    steps = item.get("steps_with_score")
    if not images or not isinstance(steps, list) or not steps:
        return None

    question = normalize_question(item, len(images))
    candidate_response = str(item.get("response") or "").strip()
    if not question or not candidate_response:
        return None

    labels = [
        label_from_score(step.get("score"), negative_threshold, positive_threshold)
        for step in steps
    ]
    if sum(label in (0, 1) for label in labels) < min_confident_steps:
        return None

    negative_steps = sum(label == 0 for label in labels)
    positive_steps = sum(label == 1 for label in labels)

    step_scores = []
    for idx, step in enumerate(steps):
        try:
            score = float(step.get("score"))
        except (TypeError, ValueError):
            score = None
        step_scores.append(
            {
                "index": idx,
                "step": sanitize_prompt_text(step.get("step", "")),
                "score": score,
                "label": labels[idx],
                "num_mc_correct": step.get("num_mc_correct"),
                "num_mc_total": step.get("num_mc_total"),
            }
        )

    ground_truth = {
        "answer": item.get("answer", ""),
        "candidate_response": candidate_response,
        "steps_with_score": step_scores,
        "step_labels": labels,
        "negative_steps": negative_steps,
        "positive_steps": positive_steps,
        "all_positive_sample": negative_steps == 0 and positive_steps > 0,
        "negative_threshold": negative_threshold,
        "positive_threshold": positive_threshold,
        "num_mc_sequences": item.get("num_mc_sequences"),
    }
    return {
        "prompt": build_prompt(question, item.get("answer", ""), steps),
        "images": images,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
    }


def step_counts(row: dict[str, Any]) -> tuple[int, int]:
    target = json.loads(row["ground_truth"])
    labels = target["step_labels"]
    return sum(label == 0 for label in labels), sum(label == 1 for label in labels)


def is_all_positive(row: dict[str, Any]) -> bool:
    neg, pos = step_counts(row)
    return neg == 0 and pos > 0


def ratio_after(neg_steps: int, pos_steps: int, row: dict[str, Any]) -> float:
    row_neg, row_pos = step_counts(row)
    total = neg_steps + pos_steps + row_neg + row_pos
    return (neg_steps + row_neg) / total if total else 0.0


def select_train_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    negative_rows = []
    all_positive_rows = []
    other_rows = []
    for row in rows:
        neg, pos = step_counts(row)
        if neg > 0:
            negative_rows.append(row)
        elif pos > 0:
            all_positive_rows.append(row)
        else:
            other_rows.append(row)

    rng.shuffle(negative_rows)
    rng.shuffle(all_positive_rows)
    rng.shuffle(other_rows)

    target_size = max_samples if max_samples > 0 else len(rows)
    target_size = min(target_size, len(rows))
    selected = []
    neg_steps = 0
    pos_steps = 0
    neg_idx = pos_idx = other_idx = 0

    while len(selected) < target_size:
        candidates = []
        if neg_idx < len(negative_rows):
            row = negative_rows[neg_idx]
            candidates.append(("negative", row, abs(ratio_after(neg_steps, pos_steps, row) - target_negative_ratio)))
        if pos_idx < len(all_positive_rows):
            row = all_positive_rows[pos_idx]
            candidates.append(("all_positive", row, abs(ratio_after(neg_steps, pos_steps, row) - target_negative_ratio)))
        if other_idx < len(other_rows):
            row = other_rows[other_idx]
            candidates.append(("other", row, abs(ratio_after(neg_steps, pos_steps, row) - target_negative_ratio)))
        if not candidates:
            break

        source, row, _ = min(candidates, key=lambda item: item[2])
        if source == "negative":
            neg_idx += 1
        elif source == "all_positive":
            pos_idx += 1
        else:
            other_idx += 1

        row_neg, row_pos = step_counts(row)
        selected.append(row)
        neg_steps += row_neg
        pos_steps += row_pos

    stats = {
        "available_negative_samples": len(negative_rows),
        "available_all_positive_samples": len(all_positive_rows),
        "available_other_samples": len(other_rows),
        "selected_negative_samples": sum(1 for row in selected if step_counts(row)[0] > 0),
        "selected_all_positive_samples": sum(1 for row in selected if is_all_positive(row)),
        "target_negative_step_ratio": target_negative_ratio,
        "selected_negative_steps": neg_steps,
        "selected_positive_steps": pos_steps,
        "selected_negative_ratio": neg_steps / (neg_steps + pos_steps) if neg_steps + pos_steps else 0.0,
    }
    return selected, stats


def order_rows_for_batch_all_positive_cap(
    rows: list[dict[str, Any]],
    batch_size: int,
    max_all_positive_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size <= 0:
        rng.shuffle(rows)
        return rows, {"batch_size": batch_size, "max_all_positive_ratio": max_all_positive_ratio}

    all_positive_rows = [row for row in rows if is_all_positive(row)]
    other_rows = [row for row in rows if not is_all_positive(row)]
    rng.shuffle(all_positive_rows)
    rng.shuffle(other_rows)

    max_all_positive = math.floor(batch_size * max_all_positive_ratio)
    max_all_positive = max(0, min(batch_size, max_all_positive))
    ordered = []
    batch_stats = []
    pos_idx = other_idx = 0
    while pos_idx < len(all_positive_rows) or other_idx < len(other_rows):
        batch = []
        take_pos = min(max_all_positive, len(all_positive_rows) - pos_idx)
        batch.extend(all_positive_rows[pos_idx : pos_idx + take_pos])
        pos_idx += take_pos

        take_other = min(batch_size - len(batch), len(other_rows) - other_idx)
        batch.extend(other_rows[other_idx : other_idx + take_other])
        other_idx += take_other

        if len(batch) < batch_size and pos_idx < len(all_positive_rows):
            take_pos = min(batch_size - len(batch), len(all_positive_rows) - pos_idx)
            batch.extend(all_positive_rows[pos_idx : pos_idx + take_pos])
            pos_idx += take_pos
        if len(batch) < batch_size and other_idx < len(other_rows):
            take_other = min(batch_size - len(batch), len(other_rows) - other_idx)
            batch.extend(other_rows[other_idx : other_idx + take_other])
            other_idx += take_other

        rng.shuffle(batch)
        if batch:
            all_pos_count = sum(1 for row in batch if is_all_positive(row))
            batch_stats.append(
                {
                    "size": len(batch),
                    "all_positive_samples": all_pos_count,
                    "all_positive_ratio": all_pos_count / len(batch),
                }
            )
            ordered.extend(batch)

    overflow_batches = [
        item for item in batch_stats
        if item["size"] == batch_size and item["all_positive_ratio"] > max_all_positive_ratio
    ]
    stats = {
        "batch_size": batch_size,
        "max_all_positive_ratio": max_all_positive_ratio,
        "max_all_positive_per_full_batch": max_all_positive,
        "num_batches": len(batch_stats),
        "max_observed_all_positive_ratio": max((item["all_positive_ratio"] for item in batch_stats), default=0.0),
        "overflow_full_batches": len(overflow_batches),
    }
    return ordered, stats
    return {
        "prompt": build_prompt(question, item.get("answer", ""), steps),
        "images": [image],
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confident = positive = negative = ignored = 0
    step_counts = []
    for row in rows:
        target = json.loads(row["ground_truth"])
        labels = target["step_labels"]
        step_counts.append(len(labels))
        for label in labels:
            if label == 1:
                confident += 1
                positive += 1
            elif label == 0:
                confident += 1
                negative += 1
            else:
                ignored += 1
    return {
        "samples": len(rows),
        "steps": sum(step_counts),
        "confident_steps": confident,
        "positive_steps": positive,
        "negative_steps": negative,
        "ignored_steps": ignored,
        "negative_ratio": negative / confident if confident else 0.0,
        "min_steps": min(step_counts) if step_counts else 0,
        "max_steps": max(step_counts) if step_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisualPRM400K raw annotations to EasyR1 global-thinking stepwise PRM RL data."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-glob", default="*.jsonl")
    parser.add_argument("--negative-threshold", default=0.125, type=float)
    parser.add_argument("--positive-threshold", default=0.75, type=float)
    parser.add_argument("--min-confident-steps", default=1, type=int)
    parser.add_argument("--val-ratio", default=0.01, type=float)
    parser.add_argument("--max-val-samples", default=2000, type=int)
    parser.add_argument("--max-train-samples", default=0, type=int, help="Max train samples after shuffling. 0 means all.")
    parser.add_argument("--target-negative-step-ratio", default=0.35, type=float)
    parser.add_argument("--max-all-positive-per-batch-ratio", default=0.5, type=float)
    parser.add_argument("--batch-size-for-ordering", default=16, type=int)
    parser.add_argument("--balance-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Optional debug limit. 0 means no limit.")
    args = parser.parse_args()

    annotation_dir = args.data_root / "annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    rows = []
    skipped = 0
    for path in sorted(annotation_dir.glob(args.annotation_glob)):
        for item in iter_jsonl(path):
            row = convert_item(
                item,
                args.data_root,
                negative_threshold=args.negative_threshold,
                positive_threshold=args.positive_threshold,
                min_confident_steps=args.min_confident_steps,
            )
            if row is None:
                skipped += 1
            else:
                rows.append(row)
                if args.limit and len(rows) >= args.limit:
                    break
        if args.limit and len(rows) >= args.limit:
            break

    if len(rows) < 2:
        raise RuntimeError(f"Need at least two valid samples under {annotation_dir}, got {len(rows)}.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = int(len(rows) * args.val_ratio)
    val_size = max(1, min(val_size, args.max_val_samples, len(rows) - 1))
    if args.balance_val:
        val_rows, val_selection_stats = select_train_rows(
            rows,
            max_samples=val_size,
            target_negative_ratio=args.target_negative_step_ratio,
            rng=rng,
        )
        val_ids = {id(row) for row in val_rows}
        train_pool = [row for row in rows if id(row) not in val_ids]
    else:
        val_rows = rows[:val_size]
        train_pool = rows[val_size:]
        val_selection_stats = None
    train_rows, selection_stats = select_train_rows(
        train_pool,
        max_samples=args.max_train_samples,
        target_negative_ratio=args.target_negative_step_ratio,
        rng=rng,
    )
    train_rows, batch_order_stats = order_rows_for_batch_all_positive_cap(
        train_rows,
        batch_size=args.batch_size_for_ordering,
        max_all_positive_ratio=args.max_all_positive_per_batch_ratio,
        rng=rng,
    )

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "test.jsonl", val_rows)

    metadata = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_val_samples": args.max_val_samples,
        "max_train_samples": args.max_train_samples,
        "target_negative_step_ratio": args.target_negative_step_ratio,
        "max_all_positive_per_batch_ratio": args.max_all_positive_per_batch_ratio,
        "batch_size_for_ordering": args.batch_size_for_ordering,
        "balance_val": args.balance_val,
        "negative_threshold": args.negative_threshold,
        "positive_threshold": args.positive_threshold,
        "min_confident_steps": args.min_confident_steps,
        "source_samples_after_filter": len(rows),
        "skipped": skipped,
        "val_selection": val_selection_stats,
        "selection": selection_stats,
        "batch_ordering": batch_order_stats,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
    }
    (args.output_dir / ".visualprm_global_stepwise_paths_v1").write_text("ok\n", encoding="utf-8")
    (args.output_dir / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(train_rows)} train samples: {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_rows)} test samples: {args.output_dir / 'test.jsonl'}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
