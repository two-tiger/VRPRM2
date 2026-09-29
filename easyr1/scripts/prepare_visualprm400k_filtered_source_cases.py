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
    raw_images = image if isinstance(image, list) else [image]
    images = []
    for raw_image in raw_images:
        image_path = normalize_image_path(raw_image, data_root)
        if image_path:
            images.append(image_path)
    return images


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


def normalize_question(item: dict[str, Any], num_images: int) -> str:
    question = str(item.get("question_orig") or item.get("question") or "").strip()
    return canonicalize_question_image_tags(question, num_images).strip()


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


def numbered_steps(steps: list[dict[str, Any]]) -> list[str]:
    result = []
    for idx, step in enumerate(steps):
        step_text = sanitize_prompt_text(step.get("step", ""))
        if not step_text:
            step_text = f"Step {idx}"
        if step_text.strip().lower().startswith(f"step {idx}:"):
            result.append(step_text)
        else:
            result.append(f"Step {idx}: {step_text}")
    return result


def convert_item(
    item: dict[str, Any],
    data_root: Path,
    source_annotation: str,
    source_index: int,
    negative_threshold: float,
    positive_threshold: float,
    drop_fuzzy_steps: bool,
) -> tuple[dict[str, Any] | None, str]:
    images = normalize_image_paths(item.get("image"), data_root)
    steps_raw = item.get("steps_with_score")
    if not images or not isinstance(steps_raw, list) or not steps_raw:
        return None, "missing_image_or_steps"

    question = normalize_question(item, len(images))
    if not question:
        return None, "missing_question"

    steps = numbered_steps(steps_raw)
    scores = []
    labels = []
    for step in steps_raw:
        try:
            score = float(step.get("score"))
        except (TypeError, ValueError):
            return None, "bad_score"
        scores.append(score)
        labels.append(label_from_score(score, negative_threshold, positive_threshold))

    if drop_fuzzy_steps and any(label == -1 for label in labels):
        return None, "fuzzy_step"
    if not any(label in (0, 1) for label in labels):
        return None, "no_confident_step"

    negative_steps = sum(label == 0 for label in labels)
    positive_steps = sum(label == 1 for label in labels)
    source_id = (
        str(item.get("id") or item.get("sample_id") or item.get("source_sample_id") or source_index)
    )
    return {
        "id": f"{source_annotation}:{source_id}",
        "source_annotation": source_annotation,
        "source_sample_id": source_id,
        "images": images,
        "question": question,
        "answer": item.get("answer", ""),
        "steps": steps,
        "source_scores": scores,
        "step_labels": labels,
        "negative_steps": negative_steps,
        "positive_steps": positive_steps,
        "has_negative_step": negative_steps > 0,
        "all_positive_sample": negative_steps == 0 and positive_steps > 0,
        "negative_threshold": negative_threshold,
        "positive_threshold": positive_threshold,
        "num_mc_sequences": item.get("num_mc_sequences"),
    }, "ok"


def step_counts(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("negative_steps", 0)), int(row.get("positive_steps", 0))


def source_is_negative(row: dict[str, Any]) -> bool:
    return step_counts(row)[0] > 0


def select_source_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_source_ratio: float,
    target_negative_step_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    negative_rows = [row for row in rows if source_is_negative(row)]
    positive_rows = [row for row in rows if not source_is_negative(row)]
    rng.shuffle(negative_rows)
    rng.shuffle(positive_rows)

    target_size = max_samples if max_samples > 0 else len(rows)
    target_size = min(target_size, len(rows))
    selected: list[dict[str, Any]] = []
    neg_idx = pos_idx = 0
    neg_steps = pos_steps = neg_sources = 0

    def cost_after(row: dict[str, Any]) -> float:
        row_neg, row_pos = step_counts(row)
        next_sources = len(selected) + 1
        next_neg_sources = neg_sources + int(row_neg > 0)
        next_neg_steps = neg_steps + row_neg
        next_pos_steps = pos_steps + row_pos
        step_total = next_neg_steps + next_pos_steps
        source_ratio = next_neg_sources / next_sources
        step_ratio = next_neg_steps / step_total if step_total else 0.0
        return (
            abs(source_ratio - target_negative_source_ratio)
            + abs(step_ratio - target_negative_step_ratio)
        )

    while len(selected) < target_size:
        candidates = []
        if neg_idx < len(negative_rows):
            candidates.append(("neg", negative_rows[neg_idx], cost_after(negative_rows[neg_idx])))
        if pos_idx < len(positive_rows):
            candidates.append(("pos", positive_rows[pos_idx], cost_after(positive_rows[pos_idx])))
        if not candidates:
            break
        source, row, _ = min(candidates, key=lambda item: item[2])
        if source == "neg":
            neg_idx += 1
        else:
            pos_idx += 1
        row_neg, row_pos = step_counts(row)
        selected.append(row)
        neg_steps += row_neg
        pos_steps += row_pos
        neg_sources += int(row_neg > 0)

    rng.shuffle(selected)
    stats = {
        "available_negative_source_samples": len(negative_rows),
        "available_positive_source_samples": len(positive_rows),
        "selected_negative_source_samples": sum(source_is_negative(row) for row in selected),
        "selected_positive_source_samples": sum(not source_is_negative(row) for row in selected),
        "target_negative_source_ratio": target_negative_source_ratio,
        "selected_negative_source_ratio": (
            sum(source_is_negative(row) for row in selected) / len(selected) if selected else 0.0
        ),
        "target_negative_step_ratio": target_negative_step_ratio,
        "selected_negative_steps": sum(step_counts(row)[0] for row in selected),
        "selected_positive_steps": sum(step_counts(row)[1] for row in selected),
        "selected_negative_step_ratio": (
            sum(step_counts(row)[0] for row in selected)
            / max(sum(sum(step_counts(row)) for row in selected), 1)
        ),
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

    all_positive_rows = [row for row in rows if not source_is_negative(row)]
    other_rows = [row for row in rows if source_is_negative(row)]
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
            all_pos_count = sum(not source_is_negative(row) for row in batch)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg_steps = sum(step_counts(row)[0] for row in rows)
    pos_steps = sum(step_counts(row)[1] for row in rows)
    step_counts_all = [sum(step_counts(row)) for row in rows]
    negative_sources = sum(source_is_negative(row) for row in rows)
    return {
        "samples": len(rows),
        "negative_source_samples": negative_sources,
        "positive_source_samples": len(rows) - negative_sources,
        "negative_source_ratio": negative_sources / len(rows) if rows else 0.0,
        "confident_steps": neg_steps + pos_steps,
        "negative_steps": neg_steps,
        "positive_steps": pos_steps,
        "negative_step_ratio": neg_steps / (neg_steps + pos_steps) if neg_steps + pos_steps else 0.0,
        "min_steps": min(step_counts_all) if step_counts_all else 0,
        "max_steps": max(step_counts_all) if step_counts_all else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-level VisualPRM400K cases with fuzzy-step cases removed."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-glob", default="*.jsonl")
    parser.add_argument("--negative-threshold", default=0.125, type=float)
    parser.add_argument("--positive-threshold", default=0.75, type=float)
    parser.add_argument("--drop-fuzzy-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-negative-source-ratio", default=0.50, type=float)
    parser.add_argument("--target-negative-step-ratio", default=0.35, type=float)
    parser.add_argument("--val-ratio", default=0.01, type=float)
    parser.add_argument("--max-val-source-samples", default=2000, type=int)
    parser.add_argument("--max-train-source-samples", default=50000, type=int)
    parser.add_argument("--max-all-positive-per-batch-ratio", default=0.50, type=float)
    parser.add_argument("--batch-size-for-ordering", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Optional source-sample debug limit. 0 means no limit.")
    args = parser.parse_args()

    annotation_dir = args.data_root / "annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    rows = []
    skipped: dict[str, int] = {}
    source_seen = 0
    for path in sorted(annotation_dir.glob(args.annotation_glob)):
        for source_index, item in enumerate(iter_jsonl(path)):
            row, reason = convert_item(
                item,
                args.data_root,
                source_annotation=path.name,
                source_index=source_index,
                negative_threshold=args.negative_threshold,
                positive_threshold=args.positive_threshold,
                drop_fuzzy_steps=args.drop_fuzzy_steps,
            )
            source_seen += 1
            if row is None:
                skipped[reason] = skipped.get(reason, 0) + 1
            else:
                rows.append(row)
            if args.limit and source_seen >= args.limit:
                break
        if args.limit and source_seen >= args.limit:
            break

    if len(rows) < 2:
        raise RuntimeError(f"Need at least two valid source rows under {annotation_dir}, got {len(rows)}.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = int(len(rows) * args.val_ratio)
    val_size = max(1, min(val_size, args.max_val_source_samples, len(rows) - 1))
    val_rows, val_selection = select_source_rows(
        rows,
        max_samples=val_size,
        target_negative_source_ratio=args.target_negative_source_ratio,
        target_negative_step_ratio=args.target_negative_step_ratio,
        rng=rng,
    )
    val_ids = {id(row) for row in val_rows}
    train_pool = [row for row in rows if id(row) not in val_ids]
    train_rows, train_selection = select_source_rows(
        train_pool,
        max_samples=args.max_train_source_samples,
        target_negative_source_ratio=args.target_negative_source_ratio,
        target_negative_step_ratio=args.target_negative_step_ratio,
        rng=rng,
    )
    train_rows, batch_ordering = order_rows_for_batch_all_positive_cap(
        train_rows,
        batch_size=args.batch_size_for_ordering,
        max_all_positive_ratio=args.max_all_positive_per_batch_ratio,
        rng=rng,
    )

    write_jsonl(args.output_dir / "train_source.jsonl", train_rows)
    write_jsonl(args.output_dir / "test_source.jsonl", val_rows)

    metadata = {
        "seed": args.seed,
        "negative_threshold": args.negative_threshold,
        "positive_threshold": args.positive_threshold,
        "drop_fuzzy_steps": args.drop_fuzzy_steps,
        "target_negative_source_ratio": args.target_negative_source_ratio,
        "target_negative_step_ratio": args.target_negative_step_ratio,
        "max_train_source_samples": args.max_train_source_samples,
        "max_val_source_samples": args.max_val_source_samples,
        "val_ratio": args.val_ratio,
        "source_samples_seen": source_seen,
        "source_samples_after_filter": len(rows),
        "skipped": skipped,
        "selection": train_selection,
        "val_selection": val_selection,
        "batch_ordering": batch_ordering,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "note": "Rows are source-level cases. Cases with any fuzzy/middle step are removed when drop_fuzzy_steps is true.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".visualprm_filtered_source_cases_v1").write_text("ok\n", encoding="utf-8")
    (args.output_dir / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(train_rows)} train source cases: {args.output_dir / 'train_source.jsonl'}")
    print(f"Wrote {len(val_rows)} val source cases: {args.output_dir / 'test_source.jsonl'}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
