#!/usr/bin/env python3
import argparse
import json
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


def context_judgment_from_score(score: Any) -> int:
    try:
        return int(float(score) >= 0.5)
    except (TypeError, ValueError):
        return 0


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


def build_previous_text(previous_scores: list[int]) -> str:
    if not previous_scores:
        return "None"
    return "\n".join(f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_scores))


def build_prompt(
    question: str,
    answer: Any,
    steps: list[str],
    previous_scores: list[int],
    current_idx: int,
    response_format: str,
) -> str:
    if response_format == "think_answer":
        task = (
            "Think briefly about the visual evidence, reference answer, previous judgments, and the current step. "
            "Then output the current step score.\n\n"
            "Output exactly in this format:\n"
            "<think>\n"
            "Brief focused reasoning for the current step in context.\n"
            "</think>\n"
            "<answer>\n"
            "0 or 1\n"
            "</answer>"
        )
    else:
        task = "Is the current step correct in context? Output only one token: 1 or 0."

    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Reference Answer]\n"
        f"{str(answer).strip()}\n\n"
        "[Candidate Solution]\n"
        f"{chr(10).join(steps)}\n\n"
        "[Global Thinking]\n"
        "<think>Review the visual evidence, the question goal, the reference-answer constraint, "
        "dependencies between candidate steps, and the likely first consequential error. Use this "
        "global context together with previous step judgments before judging the current step.</think>\n\n"
        "[Previous Step Judgments]\n"
        f"{build_previous_text(previous_scores)}\n\n"
        "[Current Step]\n"
        f"{steps[current_idx]}\n\n"
        f"{task}"
    )


def convert_item(
    item: dict[str, Any],
    data_root: Path,
    negative_threshold: float,
    positive_threshold: float,
    response_format: str,
) -> list[dict[str, Any]]:
    images = normalize_image_paths(item.get("image"), data_root)
    steps_raw = item.get("steps_with_score")
    if not images or not isinstance(steps_raw, list) or not steps_raw:
        return []

    question = normalize_question(item, len(images))
    if not question:
        return []

    steps = numbered_steps(steps_raw)
    source_scores = []
    labels = []
    context_scores = []
    for step in steps_raw:
        score = step.get("score")
        try:
            source_scores.append(float(score))
        except (TypeError, ValueError):
            source_scores.append(None)
        labels.append(label_from_score(score, negative_threshold, positive_threshold))
        context_scores.append(context_judgment_from_score(score))

    rows = []
    previous_scores = []
    for idx, label in enumerate(labels):
        if label in (0, 1):
            ground_truth = {
                "label": label,
                "step_index": idx,
                "num_steps": len(steps),
                "source_scores": source_scores,
                "step_labels": labels,
                "negative_threshold": negative_threshold,
                "positive_threshold": positive_threshold,
                "answer": item.get("answer", ""),
                "num_mc_sequences": item.get("num_mc_sequences"),
            }
            rows.append(
                {
                    "prompt": build_prompt(
                        question=question,
                        answer=item.get("answer", ""),
                        steps=steps,
                        previous_scores=previous_scores,
                        current_idx=idx,
                        response_format=response_format,
                    ),
                    "images": images,
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
                    "label": label,
                    "source_annotation": item.get("source_annotation"),
                    "source_sample_id": item.get("id") or item.get("sample_id"),
                }
            )
        previous_scores.append(context_scores[idx])
    return rows


def select_step_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    negative = [row for row in rows if row["label"] == 0]
    positive = [row for row in rows if row["label"] == 1]
    rng.shuffle(negative)
    rng.shuffle(positive)

    target_size = max_samples if max_samples > 0 else len(rows)
    target_size = min(target_size, len(rows))
    target_negative = round(target_size * target_negative_ratio)
    target_positive = target_size - target_negative

    selected_negative = negative[: min(target_negative, len(negative))]
    selected_positive = positive[: min(target_positive, len(positive))]

    # Fill any shortage from the other class.
    shortage = target_size - len(selected_negative) - len(selected_positive)
    if shortage > 0 and len(selected_negative) < len(negative):
        selected_negative.extend(negative[len(selected_negative) : len(selected_negative) + shortage])
    shortage = target_size - len(selected_negative) - len(selected_positive)
    if shortage > 0 and len(selected_positive) < len(positive):
        selected_positive.extend(positive[len(selected_positive) : len(selected_positive) + shortage])

    selected = selected_negative + selected_positive
    rng.shuffle(selected)

    neg = sum(row["label"] == 0 for row in selected)
    pos = sum(row["label"] == 1 for row in selected)
    stats = {
        "available_negative_steps": len(negative),
        "available_positive_steps": len(positive),
        "selected_negative_steps": neg,
        "selected_positive_steps": pos,
        "target_negative_ratio": target_negative_ratio,
        "selected_negative_ratio": neg / (neg + pos) if neg + pos else 0.0,
    }
    return selected, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            public_row = {k: v for k, v in row.items() if k != "label"}
            f.write(json.dumps(public_row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg = sum(row["label"] == 0 for row in rows)
    pos = sum(row["label"] == 1 for row in rows)
    prompt_lengths = [len(row["prompt"]) for row in rows]
    return {
        "samples": len(rows),
        "negative_steps": neg,
        "positive_steps": pos,
        "negative_ratio": neg / (neg + pos) if neg + pos else 0.0,
        "min_prompt_chars": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_chars": max(prompt_lengths) if prompt_lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisualPRM400K raw annotations to EasyR1 step-level PRM RL data."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-glob", default="*.jsonl")
    parser.add_argument("--negative-threshold", default=0.125, type=float)
    parser.add_argument("--positive-threshold", default=0.75, type=float)
    parser.add_argument("--target-negative-step-ratio", default=0.20, type=float)
    parser.add_argument("--response-format", choices=("score_only", "think_answer"), default="score_only")
    parser.add_argument("--val-ratio", default=0.01, type=float)
    parser.add_argument("--max-val-samples", default=2000, type=int)
    parser.add_argument("--max-train-samples", default=50000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Optional source-sample debug limit. 0 means no limit.")
    args = parser.parse_args()

    annotation_dir = args.data_root / "annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    rows = []
    skipped = 0
    source_seen = 0
    for path in sorted(annotation_dir.glob(args.annotation_glob)):
        for item in iter_jsonl(path):
            item["source_annotation"] = path.name
            converted = convert_item(
                item,
                args.data_root,
                negative_threshold=args.negative_threshold,
                positive_threshold=args.positive_threshold,
                response_format=args.response_format,
            )
            source_seen += 1
            if converted:
                rows.extend(converted)
            else:
                skipped += 1
            if args.limit and source_seen >= args.limit:
                break
        if args.limit and source_seen >= args.limit:
            break

    if len(rows) < 2:
        raise RuntimeError(f"Need at least two valid step rows under {annotation_dir}, got {len(rows)}.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = int(len(rows) * args.val_ratio)
    val_size = max(1, min(val_size, args.max_val_samples, len(rows) - 1))

    val_rows, val_selection = select_step_rows(
        rows,
        max_samples=val_size,
        target_negative_ratio=args.target_negative_step_ratio,
        rng=rng,
    )
    val_ids = {id(row) for row in val_rows}
    train_pool = [row for row in rows if id(row) not in val_ids]
    train_rows, train_selection = select_step_rows(
        train_pool,
        max_samples=args.max_train_samples,
        target_negative_ratio=args.target_negative_step_ratio,
        rng=rng,
    )

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "test.jsonl", val_rows)

    metadata = {
        "seed": args.seed,
        "negative_threshold": args.negative_threshold,
        "positive_threshold": args.positive_threshold,
        "target_negative_step_ratio": args.target_negative_step_ratio,
        "response_format": args.response_format,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "val_ratio": args.val_ratio,
        "source_samples_seen": source_seen,
        "skipped_source_samples": skipped,
        "source_step_rows_after_filter": len(rows),
        "selection": train_selection,
        "val_selection": val_selection,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "note": "Each row is one current-step scoring prompt. response_format controls whether the response is score-only or <think>...</think><answer>0/1</answer>.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".visualprm_step_level_paths_v1").write_text("ok\n", encoding="utf-8")
    (args.output_dir / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(train_rows)} train samples: {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_rows)} test samples: {args.output_dir / 'test.jsonl'}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
