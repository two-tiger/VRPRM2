#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import Counter, defaultdict
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


def labels_for_eval(row: dict[str, Any]) -> list[int]:
    labels = row.get("response", {}).get("process_correctness", [])
    return [int(label) for label in labels if int(label) != 0]


def has_incorrect(row: dict[str, Any]) -> bool:
    return any(label == -1 for label in labels_for_eval(row))


def difficulty_key(row: dict[str, Any]) -> tuple[int, int, int]:
    labels = labels_for_eval(row)
    incorrect = sum(label == -1 for label in labels)
    correct = sum(label == 1 for label in labels)
    mixed = int(incorrect > 0 and correct > 0)
    return mixed, incorrect, len(labels)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_counts = Counter()
    for row in rows:
        by_source[row.get("data_source", "unknown")].append(row)
        label_counts.update(labels_for_eval(row))

    source_stats = {}
    for source, source_rows in sorted(by_source.items()):
        source_label_counts = Counter()
        for row in source_rows:
            source_label_counts.update(labels_for_eval(row))
        negative_sources = sum(has_incorrect(row) for row in source_rows)
        source_stats[source] = {
            "samples": len(source_rows),
            "negative_source_samples": negative_sources,
            "negative_source_ratio": negative_sources / len(source_rows) if source_rows else 0.0,
            "correct_steps": source_label_counts.get(1, 0),
            "incorrect_steps": source_label_counts.get(-1, 0),
            "incorrect_step_ratio": (
                source_label_counts.get(-1, 0) / max(source_label_counts.get(1, 0) + source_label_counts.get(-1, 0), 1)
            ),
        }

    negative_sources = sum(has_incorrect(row) for row in rows)
    return {
        "samples": len(rows),
        "sources": sorted(by_source),
        "negative_source_samples": negative_sources,
        "negative_source_ratio": negative_sources / len(rows) if rows else 0.0,
        "correct_steps": label_counts.get(1, 0),
        "incorrect_steps": label_counts.get(-1, 0),
        "incorrect_step_ratio": label_counts.get(-1, 0) / max(label_counts.get(1, 0) + label_counts.get(-1, 0), 1),
        "by_source": source_stats,
    }


def target_negative_count(rows: list[dict[str, Any]], quota: int, target_ratio: float) -> int:
    negative_count = sum(has_incorrect(row) for row in rows)
    positive_count = len(rows) - negative_count
    if target_ratio < 0:
        target_ratio = negative_count / len(rows) if rows else 0.0
    target = round(quota * target_ratio)
    min_negative_needed = max(0, quota - positive_count)
    max_negative_allowed = min(negative_count, quota)
    return max(min_negative_needed, min(target, max_negative_allowed))


def sample_source_rows(
    rows: list[dict[str, Any]],
    quota: int,
    target_negative_source_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if quota <= 0 or quota >= len(rows):
        selected = rows[:]
        rng.shuffle(selected)
        return selected

    negative_rows = [row for row in rows if has_incorrect(row)]
    positive_rows = [row for row in rows if not has_incorrect(row)]
    rng.shuffle(negative_rows)
    rng.shuffle(positive_rows)

    # Keep harder negative cases near the front so the mini split is sensitive to incorrect-F1 regressions.
    negative_rows.sort(key=difficulty_key, reverse=True)
    positive_rows.sort(key=lambda row: (len(labels_for_eval(row)), rng.random()), reverse=True)

    target_neg = target_negative_count(rows, quota, target_negative_source_ratio)
    selected = negative_rows[:target_neg]
    selected.extend(positive_rows[: max(0, quota - len(selected))])
    if len(selected) < quota:
        selected.extend(negative_rows[target_neg : target_neg + quota - len(selected)])
    rng.shuffle(selected)
    return selected[:quota]


def make_images_link(source_dir: Path, output_dir: Path) -> None:
    src = (source_dir / "images").resolve()
    dst = output_dir / "images"
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        # Fall back to leaving images unresolved only if the original source lacks images.
        if not src.exists():
            raise FileNotFoundError(f"Image directory not found: {src}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-balanced mini VisualProcessBench dev split.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "VisualProcessBench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "VisualProcessBench_mini_dev",
    )
    parser.add_argument("--samples-per-source", type=int, default=60)
    parser.add_argument(
        "--target-negative-source-ratio",
        type=float,
        default=-1.0,
        help="Negative-source ratio per source. Negative value preserves each source's original ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.benchmark_dir / "test.jsonl")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row.get("data_source", "unknown")].append(row)

    rng = random.Random(args.seed)
    selected = []
    for source in sorted(by_source):
        selected.extend(
            sample_source_rows(
                by_source[source],
                quota=args.samples_per_source,
                target_negative_source_ratio=args.target_negative_source_ratio,
                rng=rng,
            )
        )

    rng.shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_images_link(args.benchmark_dir, args.output_dir)
    write_jsonl(args.output_dir / "test.jsonl", selected)
    metadata = {
        "seed": args.seed,
        "benchmark_dir": str(args.benchmark_dir),
        "output_dir": str(args.output_dir),
        "samples_per_source": args.samples_per_source,
        "target_negative_source_ratio": args.target_negative_source_ratio,
        "full": summarize(rows),
        "mini": summarize(selected),
        "note": "Mini VPB dev is source-balanced and intended only for checkpoint selection.",
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
