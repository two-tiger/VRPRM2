#!/usr/bin/env python3
import argparse
import json
import random
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


def normalize_prompt(item: dict[str, Any]) -> str:
    prompt = item.get("question_orig") or item.get("question") or ""
    prompt = str(prompt).strip()
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt


def normalize_image_path(image: Any, data_root: Path) -> str | None:
    image_path = str(image or "").strip()
    if not image_path:
        return None

    if Path(image_path).is_absolute():
        return image_path if Path(image_path).is_file() else None

    dataset_name = data_root.name
    parts = image_path.split("/")
    candidates = [image_path]

    if parts and parts[0] == dataset_name and (len(parts) < 2 or parts[1] != "images"):
        candidates.insert(0, "/".join([parts[0], "images", *parts[1:]]))

    for candidate in candidates:
        if (data_root.parent / candidate).is_file():
            return candidate

    return None


def convert_item(item: dict[str, Any], data_root: Path) -> dict[str, Any] | None:
    image = item.get("image")
    steps = item.get("steps_with_score")
    image = normalize_image_path(image, data_root)
    if not image or not isinstance(steps, list) or not steps:
        return None

    ground_truth = {
        "answer": item.get("answer", ""),
        "reference_response": item.get("response", ""),
        "steps_with_score": steps,
        "num_mc_sequences": item.get("num_mc_sequences"),
    }
    return {
        "prompt": normalize_prompt(item),
        "images": [image],
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisualPRM400K raw annotations to EasyR1 jsonl splits.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-glob", default="*.jsonl")
    parser.add_argument("--val-ratio", default=0.01, type=float)
    parser.add_argument("--max-val-samples", default=2000, type=int)
    parser.add_argument("--max-train-samples", default=0, type=int, help="Max train samples after shuffling. 0 means all.")
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
            row = convert_item(item, args.data_root)
            if row is not None:
                rows.append(row)
                if args.limit and len(rows) >= args.limit:
                    break
            else:
                skipped += 1
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise RuntimeError(f"No process-reward samples found under {annotation_dir}")
    if len(rows) < 2:
        raise RuntimeError("At least two samples are required to create train/test splits.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = int(len(rows) * args.val_ratio)
    val_size = max(1, min(val_size, args.max_val_samples, len(rows) - 1))

    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    if args.max_train_samples > 0:
        train_rows = train_rows[:args.max_train_samples]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "test.jsonl", val_rows)
    metadata = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_val_samples": args.max_val_samples,
        "max_train_samples": args.max_train_samples,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "source_samples_after_filter": len(rows),
        "skipped": skipped,
    }
    (args.output_dir / ".visualprm_paths_v3").write_text("ok\n", encoding="utf-8")
    (args.output_dir / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(train_rows)} train samples: {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_rows)} test samples: {args.output_dir / 'test.jsonl'}")
    print(f"Skipped {skipped} samples without process scores or existing images.")


if __name__ == "__main__":
    main()
