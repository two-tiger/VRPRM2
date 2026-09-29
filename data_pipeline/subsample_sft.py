#!/usr/bin/env python3
"""Subsample the multi-turn SFT dataset for the SFT data-scale ablation.

Keeps the source-polarity composition (negative/positive source samples) of
the full dataset and re-reports trainable step statistics after subsampling.
Used by the revision plan (SFT-scale curve: e.g. 25% / 50% / 100%).

Example:
    python data_pipeline/subsample_sft.py \
        --input rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35.json \
        --fraction 0.5 --seed 42 \
        --output rollout_outputs/visualprm400k_global_think_stepwise_multiturn_sft_neg35_50pct.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def stats(rows: list[dict]) -> dict:
    neg = sum(r.get("negative_trainable_steps", 0) for r in rows)
    pos = sum(r.get("positive_trainable_steps", 0) for r in rows)
    total = neg + pos
    return {
        "samples": len(rows),
        "trainable_steps": total,
        "negative_trainable": neg,
        "positive_trainable": pos,
        "negative_ratio": round(neg / total, 4) if total else 0.0,
        "by_source_polarity": dict(Counter(r.get("source_polarity", "?") for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Final multi-turn SFT dataset JSON.")
    parser.add_argument("--fraction", type=float, required=True, help="Fraction of conversations to keep in (0, 1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be in (0, 1]")

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    # Stratify by source polarity so the negative/positive pool composition
    # of the full dataset is preserved at every scale.
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("source_polarity", "unknown"), []).append(row)

    kept: list[dict] = []
    for polarity, bucket in sorted(buckets.items()):
        take = max(1, round(len(bucket) * args.fraction)) if args.fraction < 1 else len(bucket)
        kept.extend(rng.sample(bucket, take))

    rng.shuffle(kept)
    before, after = stats(rows), stats(kept)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out.with_suffix(".stats.json")).write_text(
        json.dumps({"fraction": args.fraction, "seed": args.seed, "before": before, "after": after},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
