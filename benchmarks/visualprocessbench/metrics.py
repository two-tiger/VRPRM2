"""VisualProcessBench step-level metrics.

Two Overall conventions are reported together (see docs/PROTOCOL.md):

- ``overall_step_macro_f1``: macro F1 over the correct/incorrect classes
  computed on ALL steps pooled across the five sources (step-weighted).
  This is the official VisualProcessBench protocol ("the overall score is
  the micro average of the score from different data sources", Wang et al.)
  and the number quoted as "Overall" in the paper's main tables.
- ``mean_source_macro_f1``: unweighted mean of the five per-source macro
  F1 values. The paper reports ONLY the pooled number as Overall (aligned
  with the reference-paper convention); the subset mean is kept here as an
  internal diagnostic. The two can differ by more than a point because the
  sources are uneven in size (MathVerse 1026 / MathVision 712 / DynaMath
  570 / WeMath 291 / MMMU 267 samples out of 2,866).

Neutral ground-truth steps (label 0) are excluded from both. Predictions
that cannot be parsed to 0/1 are kept as an invalid class and count against
both classes; ``invalid_predictions`` reports how many.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def binary_f1(labels, preds, positive_label):
    tp = fp = fn = 0
    for label, pred in zip(labels, preds):
        label_pos = label == positive_label
        pred_pos = pred == positive_label
        if label_pos and pred_pos:
            tp += 1
        elif not label_pos and pred_pos:
            fp += 1
        elif label_pos and not pred_pos:
            fn += 1
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom)


def macro_f1_correct_incorrect(labels, preds):
    if len(labels) != len(preds):
        raise ValueError(f"label/pred length mismatch: {len(labels)} != {len(preds)}")
    return {
        "correct_f1": binary_f1(labels, preds, 1),
        "incorrect_f1": binary_f1(labels, preds, -1),
        "macro_f1": (binary_f1(labels, preds, 1) + binary_f1(labels, preds, -1)) / 2,
        "num_steps_eval": len(labels),
    }


def normalize_pred(value):
    if value in (1, "1", True):
        return 1
    if value in (0, "0", -1, "-1", False):
        return -1
    return -2


def collect_labels_preds(rows):
    labels = []
    preds = []
    dropped_neutral = 0
    invalid_predictions = 0

    for row in rows:
        gt = row["ground_truth"]
        pred = row["pred"]
        if len(gt) != len(pred):
            raise ValueError(f"Row {row.get('id')} length mismatch: {len(gt)} != {len(pred)}")
        for label, value in zip(gt, pred):
            if label == 0:
                dropped_neutral += 1
                continue
            pred_label = normalize_pred(value)
            if pred_label not in (1, -1):
                invalid_predictions += 1
            labels.append(label)
            preds.append(pred_label)
    return labels, preds, dropped_neutral, invalid_predictions


def compute_metrics(rows):
    labels, preds, dropped_neutral, invalid_predictions = collect_labels_preds(rows)
    overall = macro_f1_correct_incorrect(labels, preds)
    overall["dropped_neutral_steps"] = dropped_neutral
    overall["invalid_predictions"] = invalid_predictions

    by_source = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("data_source", "unknown")].append(row)
    for source, source_rows in sorted(groups.items()):
        source_labels, source_preds, source_dropped, source_invalid = collect_labels_preds(source_rows)
        metrics = macro_f1_correct_incorrect(source_labels, source_preds)
        metrics["num_samples"] = len(source_rows)
        metrics["dropped_neutral_steps"] = source_dropped
        metrics["invalid_predictions"] = source_invalid
        by_source[source] = metrics

    source_macro = (
        sum(m["macro_f1"] for m in by_source.values()) / len(by_source)
        if by_source
        else 0.0
    )
    return {
        "overall_step_macro_f1": overall,
        "mean_source_macro_f1": source_macro,
        "by_source": by_source,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VisualProcessBench metrics from predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL produced by api_eval_sft.py or visualprm_paper_eval.py")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.predictions)
    metrics = compute_metrics(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        write_json(metrics, args.output)


if __name__ == "__main__":
    main()
