#!/usr/bin/env python3
"""Assemble the paper Table 2 (VisualProcessBench macro F1) from prediction
metrics JSONs produced by benchmarks/visualprocessbench/run_eval_vpb.sh.

Every model row is described by (--label, --metrics, optional --samples-note).
The output CSV/markdown contains, per source and overall:

- per-source macro F1 (DynaMath, MMMU, MathVerse, MathVision, WeMath)
- Overall (pooled)  = overall_step_macro_f1  [official protocol, step-weighted]
- Overall (mean)    = mean_source_macro_f1   [unweighted subset mean]

The paper reports ONE Overall column: the official pooled macro F1,
aligned with the VisualPRM paper convention (docs/PROTOCOL.md). The
unweighted subset mean is printed to the console as an internal diagnostic
only and does not appear in the table. Sources are uneven in size
(MathVerse 1026 / MathVision 712 / DynaMath 570 / WeMath 291 / MMMU 267 of
2,866), so the two conventions can diverge; baseline rows quoted from other
papers must use the pooled Overall from the original paper.

Example:
    python analysis/make_paper_table2_vpb.py \
      --row "VRPRM-RL" outputs/vrprm_rl...metrics.json \
      --row "VRPRM-SFT" outputs/vrprm_sft...metrics.json \
      --row "w/o Thinking" outputs/no_think...metrics.json \
      --output-md table2.md --output-csv table2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SOURCE_ORDER = ["DynaMath", "MMMU_DEV_VAL", "MathVerse_MINI_Vision_Only", "MathVision_MINI", "WeMath"]
SOURCE_DISPLAY = {
    "DynaMath": "DynaMath",
    "MMMU_DEV_VAL": "MMMU",
    "MathVerse_MINI_Vision_Only": "MathVerse",
    "MathVision_MINI": "MathVision",
    "WeMath": "WeMath",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row", action="append", nargs=2, metavar=("LABEL", "METRICS_JSON"), required=True,
        help="Model row: label + metrics JSON path. Repeat for each row.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def load_row(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_source = data.get("by_source", {})
    row = {SOURCE_DISPLAY[s]: round(by_source[s]["macro_f1"] * 100, 2) for s in SOURCE_ORDER if s in by_source}
    row["Overall"] = round(data["overall_step_macro_f1"]["macro_f1"] * 100, 2)
    row["_diagnostic_mean"] = round(data["mean_source_macro_f1"] * 100, 2)
    row["_samples"] = {SOURCE_DISPLAY[s]: by_source[s].get("num_samples") for s in SOURCE_ORDER if s in by_source}
    return row


def main():
    args = parse_args()
    columns = [SOURCE_DISPLAY[s] for s in SOURCE_ORDER] + ["Overall"]

    rows = {}
    for label, path in args.row:
        rows[label] = load_row(path)

    lines = ["| Model | " + " | ".join(columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for label, row in rows.items():
        lines.append(f"| {label} | " + " | ".join(str(row[c]) for c in columns) + " |")

    md = "\n".join(lines) + "\n\nPer-source sample counts: " + json.dumps(
        next(iter(rows.values()))["_samples"], ensure_ascii=False) + "\n"
    md += "Diagnostic (not for the paper): unweighted subset means - " + ", ".join(
        f"{label}={row['_diagnostic_mean']}" for label, row in rows.items()) + "\n"

    if args.output_md:
        Path(args.output_md).write_text(md, encoding="utf-8")
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + columns)
            for label, row in rows.items():
                writer.writerow([label] + [row[c] for c in columns])
    print(md)


if __name__ == "__main__":
    main()
