import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import compute_metrics, write_json


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_rows_by_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "id" in row:
            latest[int(row["id"])] = row
    return [latest[idx] for idx in sorted(latest)]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def numeric_summary(values: Iterable[Any]) -> Dict[str, float]:
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return {"sum": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "sum": sum(vals),
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
    }


def normalize_row(row: Dict[str, Any], backend: str, mode: str) -> Dict[str, Any]:
    row = dict(row)
    row.setdefault("backend", backend)
    row.setdefault("mode", mode)
    if "latency_sec" not in row:
        row["latency_sec"] = row.get("latency", 0.0)
    if "num_steps" not in row:
        row["num_steps"] = len(row.get("ground_truth") or row.get("pred") or [])
    if "num_images" not in row:
        row["num_images"] = len(row.get("image") or [])
    row.setdefault("num_requests", 0)
    row.setdefault("usage", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_available_requests": 0,
    })
    if backend == "visualprm":
        row.setdefault("generated_tokens", 0)
        row.setdefault("generated_tokens_source", "not_applicable_visualprm_soft_score")
    else:
        row.setdefault("generated_tokens", row.get("usage", {}).get("completion_tokens", 0))
        row.setdefault("generated_tokens_source", "api_usage")
    row.setdefault("parse_error", any(value == -2 for value in row.get("pred", [])))
    return row


def usage_value(row: Dict[str, Any], key: str) -> int:
    value = (row.get("usage") or {}).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def summarize(rows: List[Dict[str, Any]], wall_time_sec: Optional[float]) -> Dict[str, Any]:
    if wall_time_sec is None or wall_time_sec <= 0:
        wall_time_sec = sum(float(row.get("latency_sec") or 0.0) for row in rows)
    num_samples = len(rows)
    num_steps = sum(int(row.get("num_steps") or 0) for row in rows)
    prompt_tokens = sum(usage_value(row, "prompt_tokens") for row in rows)
    completion_tokens = sum(usage_value(row, "completion_tokens") for row in rows)
    total_tokens = sum(usage_value(row, "total_tokens") for row in rows)
    usage_available = sum(usage_value(row, "usage_available_requests") for row in rows)
    generated_tokens = sum(int(row.get("generated_tokens") or 0) for row in rows)
    text_input_tokens = sum(int(row.get("text_input_tokens") or 0) for row in rows)

    summary = {
        "backend": rows[0].get("backend") if rows else "unknown",
        "mode": rows[0].get("mode") if rows else "unknown",
        "num_samples": num_samples,
        "num_steps": num_steps,
        "num_errors": sum(1 for row in rows if row.get("error")),
        "num_parse_errors": sum(1 for row in rows if row.get("parse_error")),
        "wall_time_sec": wall_time_sec,
        "throughput_samples_per_sec": num_samples / wall_time_sec if wall_time_sec > 0 else 0.0,
        "throughput_steps_per_sec": num_steps / wall_time_sec if wall_time_sec > 0 else 0.0,
        "latency_sec": numeric_summary(row.get("latency_sec", 0.0) for row in rows),
        "image_latency_sec": numeric_summary(
            row.get("image_encode_latency_sec", row.get("image_preprocess_latency_sec", 0.0))
            for row in rows
        ),
        "score_latency_sec": numeric_summary(row.get("score_latency_sec", 0.0) for row in rows),
        "requests_per_sample": numeric_summary(row.get("num_requests", 0) for row in rows),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_available_requests": usage_available,
        },
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_tokens / num_samples if num_samples else 0.0,
        "generated_tokens_per_step": generated_tokens / num_steps if num_steps else 0.0,
        "generated_tokens_per_sec": generated_tokens / wall_time_sec if wall_time_sec > 0 else 0.0,
        "text_input_tokens": text_input_tokens,
        "text_input_tokens_per_sample": text_input_tokens / num_samples if num_samples else 0.0,
    }
    try:
        summary["quality_metrics"] = compute_metrics(rows)
    except Exception as exc:
        summary["quality_metrics_error"] = repr(exc)
    return summary


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    fields = [
        "backend",
        "mode",
        "num_samples",
        "num_steps",
        "num_errors",
        "num_parse_errors",
        "wall_time_sec",
        "throughput_samples_per_sec",
        "throughput_steps_per_sec",
        "latency_mean_sec",
        "latency_p50_sec",
        "latency_p90_sec",
        "latency_p95_sec",
        "latency_p99_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generated_tokens",
        "generated_tokens_per_sample",
        "generated_tokens_per_step",
        "generated_tokens_per_sec",
        "text_input_tokens",
        "text_input_tokens_per_sample",
        "usage_available_requests",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        usage = summary.get("usage", {})
        writer.writerow({
            "backend": summary.get("backend"),
            "mode": summary.get("mode"),
            "num_samples": summary.get("num_samples"),
            "num_steps": summary.get("num_steps"),
            "num_errors": summary.get("num_errors"),
            "num_parse_errors": summary.get("num_parse_errors"),
            "wall_time_sec": summary.get("wall_time_sec"),
            "throughput_samples_per_sec": summary.get("throughput_samples_per_sec"),
            "throughput_steps_per_sec": summary.get("throughput_steps_per_sec"),
            "latency_mean_sec": summary.get("latency_sec", {}).get("mean"),
            "latency_p50_sec": summary.get("latency_sec", {}).get("p50"),
            "latency_p90_sec": summary.get("latency_sec", {}).get("p90"),
            "latency_p95_sec": summary.get("latency_sec", {}).get("p95"),
            "latency_p99_sec": summary.get("latency_sec", {}).get("p99"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "generated_tokens": summary.get("generated_tokens"),
            "generated_tokens_per_sample": summary.get("generated_tokens_per_sample"),
            "generated_tokens_per_step": summary.get("generated_tokens_per_step"),
            "generated_tokens_per_sec": summary.get("generated_tokens_per_sec"),
            "text_input_tokens": summary.get("text_input_tokens"),
            "text_input_tokens_per_sample": summary.get("text_input_tokens_per_sample"),
            "usage_available_requests": usage.get("usage_available_requests"),
        })


def write_samples_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "backend",
        "mode",
        "id",
        "data_source",
        "num_images",
        "num_steps",
        "latency_sec",
        "image_latency_sec",
        "score_latency_sec",
        "num_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generated_tokens",
        "generated_tokens_source",
        "text_input_tokens",
        "num_step_scores",
        "gpu_max_memory_allocated_mb",
        "gpu_max_memory_reserved_mb",
        "error",
        "parse_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "backend": row.get("backend"),
                "mode": row.get("mode"),
                "id": row.get("id"),
                "data_source": row.get("data_source"),
                "num_images": row.get("num_images"),
                "num_steps": row.get("num_steps"),
                "latency_sec": row.get("latency_sec"),
                "image_latency_sec": row.get("image_encode_latency_sec", row.get("image_preprocess_latency_sec", 0.0)),
                "score_latency_sec": row.get("score_latency_sec", 0.0),
                "num_requests": row.get("num_requests"),
                "prompt_tokens": usage_value(row, "prompt_tokens"),
                "completion_tokens": usage_value(row, "completion_tokens"),
                "total_tokens": usage_value(row, "total_tokens"),
                "generated_tokens": row.get("generated_tokens"),
                "generated_tokens_source": row.get("generated_tokens_source"),
                "text_input_tokens": row.get("text_input_tokens", 0),
                "num_step_scores": row.get("num_step_scores", 0),
                "gpu_max_memory_allocated_mb": row.get("gpu_max_memory_allocated_mb"),
                "gpu_max_memory_reserved_mb": row.get("gpu_max_memory_reserved_mb"),
                "error": row.get("error"),
                "parse_error": row.get("parse_error"),
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize compute-cost prediction JSONL files.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--wall-time-sec", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prefix = args.prefix or args.backend
    rows = [
        normalize_row(row, args.backend, args.mode)
        for row in latest_rows_by_id(load_jsonl(Path(args.predictions)))
    ]
    summary = summarize(rows, args.wall_time_sec if args.wall_time_sec > 0 else None)
    write_json({
        "predictions": args.predictions,
        "summary": summary,
    }, output_dir / f"{prefix}_summary.json")
    write_summary_csv(output_dir / f"{prefix}_summary.csv", summary)
    write_samples_csv(output_dir / f"{prefix}_samples.csv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {output_dir / f'{prefix}_summary.json'}")
    print(f"samples: {output_dir / f'{prefix}_samples.csv'}")


if __name__ == "__main__":
    main()
