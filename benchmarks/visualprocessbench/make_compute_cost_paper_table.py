import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


LABELS = {
    ("sft", "global_think_stepwise"): "SFT PRM (global thinking + stepwise)",
    ("sft", "direct_scores"): "SFT PRM (direct score generation)",
    ("sft", "stepwise"): "SFT PRM (warmup thinking + stepwise)",
    ("sft", "full"): "SFT PRM (full response)",
    ("visualprm", "paper_soft_score"): "VisualPRM-8B",
}

ORDER = {
    ("sft", "global_think_stepwise"): 0,
    ("sft", "direct_scores"): 1,
    ("visualprm", "paper_soft_score"): 2,
    ("sft", "stepwise"): 3,
    ("sft", "full"): 4,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_summaries(path: Path) -> Iterable[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("summaries"), list):
        for item in data["summaries"]:
            if isinstance(item, dict):
                row = dict(item)
                row["_summary_path"] = str(path)
                yield row
    elif isinstance(data, dict) and isinstance(data.get("summary"), dict):
        row = dict(data["summary"])
        row["_summary_path"] = str(path)
        yield row
    elif isinstance(data, dict) and "backend" in data and "mode" in data:
        row = dict(data)
        row["_summary_path"] = str(path)
        yield row


def discover_summaries(root: Path) -> List[Path]:
    paths = []
    for pattern in ("**/summary.json", "**/*_summary.json"):
        paths.extend(root.glob(pattern))
    return sorted(set(path for path in paths if path.is_file()))


def safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def metric(summary: Dict[str, Any], *keys: str, default=0.0):
    cur: Any = summary
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def paper_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    backend = str(summary.get("backend", ""))
    mode = str(summary.get("mode", ""))
    num_samples = int(summary.get("num_samples") or 0)
    num_steps = int(summary.get("num_steps") or 0)
    wall_time = float(summary.get("wall_time_sec") or 0.0)
    usage = summary.get("usage") or {}
    prompt_tokens = float(usage.get("prompt_tokens") or 0.0)
    completion_tokens = float(usage.get("completion_tokens") or 0.0)
    text_input_tokens = float(summary.get("text_input_tokens") or 0.0)
    generated_tokens = float(summary.get("generated_tokens") or 0.0)

    if prompt_tokens > 0:
        input_tokens = safe_div(prompt_tokens, num_samples)
        input_token_source = "api_prompt_tokens"
    else:
        input_tokens = safe_div(text_input_tokens, num_samples)
        input_token_source = "text_tokenizer_estimate"

    if completion_tokens > 0:
        output_tokens = safe_div(completion_tokens, num_samples)
        output_token_source = "api_completion_tokens"
    elif backend == "visualprm":
        output_tokens = 0.0
        output_token_source = "not_applicable_visualprm_soft_score"
    else:
        output_tokens = safe_div(generated_tokens, num_samples)
        output_token_source = str(summary.get("generated_tokens_source") or "generated_tokens")

    macro_f1 = float(metric(summary, "quality_metrics", "overall_step_macro_f1", "macro_f1", default=0.0))

    return {
        "method": LABELS.get((backend, mode), f"{backend}:{mode}"),
        "backend": backend,
        "mode": mode,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "avg_time_per_sample_s": safe_div(wall_time, num_samples),
        "avg_latency_per_sample_s": float(metric(summary, "latency_sec", "mean", default=0.0)),
        "avg_input_tokens_per_sample": input_tokens,
        "input_token_source": input_token_source,
        "avg_output_tokens_per_sample": output_tokens,
        "output_token_source": output_token_source,
        "macro_f1": macro_f1,
        "num_errors": int(summary.get("num_errors") or 0),
        "num_parse_errors": int(summary.get("num_parse_errors") or 0),
        "summary_path": summary.get("_summary_path", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "backend",
        "mode",
        "num_samples",
        "num_steps",
        "avg_time_per_sample_s",
        "avg_latency_per_sample_s",
        "avg_input_tokens_per_sample",
        "input_token_source",
        "avg_output_tokens_per_sample",
        "output_token_source",
        "macro_f1",
        "num_errors",
        "num_parse_errors",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | Samples | Steps | Avg time / sample (s) | Avg input tokens / sample | Avg output tokens / sample | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {num_samples} | {num_steps} | {avg_time:.3f} | {input_tokens:.1f} | {output_tokens:.1f} | {macro_f1:.4f} |".format(
                method=row["method"],
                num_samples=row["num_samples"],
                num_steps=row["num_steps"],
                avg_time=row["avg_time_per_sample_s"],
                input_tokens=row["avg_input_tokens_per_sample"],
                output_tokens=row["avg_output_tokens_per_sample"],
                macro_f1=row["macro_f1"],
            )
        )
    lines.extend([
        "",
        "Notes:",
        "",
        "- `Avg time / sample` is wall-clock run time divided by number of samples.",
        "- SFT input/output tokens use OpenAI-compatible API `usage.prompt_tokens` and `usage.completion_tokens`.",
        "- VisualPRM-8B does not generate text tokens in the paper soft-score path; its input tokens are text-tokenizer estimates and exclude visual patch tokens.",
        "- Use `avg_latency_per_sample_s` from the CSV if you need mean per-sample latency instead of throughput-normalized wall time.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready compute-cost table from benchmark summaries.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent / "outputs" / "compute_cost"))
    parser.add_argument("--summary", nargs="*", default=[],
                        help="Explicit summary JSON files. If omitted, scan --root.")
    parser.add_argument("--min-samples", type=int, default=0,
                        help="Filter out smoke-test runs. Use 2866 for full VisualProcessBench.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    paths = [Path(path) for path in args.summary] if args.summary else discover_summaries(root)
    rows = []
    for path in paths:
        for summary in iter_summaries(path):
            row = paper_row(summary)
            if args.min_samples and row["num_samples"] < args.min_samples:
                continue
            rows.append(row)

    rows.sort(key=lambda row: (ORDER.get((row["backend"], row["mode"]), 100), row["num_samples"], row["summary_path"]))
    if not rows:
        raise RuntimeError("No matching summary rows found.")

    output_csv = Path(args.output_csv) if args.output_csv else root / "paper_compute_cost_table.csv"
    output_md = Path(args.output_md) if args.output_md else root / "paper_compute_cost_table.md"
    write_csv(output_csv, rows)
    write_markdown(output_md, rows)
    print(output_md.read_text(encoding="utf-8"))
    print(f"csv: {output_csv}")
    print(f"markdown: {output_md}")


if __name__ == "__main__":
    main()
