import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import benchmark_compute_cost as cost


METHOD_TO_SFT_MODE = {
    "no_think": "no_think_single_pass",
    "global_think": "global_think_stepwise",
}


def split_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clear_outputs(run_dir: Path) -> None:
    patterns = [
        "summary.json",
        "summary.csv",
        "all_samples.csv",
        "comparison.json",
        "comparison.csv",
        "metadata.json",
        "selected_ids.json",
        "sft_*_samples.jsonl",
        "sft_*_samples.csv",
        "sft_*_requests.csv",
        "sft_*_samples.runtime.json",
        "visualprm_samples.jsonl",
        "visualprm_samples.csv",
        "visualprm_samples.runtime.json",
    ]
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.exists() and path.is_file():
                path.unlink()


def build_cost_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_dir=args.benchmark_dir,
        backends="",
        limit=args.limit,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        no_progress=args.no_progress,
        log_every=args.log_every,
        sft_base_url=args.sft_base_url,
        sft_api_key=args.sft_api_key,
        sft_model=args.sft_model,
        sft_mode="global_think_stepwise",
        sft_concurrency=args.sft_concurrency,
        sft_temperature=args.sft_temperature,
        sft_use_reference_answer=args.sft_use_reference_answer,
        sft_think_max_tokens=args.sft_think_max_tokens,
        sft_warmup_max_tokens=args.sft_think_max_tokens,
        sft_full_max_tokens=args.sft_full_max_tokens,
        sft_direct_max_tokens=args.sft_no_think_max_tokens,
        sft_max_retries=args.sft_max_retries,
        sft_request_timeout=args.sft_request_timeout,
        sft_tokenizer_path=args.sft_tokenizer_path,
        visualprm_model_path=args.visualprm_model_path,
        visualprm_dtype=args.visualprm_dtype,
        visualprm_threshold=args.visualprm_threshold,
    )


def load_selected_ids(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            if "selected_ids" in data:
                return [int(item) for item in data["selected_ids"]]
            metadata = data.get("metadata")
            if isinstance(metadata, dict) and "selected_ids" in metadata:
                return [int(item) for item in metadata["selected_ids"]]
        if isinstance(data, list):
            return [int(item) for item in data]
        raise ValueError(f"Cannot find selected_ids in {path}")
    text = path.read_text(encoding="utf-8")
    ids = [int(item) for item in text.replace(",", " ").split() if item.strip()]
    if not ids:
        raise ValueError(f"No ids found in {path}")
    return ids


def select_rows(args: argparse.Namespace, cost_args: SimpleNamespace):
    if not args.selected_ids_file:
        return cost.sample_rows(cost_args)
    rows = cost.load_jsonl(Path(args.benchmark_dir) / "test.jsonl")
    ids = load_selected_ids(Path(args.selected_ids_file))
    selected = []
    for idx in ids:
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"selected id {idx} out of range for {len(rows)} VPB rows")
        selected.append((idx, rows[idx]))
    return selected


def merge_and_write_metadata(path: Path, metadata: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    if not path.exists() or overwrite:
        write_json(path, metadata)
        return metadata

    old = read_json(path)
    if not isinstance(old, dict):
        old = {}
    old_ids = old.get("selected_ids")
    new_ids = metadata.get("selected_ids")
    if old_ids is not None and new_ids is not None and list(old_ids) != list(new_ids):
        raise ValueError(
            f"{path} already exists with different selected_ids. "
            "Use a new --run-name or pass the matching --selected-ids-file."
        )
    old_methods = old.get("methods", [])
    merged_methods = []
    for item in list(old_methods) + list(metadata.get("methods", [])):
        if item not in merged_methods:
            merged_methods.append(item)
    current_methods = set(metadata.get("methods", []))
    uses_sft = bool(current_methods & set(METHOD_TO_SFT_MODE))
    uses_visualprm = "visualprm" in current_methods
    merged = dict(old)
    for key, value in metadata.items():
        if key in {"methods", "selected_ids"}:
            continue
        if key.startswith("sft_") and not uses_sft and key in merged:
            continue
        if key.startswith("visualprm_") and not uses_visualprm and key in merged:
            continue
        merged[key] = value
    merged["methods"] = merged_methods
    merged["selected_ids"] = old_ids if old_ids is not None else new_ids
    write_json(path, merged)
    return merged


def require_absent_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite or choose another --run-name.")


def run_sft_method(cost_args: SimpleNamespace, rows_with_ids, run_dir: Path, method: str) -> None:
    mode = METHOD_TO_SFT_MODE[method]
    cost_args.sft_mode = mode
    output = run_dir / f"sft_{mode}_samples.jsonl"
    require_absent_or_overwrite(output, cost_args.overwrite)
    start = time.perf_counter()
    asyncio.run(cost.run_sft(cost_args, rows_with_ids, output))
    wall_time = time.perf_counter() - start
    cost.write_runtime(output, wall_time)
    rows = cost.latest_rows_by_id(cost.load_jsonl(output))
    cost.write_sample_csv(run_dir / f"sft_{mode}_samples.csv", rows)
    cost.write_request_csv(run_dir / f"sft_{mode}_requests.csv", rows)


def run_visualprm_method(cost_args: SimpleNamespace, rows_with_ids, run_dir: Path) -> None:
    output = run_dir / "visualprm_samples.jsonl"
    require_absent_or_overwrite(output, cost_args.overwrite)
    start = time.perf_counter()
    cost.run_visualprm(cost_args, rows_with_ids, output)
    wall_time = time.perf_counter() - start
    cost.write_runtime(output, wall_time)
    rows = cost.latest_rows_by_id(cost.load_jsonl(output))
    cost.write_sample_csv(run_dir / "visualprm_samples.csv", rows)


def comparison_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    usage = summary.get("usage", {})
    quality = summary.get("quality_metrics", {})
    macro = (quality.get("overall_step_macro_f1") or {}).get("macro_f1")
    num_samples = int(summary.get("num_samples") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    output_note = ""
    if summary.get("backend") == "visualprm":
        output_note = (
            "VisualPRM paper soft-score path is non-generative: generated tokens are 0; "
            "score output units equal the number of scored steps."
        )
    elif not completion_tokens and summary.get("generated_tokens"):
        output_note = "API completion_tokens unavailable; generated_tokens uses tokenizer output estimate."
    elif not completion_tokens:
        output_note = "No API completion_tokens and no tokenizer estimate available."
    return {
        "backend": summary.get("backend"),
        "mode": summary.get("mode"),
        "num_samples": num_samples,
        "num_steps": summary.get("num_steps"),
        "score_outputs": summary.get("score_outputs"),
        "avg_score_outputs_per_sample": summary.get("score_outputs_per_sample"),
        "avg_score_outputs_per_step": summary.get("score_outputs_per_step"),
        "avg_latency_sec_per_sample": (summary.get("latency_sec") or {}).get("mean"),
        "p50_latency_sec_per_sample": (summary.get("latency_sec") or {}).get("p50"),
        "p90_latency_sec_per_sample": (summary.get("latency_sec") or {}).get("p90"),
        "avg_generated_tokens_per_sample": summary.get("generated_tokens_per_sample"),
        "avg_generated_tokens_per_step": summary.get("generated_tokens_per_step"),
        "avg_output_tokens_per_sample": summary.get("generated_tokens_per_sample"),
        "avg_output_tokens_per_step": summary.get("generated_tokens_per_step"),
        "completion_tokens_per_sample": completion_tokens / num_samples if num_samples else 0.0,
        "avg_input_text_tokens_per_sample": summary.get("text_input_tokens_per_sample"),
        "requests_per_sample": (summary.get("requests_per_sample") or {}).get("mean"),
        "throughput_samples_per_sec": summary.get("throughput_samples_per_sec"),
        "macro_f1": macro,
        "num_errors": summary.get("num_errors"),
        "num_parse_errors": summary.get("num_parse_errors"),
        "output_token_note": output_note,
    }


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "mode",
        "num_samples",
        "num_steps",
        "score_outputs",
        "avg_score_outputs_per_sample",
        "avg_score_outputs_per_step",
        "avg_latency_sec_per_sample",
        "p50_latency_sec_per_sample",
        "p90_latency_sec_per_sample",
        "avg_generated_tokens_per_sample",
        "avg_generated_tokens_per_step",
        "avg_output_tokens_per_sample",
        "avg_output_tokens_per_step",
        "completion_tokens_per_sample",
        "avg_input_text_tokens_per_sample",
        "requests_per_sample",
        "throughput_samples_per_sec",
        "macro_f1",
        "num_errors",
        "num_parse_errors",
        "output_token_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure no-thinking, global-thinking, and VisualPRM paper-style cost "
            "on the same small VisualProcessBench subset."
        )
    )
    parser.add_argument("--benchmark-dir", default=str(cost.DEFAULT_BENCH_DIR))
    parser.add_argument("--methods", default="no_think,global_think,visualprm",
                        help="Comma-separated: no_think,global_think,visualprm")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--sample-strategy", choices=("first", "random"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selected-ids-file",
        default="",
        help=(
            "Optional JSON/TXT file containing selected_ids. Accepts this script's metadata.json, "
            "comparison.json, a JSON list, or comma/space-separated ids."
        ),
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs" / "compute_cost"))
    parser.add_argument("--run-name", default=time.strftime("think_ablation_cost_%Y%m%d_%H%M%S"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--sft-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--sft-api-key", default="EMPTY")
    parser.add_argument("--sft-model", default="auto")
    parser.add_argument("--sft-concurrency", type=int, default=4)
    parser.add_argument("--sft-temperature", type=float, default=0.0)
    parser.add_argument(
        "--sft-use-reference-answer",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether to include VPB reference answers in SFT/VRPRM prompts. Default 0 avoids answer leakage.",
    )
    parser.add_argument("--sft-think-max-tokens", type=int, default=1024)
    parser.add_argument("--sft-no-think-max-tokens", type=int, default=512)
    parser.add_argument("--sft-full-max-tokens", type=int, default=2048)
    parser.add_argument("--sft-max-retries", type=int, default=3)
    parser.add_argument("--sft-request-timeout", type=float, default=300.0)
    parser.add_argument("--sft-tokenizer-path", default="")

    parser.add_argument(
        "--visualprm-model-path",
        default="VisualPRM/VisualPRM-8B",
    )
    parser.add_argument("--visualprm-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--visualprm-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = split_csv(args.methods)
    unknown = sorted(set(methods) - {"no_think", "global_think", "visualprm"})
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if args.limit < 1 and not args.selected_ids_file:
        raise ValueError("--limit should be a positive small subset size for this cost probe")

    cost_args = build_cost_args(args)
    rows_with_ids = select_rows(args, cost_args)
    if not rows_with_ids:
        raise RuntimeError("No VisualProcessBench samples selected")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_outputs(run_dir)

    metadata = {
        "benchmark_dir": args.benchmark_dir,
        "methods": methods,
        "num_selected_samples": len(rows_with_ids),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "selected_ids_file": args.selected_ids_file,
        "selected_ids": [idx for idx, _ in rows_with_ids],
        "sft_base_url": args.sft_base_url,
        "sft_model": args.sft_model,
        "sft_concurrency": args.sft_concurrency,
        "sft_use_reference_answer": bool(args.sft_use_reference_answer),
        "sft_think_max_tokens": args.sft_think_max_tokens,
        "sft_no_think_max_tokens": args.sft_no_think_max_tokens,
        "sft_tokenizer_path": args.sft_tokenizer_path,
        "visualprm_model_path": args.visualprm_model_path,
        "visualprm_dtype": args.visualprm_dtype,
        "visualprm_threshold": args.visualprm_threshold,
    }
    metadata = merge_and_write_metadata(run_dir / "metadata.json", metadata, args.overwrite)
    write_json(run_dir / "selected_ids.json", {"selected_ids": [idx for idx, _ in rows_with_ids]})

    for method in methods:
        if method in METHOD_TO_SFT_MODE:
            print(f"==> running {method} ({METHOD_TO_SFT_MODE[method]})", flush=True)
            run_sft_method(cost_args, rows_with_ids, run_dir, method)
        elif method == "visualprm":
            print("==> running visualprm (paper_soft_score)", flush=True)
            run_visualprm_method(cost_args, rows_with_ids, run_dir)

    summaries = []
    all_rows = []
    for _path, rows, wall_time_sec in cost.collect_existing_backend_rows(run_dir):
        all_rows.extend(rows)
        summaries.append(cost.summarize_backend(rows, wall_time_sec))

    summary = {"metadata": metadata, "summaries": summaries}
    cost.write_json(summary, run_dir / "summary.json")
    cost.write_summary_csv(run_dir / "summary.csv", summaries)
    cost.write_sample_csv(run_dir / "all_samples.csv", all_rows)

    comparison = [comparison_row(item) for item in summaries]
    write_json(run_dir / "comparison.json", {"metadata": metadata, "rows": comparison})
    write_comparison_csv(run_dir / "comparison.csv", comparison)

    print(json.dumps({"metadata": metadata, "comparison": comparison}, ensure_ascii=False, indent=2))
    print(f"think ablation cost outputs: {run_dir}")


if __name__ == "__main__":
    main()
