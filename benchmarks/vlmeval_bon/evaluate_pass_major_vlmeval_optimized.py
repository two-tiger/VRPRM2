import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import (
    LOCAL_LMU_DATA,
    apply_eval_api_env,
    default_judge_for,
    load_jsonl,
    probe_eval_judge_api,
    resolve_dataset_name,
    sanitize_dataframe_for_excel,
    write_json,
)
from aggregate_pass_major_vlmeval import candidate_scored_paths, hit_map_from_prediction, wemath_official_from_hits
from evaluate_pass_major_from_rollouts import (
    cleanup_answer,
    extract_final_answer,
    is_correct,
    latest_rows_by_index,
    parse_choice_map,
    parse_k_list,
    vote_key,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimized official Pass@K plus local-vote Major@K evaluation from cached rollout candidates. "
            "Pass@K evaluates only samples that have not passed yet."
        )
    )
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n<N>.jsonl file.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--k-list", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--csv-output", default="")
    parser.add_argument("--rollout-n", type=int, default=128)
    parser.add_argument("--api-nproc", type=int, default=4)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--judge", default="")
    parser.add_argument("--judge-args", default="", help="Extra judge kwargs as JSON.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--candidate-prediction-mode",
        choices=("raw", "final"),
        default="raw",
        help="Use raw candidate responses for Pass@K official judging, or extract compact final answers.",
    )
    parser.add_argument(
        "--skip-errors",
        type=int,
        choices=(0, 1),
        default=1,
        help="Treat rollout candidates with generation errors as empty/wrong.",
    )
    parser.add_argument(
        "--wemath-primary",
        choices=("strict", "loose", "row"),
        default="strict",
        help="Primary Pass@K score for WeMath. strict matches VLMEvalKit Score (Strict).",
    )
    parser.add_argument("--numeric-tol", type=float, default=1e-6)
    return parser.parse_args()


def candidate_error(row: Dict[str, Any], candidate_idx: int) -> str:
    errors = row.get("errors")
    if isinstance(errors, list) and 0 <= candidate_idx < len(errors):
        return str(errors[candidate_idx] or "")
    return ""


def candidate_prediction(row: Dict[str, Any], candidate_idx: int, mode: str, skip_errors: bool) -> str:
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or candidate_idx >= len(rollouts):
        return ""
    if skip_errors and candidate_error(row, candidate_idx):
        return ""
    rollout = str(rollouts[candidate_idx] or "")
    if not rollout.strip():
        return ""
    return extract_final_answer(rollout) if mode == "final" else rollout


def load_local_dataset_frame(dataset_name: str):
    import pandas as pd

    candidates = [
        LOCAL_LMU_DATA / f"{dataset_name}.tsv",
        LOCAL_LMU_DATA / "_downloads" / f"{dataset_name}.tsv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep="\t"), f"local TSV: {path}"
    return None, None


def load_dataset_frame(dataset_name: str):
    local_data, local_source = load_local_dataset_frame(dataset_name)
    if local_data is not None:
        return local_data, local_source

    from vlmeval.dataset import build_dataset

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")
    return dataset.data.copy(), "VLMEvalKit build_dataset"


def restrict_dataset_frame(dataset_frame, indices: Iterable[str]):
    index_set = {str(index) for index in indices}
    data = dataset_frame.copy()
    data["_index_str"] = data["index"].astype(str)
    data = data[data["_index_str"].isin(index_set)].copy()
    data = data.drop(columns=["_index_str"])
    return data


def write_prediction_xlsx(template, predictions: Dict[str, str], output_path: Path) -> int:
    data = template.copy()
    data["_index_str"] = data["index"].astype(str)
    data = data[data["_index_str"].isin(predictions)].copy()
    if data.empty:
        return 0
    data["prediction"] = data["_index_str"].map(predictions).fillna("")
    data = data.drop(columns=["_index_str"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(data).to_excel(output_path, index=False)
    return len(data)


def dump_eval_result(result: Any, output: Path) -> None:
    import pandas as pd

    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, pd.DataFrame):
        if output.suffix.lower() == ".json":
            write_json(output, result.to_dict(orient="records"))
        else:
            result.to_csv(output, index=False)
    elif isinstance(result, dict):
        write_json(output, result)
    else:
        write_json(output, {"result": None})


def evaluate_prediction(dataset, prediction: Path, output: Path, judge_kwargs: Dict[str, Any], dataset_name: str) -> None:
    try:
        result = dataset.evaluate(str(prediction), **judge_kwargs.copy())
    except AssertionError as err:
        probe_eval_judge_api(
            judge_kwargs,
            context=f"dataset={dataset_name}, prediction={prediction}",
        )
        raise AssertionError(
            f"{err}\n"
            f"Current VLMEvalKit judge={judge_kwargs.get('model')}. "
            "See the VLMEvalKit judge API debug block above for the exact proxy response."
        ) from err
    except ZeroDivisionError as err:
        if dataset_name == "WeMath":
            try:
                hit_map, _, scored_path = hit_map_from_prediction(prediction)
            except RuntimeError:
                raise
            write_json(
                output,
                {
                    "status": "skipped_wemath_subset_aggregate",
                    "reason": (
                        "VLMEvalKit WeMath aggregate assumes the full 525-question set; "
                        "the optimized Pass@K path evaluates only the current pending subset."
                    ),
                    "scored_path": str(scored_path),
                    "num_scored_rows": len(hit_map),
                    "error": repr(err),
                },
            )
            print(
                f"[warn] skipped WeMath subset aggregate for {prediction}; "
                f"using per-sample hits from {scored_path}"
            )
            return
        raise
    dump_eval_result(result, output)


def clear_eval_sidecars(prediction: Path, eval_output: Path) -> None:
    if eval_output.exists():
        eval_output.unlink()
    for path in candidate_scored_paths(prediction):
        if path != prediction and path.exists():
            path.unlink()


def has_hit_sidecar(prediction: Path) -> bool:
    try:
        hit_map_from_prediction(prediction)
    except RuntimeError:
        return False
    return True


def parse_extra_judge_args(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def make_judge_kwargs(args, dataset_name: str, dataset_type: str) -> Dict[str, Any]:
    judge = args.judge or default_judge_for(dataset_name, dataset_type)
    judge_kwargs = {
        "model": judge,
        "nproc": args.api_nproc,
        "retry": args.retry,
        "verbose": args.verbose,
    }
    judge_kwargs.update(parse_extra_judge_args(args.judge_args))
    return judge_kwargs


def compute_pass_metric(dataset: str, frame, hit_map: Dict[str, bool], wemath_primary: str) -> Tuple[float, str, Dict[str, Any]]:
    if dataset == "WeMath":
        value, details = wemath_official_from_hits(frame, hit_map, wemath_primary)
        return value, f"WeMath {wemath_primary}", details
    value = sum(hit_map.values()) / len(hit_map) if hit_map else 0.0
    return value, "row official hit mean", {"row_accuracy": value}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def choice_map_from_record(record: Dict[str, Any]) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for letter in "ABCDEFGHI":
        if letter in record and not is_missing(record[letter]):
            choices[letter] = str(record[letter])
    if choices:
        return choices
    return parse_choice_map(str(record.get("question") or record.get("prompt") or ""))


def compute_major_metrics(
    rollout_rows: List[Dict[str, Any]],
    records_by_index: Dict[str, Dict[str, Any]],
    k_values: List[int],
    skip_errors: bool,
    numeric_tol: float,
) -> Dict[str, Any]:
    target_ks = set(k_values)
    max_k = max(k_values)
    stats = {
        k: {"correct": 0, "total": 0, "ties": 0, "no_vote": 0}
        for k in k_values
    }

    for row in rollout_rows:
        index = str(row["index"])
        record = records_by_index.get(index)
        if record is None:
            continue
        choice_map = choice_map_from_record(record)
        reference = record.get("answer")
        rollouts = row.get("rollouts") if isinstance(row.get("rollouts"), list) else []
        counts: Counter[str] = Counter()
        first_seen: Dict[str, int] = {}
        display: Dict[str, str] = {}

        for candidate_idx in range(max_k):
            if candidate_idx < len(rollouts) and not (skip_errors and candidate_error(row, candidate_idx)):
                answer = cleanup_answer(extract_final_answer(str(rollouts[candidate_idx] or "")))
                if answer:
                    key = vote_key(answer, choice_map)
                    if key and key != "text:":
                        counts[key] += 1
                        first_seen.setdefault(key, candidate_idx)
                        display.setdefault(key, answer)

            current_k = candidate_idx + 1
            if current_k not in target_ks:
                continue

            if counts:
                max_count = max(counts.values())
                tied_keys = [key for key, count in counts.items() if count == max_count]
                best_key = min(tied_keys, key=lambda key: first_seen[key])
                major_answer = display[best_key]
                tied = len(tied_keys) > 1
            else:
                major_answer = ""
                tied = False

            stats[current_k]["total"] += 1
            stats[current_k]["ties"] += int(tied)
            stats[current_k]["no_vote"] += int(not major_answer)
            if is_correct(major_answer, reference, choice_map, numeric_tol):
                stats[current_k]["correct"] += 1

    metrics: Dict[str, Any] = {}
    for k in k_values:
        item = stats[k]
        metrics[str(k)] = {
            "major_at_k": item["correct"] / item["total"] if item["total"] else 0.0,
            "major_correct": item["correct"],
            "num_samples": item["total"],
            "num_ties": item["ties"],
            "num_no_vote": item["no_vote"],
            "metric_name": "local final-answer majority vote exact",
        }
    return metrics


def candidate_output_paths(output_dir: Path, candidate_idx: int, rollout_n: int) -> Tuple[Path, Path]:
    prefix = f"pass_candidate{candidate_idx:03d}_pending_from_n{rollout_n}"
    return output_dir / f"{prefix}_selected.xlsx", output_dir / f"{prefix}_eval.json"


def write_metrics_csv(path: Path, summary: Dict[str, Any], k_values: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "dataset",
                "metric_mode",
                "k",
                "pass_at_k",
                "major_at_k",
                "pass_metric_name",
                "major_metric_name",
                "num_samples",
            ],
        )
        writer.writeheader()
        for k in k_values:
            item = summary["k"][str(k)]
            writer.writerow(
                {
                    "model_label": summary.get("model_label", ""),
                    "dataset": summary["dataset"],
                    "metric_mode": summary["metric_mode"],
                    "k": k,
                    "pass_at_k": item["pass_at_k"],
                    "major_at_k": item["major_at_k"],
                    "pass_metric_name": item["pass_metric_name"],
                    "major_metric_name": item["major_metric_name"],
                    "num_samples": item["num_samples"],
                }
            )


def main():
    args = parse_args()
    apply_eval_api_env()
    k_values = parse_k_list(args.k_list)
    max_k = max(k_values)
    if max_k > args.rollout_n:
        raise ValueError(f"max K ({max_k}) cannot exceed --rollout-n ({args.rollout_n})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    if not rollout_rows:
        raise RuntimeError(f"no rollout rows found: {args.rollouts}")

    dataset_name = resolve_dataset_name(args.dataset)
    dataset_frame, dataset_source = load_dataset_frame(dataset_name)
    records_by_index = {
        str(record["index"]): record
        for record in dataset_frame.to_dict(orient="records")
        if "index" in record
    }
    rollout_rows = [row for row in rollout_rows if str(row.get("index")) in records_by_index]
    if not rollout_rows:
        raise RuntimeError("No rollout rows match the VLMEvalKit dataset indices")

    eval_indices = [str(row["index"]) for row in rollout_rows]
    eval_frame = restrict_dataset_frame(dataset_frame, eval_indices)

    from vlmeval.dataset import DATASET_TYPE, build_dataset

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")
    judge_kwargs = make_judge_kwargs(args, dataset_name, DATASET_TYPE(dataset_name))

    pass_hit_map = {index: False for index in eval_indices}
    pass_metrics_by_k: Dict[str, Dict[str, Any]] = {}
    candidate_summaries = []

    pending = set(eval_indices)
    next_k_pos = 0
    for candidate_idx in range(max_k):
        selected_path, eval_path = candidate_output_paths(output_dir, candidate_idx, args.rollout_n)
        pending_before = len(pending)
        predictions = {}
        if pending:
            for row in rollout_rows:
                index = str(row["index"])
                if index not in pending:
                    continue
                prediction = candidate_prediction(
                    row,
                    candidate_idx=candidate_idx,
                    mode=args.candidate_prediction_mode,
                    skip_errors=bool(args.skip_errors),
                )
                if prediction:
                    predictions[index] = prediction

        num_written = 0
        num_new_hits = 0
        scored_path = ""
        if predictions:
            if args.overwrite:
                clear_eval_sidecars(selected_path, eval_path)

            need_eval = args.overwrite or not eval_path.exists() or not selected_path.exists()
            if not need_eval and not has_hit_sidecar(selected_path):
                need_eval = True

            if need_eval:
                num_written = write_prediction_xlsx(eval_frame, predictions, selected_path)
                evaluate_prediction(dataset, selected_path, eval_path, judge_kwargs, dataset_name)
            else:
                num_written = len(predictions)

            hit_map, _, sidecar_path = hit_map_from_prediction(selected_path)
            scored_path = str(sidecar_path)
            for index, hit in hit_map.items():
                if hit and index in pending:
                    pass_hit_map[index] = True
                    pending.remove(index)
                    num_new_hits += 1

        candidate_summaries.append(
            {
                "candidate_idx": candidate_idx,
                "pending_before": pending_before,
                "num_evaluated": num_written,
                "num_new_hits": num_new_hits,
                "pending_after": len(pending),
                "prediction": str(selected_path) if predictions else "",
                "eval_output": str(eval_path) if predictions else "",
                "scored_path": scored_path,
            }
        )

        current_k = candidate_idx + 1
        while next_k_pos < len(k_values) and k_values[next_k_pos] == current_k:
            value, metric_name, details = compute_pass_metric(args.dataset, eval_frame, pass_hit_map, args.wemath_primary)
            pass_metrics_by_k[str(current_k)] = {
                "pass_at_k": value,
                "pass_metric_name": metric_name,
                "pass_details": details,
                "pass_correct_rows": sum(pass_hit_map.values()),
                "num_samples": len(pass_hit_map),
            }
            next_k_pos += 1

        if not pending:
            while next_k_pos < len(k_values):
                k = k_values[next_k_pos]
                value, metric_name, details = compute_pass_metric(args.dataset, eval_frame, pass_hit_map, args.wemath_primary)
                pass_metrics_by_k[str(k)] = {
                    "pass_at_k": value,
                    "pass_metric_name": metric_name,
                    "pass_details": details,
                    "pass_correct_rows": sum(pass_hit_map.values()),
                    "num_samples": len(pass_hit_map),
                }
                next_k_pos += 1
            break

    while next_k_pos < len(k_values):
        k = k_values[next_k_pos]
        value, metric_name, details = compute_pass_metric(args.dataset, eval_frame, pass_hit_map, args.wemath_primary)
        pass_metrics_by_k[str(k)] = {
            "pass_at_k": value,
            "pass_metric_name": metric_name,
            "pass_details": details,
            "pass_correct_rows": sum(pass_hit_map.values()),
            "num_samples": len(pass_hit_map),
        }
        next_k_pos += 1

    major_metrics_by_k = compute_major_metrics(
        rollout_rows,
        records_by_index,
        k_values,
        skip_errors=bool(args.skip_errors),
        numeric_tol=args.numeric_tol,
    )

    metrics_by_k = {}
    for k in k_values:
        key = str(k)
        pass_item = pass_metrics_by_k[key]
        major_item = major_metrics_by_k[key]
        metrics_by_k[key] = {
            "pass_at_k": pass_item["pass_at_k"],
            "major_at_k": major_item["major_at_k"],
            "pass_metric_name": pass_item["pass_metric_name"],
            "major_metric_name": major_item["metric_name"],
            "num_samples": pass_item["num_samples"],
            "pass_correct_rows": pass_item["pass_correct_rows"],
            "major_correct": major_item["major_correct"],
            "pass_details": pass_item["pass_details"],
            "major_details": {
                "num_ties": major_item["num_ties"],
                "num_no_vote": major_item["num_no_vote"],
            },
        }

    summary = {
        "dataset": args.dataset,
        "vlmeval_dataset": dataset_name,
        "model_label": args.model_label,
        "metric_mode": "vlmeval_pass_early_stop_major_local_vote",
        "rollouts": args.rollouts,
        "rollout_n": args.rollout_n,
        "k_list": k_values,
        "dataset_source": dataset_source,
        "candidate_prediction_mode": args.candidate_prediction_mode,
        "skip_errors": bool(args.skip_errors),
        "wemath_primary": args.wemath_primary,
        "num_rollout_rows": len(rollout_rows),
        "candidate_evaluations": candidate_summaries,
        "k": metrics_by_k,
        "definition": {
            "pass_at_k": (
                "official VLMEvalKit judging with per-sample early stopping; a sample is not evaluated for "
                "later candidates after its first passing candidate"
            ),
            "major_at_k": (
                "local final-answer extraction plus majority vote; correctness is local exact/numeric/choice "
                "matching against the reference"
            ),
            "major_tie_break": "ties are broken by the earliest candidate order among tied answers",
        },
    }

    write_json(Path(args.summary_output), summary)
    if args.csv_output:
        write_metrics_csv(Path(args.csv_output), summary, k_values)
    print(json.dumps(metrics_by_k, ensure_ascii=False, indent=2))
    print(f"wrote optimized metrics: {args.summary_output}")
    if args.csv_output:
        print(f"wrote csv: {args.csv_output}")


if __name__ == "__main__":
    main()
