import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import LOCAL_LMU_DATA, load_jsonl, resolve_dataset_name, sanitize_dataframe_for_excel, write_json
from evaluate_pass_major_from_rollouts import (
    extract_final_answer,
    latest_rows_by_index,
    majority_answer,
    parse_choice_map,
    parse_k_list,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build VLMEvalKit prediction xlsx files for official Pass@K and Major@K evaluation from rollout caches."
        )
    )
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n<N>.jsonl file.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--k-list", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--rollout-n", type=int, default=128)
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
        help="Blank rollout candidates with generation errors instead of sending them to VLMEvalKit.",
    )
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


def write_prediction_xlsx(template, predictions: Dict[str, str], output_path: Path) -> int:
    data = template.copy()
    data["_index_str"] = data["index"].astype(str)
    data = data[data["_index_str"].isin(predictions)].copy()
    if data.empty:
        raise RuntimeError(f"No predictions match dataset indices for {output_path}")
    data["prediction"] = data["_index_str"].map(predictions).fillna("")
    data = data.drop(columns=["_index_str"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(data).to_excel(output_path, index=False)
    return len(data)


def rollout_answers_for_major(row: Dict[str, Any], k: int, skip_errors: bool) -> List[Tuple[int, str]]:
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list):
        return []

    answers = []
    for idx, rollout in enumerate(rollouts[:k]):
        if skip_errors and candidate_error(row, idx):
            continue
        answer = extract_final_answer(str(rollout or ""))
        if answer:
            answers.append((idx, answer))
    return answers


def build_major_predictions(rows: Iterable[Dict[str, Any]], k: int, skip_errors: bool) -> Dict[str, str]:
    predictions = {}
    for row in rows:
        index = str(row["index"])
        choice_map = parse_choice_map(row.get("question") or row.get("prompt") or "")
        answer, _, _, _, _ = majority_answer(rollout_answers_for_major(row, k, skip_errors), choice_map)
        predictions[index] = answer
    return predictions


def main():
    args = parse_args()
    k_values = parse_k_list(args.k_list)
    max_k = max(k_values)
    if max_k > args.rollout_n:
        raise ValueError(f"max K ({max_k}) cannot exceed --rollout-n ({args.rollout_n})")

    rollout_rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    if not rollout_rows:
        raise RuntimeError(f"no rollout rows found: {args.rollouts}")

    dataset_name = resolve_dataset_name(args.dataset)
    dataset_frame, dataset_source = load_dataset_frame(dataset_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "dataset": args.dataset,
        "vlmeval_dataset": dataset_name,
        "model_label": args.model_label,
        "rollouts": args.rollouts,
        "rollout_n": args.rollout_n,
        "k_list": k_values,
        "candidate_prediction_mode": args.candidate_prediction_mode,
        "skip_errors": bool(args.skip_errors),
        "dataset_source": dataset_source,
        "num_rollout_rows": len(rollout_rows),
        "candidate_files": [],
        "major_files": [],
    }

    indices = [str(row["index"]) for row in rollout_rows]
    for candidate_idx in range(max_k):
        predictions = {
            str(row["index"]): candidate_prediction(
                row,
                candidate_idx=candidate_idx,
                mode=args.candidate_prediction_mode,
                skip_errors=bool(args.skip_errors),
            )
            for row in rollout_rows
        }
        selected = output_dir / f"pass_candidate{candidate_idx:03d}_from_n{args.rollout_n}_selected.xlsx"
        num_written = write_prediction_xlsx(dataset_frame, predictions, selected)
        manifest["candidate_files"].append(
            {
                "candidate_idx": candidate_idx,
                "prediction": str(selected),
                "eval_output": str(selected.with_name(selected.name.replace("_selected.xlsx", "_eval.json"))),
                "num_written": num_written,
            }
        )

    for k in k_values:
        predictions = build_major_predictions(rollout_rows, k, bool(args.skip_errors))
        selected = output_dir / f"major{k}_from_n{args.rollout_n}_selected.xlsx"
        num_written = write_prediction_xlsx(dataset_frame, predictions, selected)
        manifest["major_files"].append(
            {
                "k": k,
                "prediction": str(selected),
                "eval_output": str(selected.with_name(selected.name.replace("_selected.xlsx", "_eval.json"))),
                "num_written": num_written,
            }
        )

    missing_from_dataset = sorted(set(indices) - set(dataset_frame["index"].astype(str)))
    manifest["num_missing_from_dataset"] = len(missing_from_dataset)
    manifest["missing_from_dataset"] = missing_from_dataset[:100]
    write_json(Path(args.manifest_output), manifest)
    print(f"wrote manifest: {args.manifest_output}")
    print(f"wrote {len(manifest['candidate_files'])} Pass@K candidate prediction files")
    print(f"wrote {len(manifest['major_files'])} Major@K prediction files")


if __name__ == "__main__":
    main()
