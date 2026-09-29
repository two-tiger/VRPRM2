import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import load_jsonl, resolve_dataset_name, sanitize_dataframe_for_excel, write_json


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Select Bo1 policy predictions directly from cached rollouts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n128.jsonl file.")
    parser.add_argument("--output-xlsx", required=True, help="VLMEvalKit-compatible prediction xlsx.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--candidate-idx", type=int, default=0, help="Rollout candidate index to evaluate as Bo1.")
    parser.add_argument(
        "--prediction-mode",
        choices=("raw", "final"),
        default="raw",
        help="raw keeps the full model response; final extracts a compact final answer when possible.",
    )
    parser.add_argument(
        "--fallback-first-valid",
        action="store_true",
        help="If candidate-idx is missing or errored, use the first non-empty non-errored rollout.",
    )
    return parser.parse_args()


def latest_rows_by_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "index" in row:
            latest[str(row["index"])] = row
    return list(latest.values())


def candidate_error(row: Dict[str, Any], candidate_idx: int) -> str:
    errors = row.get("errors")
    if isinstance(errors, list) and 0 <= candidate_idx < len(errors):
        return str(errors[candidate_idx] or "")
    return ""


def choose_rollout(row: Dict[str, Any], candidate_idx: int, fallback_first_valid: bool) -> tuple[Optional[int], str, str]:
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list):
        return None, "", "missing rollouts list"

    if 0 <= candidate_idx < len(rollouts):
        rollout = str(rollouts[candidate_idx] or "")
        error = candidate_error(row, candidate_idx)
        if rollout.strip() and not error:
            return candidate_idx, rollout, ""
        if not fallback_first_valid:
            return None, "", error or f"empty rollout at candidate_idx={candidate_idx}"

    if fallback_first_valid:
        for idx, rollout in enumerate(rollouts):
            rollout = str(rollout or "")
            error = candidate_error(row, idx)
            if rollout.strip() and not error:
                return idx, rollout, ""

    return None, "", f"no valid rollout found for candidate_idx={candidate_idx}"


def extract_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    answer_match = ANSWER_RE.search(text)
    if answer_match:
        answer = answer_match.group(1).strip()
        answer = re.sub(r"^\\boxed\{(.+)\}$", r"\1", answer).strip()
        return answer

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else text


def main():
    import pandas as pd
    from vlmeval.dataset import build_dataset

    args = parse_args()
    if args.candidate_idx < 0:
        raise ValueError("--candidate-idx must be >= 0")

    rollout_rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    selected: Dict[str, str] = {}
    summary_rows = []
    missing = []

    for row in rollout_rows:
        index = str(row["index"])
        chosen_idx, rollout, error = choose_rollout(
            row,
            candidate_idx=args.candidate_idx,
            fallback_first_valid=args.fallback_first_valid,
        )
        if chosen_idx is None:
            missing.append({"index": index, "error": error})
            continue

        prediction = extract_final_answer(rollout) if args.prediction_mode == "final" else rollout
        selected[index] = prediction
        summary_rows.append(
            {
                "index": index,
                "requested_candidate_idx": args.candidate_idx,
                "selected_candidate_idx": chosen_idx,
                "prediction_mode": args.prediction_mode,
                "num_rollouts": len(row.get("rollouts") or []),
            }
        )

    dataset_name = resolve_dataset_name(args.dataset)
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")

    data = dataset.data.copy()
    data["_index_str"] = data["index"].astype(str)
    data = data[data["_index_str"].isin(selected)].copy()
    if data.empty:
        raise RuntimeError("No Bo1 predictions match the VLMEvalKit dataset indices")
    data["prediction"] = data["_index_str"].map(selected)
    data = data.drop(columns=["_index_str"])

    output_xlsx = Path(args.output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(data).to_excel(output_xlsx, index=False)
    print(f"wrote Bo1 policy predictions: {output_xlsx}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        write_json(
            summary_path,
            {
                "dataset": args.dataset,
                "vlmeval_dataset": dataset_name,
                "candidate_idx": args.candidate_idx,
                "prediction_mode": args.prediction_mode,
                "num_selected": len(selected),
                "num_rollout_rows": len(rollout_rows),
                "num_missing": len(missing),
                "missing": missing[:100],
                "selected": summary_rows,
            },
        )
        print(f"wrote Bo1 selection summary: {summary_path}")


if __name__ == "__main__":
    main()
