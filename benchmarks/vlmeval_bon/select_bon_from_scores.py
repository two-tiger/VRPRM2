import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import LOCAL_LMU_DATA, load_jsonl, resolve_dataset_name, sanitize_dataframe_for_excel, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Select BoN predictions from a cached PRM score JSONL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--score-output", required=True, help="JSONL produced by score_rollouts_with_prm.py.")
    parser.add_argument("--output-xlsx", required=True, help="VLMEvalKit-compatible prediction xlsx.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--bon", type=int, required=True, help="Use only the first N rollout candidates.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Deprecated; partial PRM score caches are allowed by default.")
    parser.add_argument("--strict-complete", action="store_true",
                        help="Require every sample to have N scored candidates before selection.")
    return parser.parse_args()


def candidate_rank(candidate: Dict[str, Any]) -> int:
    try:
        return int(candidate.get("candidate_idx", 0))
    except (TypeError, ValueError):
        return 0


def select_best(candidates: List[Dict[str, Any]], bon: int) -> Optional[Dict[str, Any]]:
    eligible = [
        row for row in candidates
        if candidate_rank(row) < bon and candidate_scored(row)
    ]
    if not eligible:
        return None
    return max(eligible, key=candidate_score)


def candidate_scored(candidate: Dict[str, Any]) -> bool:
    return candidate_score(candidate) is not None


def candidate_score(candidate: Dict[str, Any]) -> Optional[float]:
    score = candidate.get("prm_step_score", candidate.get("score"))
    if score is None:
        return None
    score = float(score)
    if math.isnan(score):
        return None
    return score


def candidate_cached(candidate: Dict[str, Any]) -> bool:
    return candidate_scored(candidate) or candidate.get("step_scores") is not None


def latest_rows_by_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "index" in row:
            latest[str(row["index"])] = row
    return list(latest.values())


def load_local_dataset_frame(dataset_name: str):
    import pandas as pd

    candidates = [
        LOCAL_LMU_DATA / f"{dataset_name}.tsv",
        LOCAL_LMU_DATA / "_downloads" / f"{dataset_name}.tsv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep="\t"), path
    return None, None


def load_dataset_frame(dataset_name: str):
    local_data, local_path = load_local_dataset_frame(dataset_name)
    if local_data is not None:
        return local_data, f"local TSV: {local_path}"

    from vlmeval.dataset import build_dataset

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")
    return dataset.data.copy(), "VLMEvalKit build_dataset"


def main():
    args = parse_args()
    if args.bon < 1:
        raise ValueError("--bon must be >= 1")

    score_rows = latest_rows_by_index(load_jsonl(Path(args.score_output)))
    selected: Dict[str, str] = {}
    summary_rows = []
    incomplete = []
    for row in score_rows:
        index = str(row["index"])
        candidates = sorted(row.get("candidates", []), key=candidate_rank)
        cached_candidates = [
            c for c in candidates
            if candidate_rank(c) < args.bon and candidate_cached(c)
        ]
        if args.strict_complete and len(cached_candidates) < args.bon:
            incomplete.append(index)
            continue
        best = select_best(candidates, args.bon)
        if best is None:
            incomplete.append(index)
            continue
        selected[index] = best.get("rollout", "")
        summary_rows.append({
            "index": index,
            "bon": args.bon,
            "best_idx": best.get("candidate_idx"),
            "best_score": candidate_score(best),
            "num_cached_candidates": len(cached_candidates),
        })

    if incomplete and args.strict_complete:
        preview = ", ".join(incomplete[:10])
        raise RuntimeError(
            f"{len(incomplete)} samples have fewer than {args.bon} cached scores. "
            f"Examples: {preview}. Remove --strict-complete to ignore this check."
        )

    dataset_name = resolve_dataset_name(args.dataset)
    data, data_source = load_dataset_frame(dataset_name)
    data["_index_str"] = data["index"].astype(str)
    data = data[data["_index_str"].isin(selected)].copy()
    if data.empty:
        raise RuntimeError("No scored predictions match the VLMEvalKit dataset indices")
    data["prediction"] = data["_index_str"].map(selected)
    data = data.drop(columns=["_index_str"])

    output_xlsx = Path(args.output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(data).to_excel(output_xlsx, index=False)
    print(f"wrote Bo{args.bon} selected predictions: {output_xlsx}")
    print(f"dataset source: {data_source}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        write_json(summary_path, {
            "dataset": args.dataset,
            "vlmeval_dataset": dataset_name,
            "dataset_source": data_source,
            "bon": args.bon,
            "num_selected": len(selected),
            "num_score_rows": len(score_rows),
            "num_incomplete": len(incomplete),
            "selected": summary_rows,
        })
        print(f"wrote Bo{args.bon} selection summary: {summary_path}")


if __name__ == "__main__":
    main()
