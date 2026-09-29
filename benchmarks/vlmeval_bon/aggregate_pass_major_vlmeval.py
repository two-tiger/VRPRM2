import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate official VLMEvalKit Pass@K and Major@K results from evaluated prediction files."
    )
    parser.add_argument("--manifest", required=True, help="Manifest from build_pass_major_vlmeval_predictions.py.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--csv-output", default="")
    parser.add_argument(
        "--wemath-primary",
        choices=("strict", "loose", "row"),
        default="strict",
        help="Primary Pass/Major score for WeMath. Use strict to match VLMEvalKit Score (Strict).",
    )
    return parser.parse_args()


def parse_percent_or_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value) / 100.0 if float(value) > 1.0 else float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    return number / 100.0 if "%" in text or number > 1.0 else number


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    fallback = manifest_dir / path.name
    if fallback.exists():
        return fallback
    return path


def first_row(data: Any) -> Dict[str, Any]:
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def primary_metric_from_eval(eval_path: Path, dataset: str, wemath_primary: str) -> Tuple[str, Optional[float]]:
    if not eval_path.exists():
        return "missing_eval_json", None
    data = load_json(eval_path)
    rows = data if isinstance(data, list) else [data]

    if dataset == "WeMath":
        row = first_row(data)
        if wemath_primary == "strict":
            return "Score (Strict)", parse_percent_or_float(row.get("Score (Strict)"))
        if wemath_primary == "loose":
            return "Score (Loose)", parse_percent_or_float(row.get("Score (Loose)"))

    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("Task&Skill") == "Overall" and "acc" in row:
            return "Overall acc", parse_percent_or_float(row.get("acc"))
        if row.get("Subject") == "Overall" and "acc" in row:
            return "Overall acc", parse_percent_or_float(row.get("acc"))
        if "split" in row and "Overall" in row:
            return "Overall", parse_percent_or_float(row.get("Overall"))
        if "Overall" in row:
            return "Overall", parse_percent_or_float(row.get("Overall"))
        if "acc" in row:
            return "acc", parse_percent_or_float(row.get("acc"))

    row = first_row(data)
    for key in ("Score", "score", "Accuracy", "accuracy"):
        if key in row:
            return key, parse_percent_or_float(row.get(key))
    return "unknown", None


def candidate_scored_paths(prediction: Path) -> List[Path]:
    stem = prediction.with_suffix("")
    candidates = [prediction]
    for suffix in ("*.xlsx", "*.csv", "*.pkl"):
        candidates.extend(sorted(prediction.parent.glob(f"{stem.name}_*{suffix[1:]}")))
    filtered = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.name.endswith("_score.csv") or path.name.endswith("_eval.json"):
            continue
        filtered.append(path)
    return filtered


def load_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pkl":
        obj = pd.read_pickle(path)
        if hasattr(obj, "columns"):
            return obj
    return None


def lower_column_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).lower(): str(col) for col in columns}


def frame_has_usable_hits(frame) -> bool:
    columns = lower_column_map(frame.columns)
    if "index" not in columns:
        return False
    if "hit" in columns or "joker" in columns:
        return True
    if "score" in columns:
        return True
    if "res" in columns and "answer" in columns:
        return True
    return False


def find_scored_frame(prediction: Path):
    for path in candidate_scored_paths(prediction):
        try:
            frame = load_table(path)
        except Exception:
            continue
        if frame is None or not hasattr(frame, "columns"):
            continue
        if frame_has_usable_hits(frame):
            return frame, path
    return None, None


def bool_hit(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "correct"}:
            return True
        if text in {"0", "false", "no", "n", "incorrect", ""}:
            return False
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
    except TypeError:
        pass
    return bool(value)


def post_check_hit_map(frame, columns: Dict[str, str]) -> Dict[str, bool]:
    index_col = columns["index"]
    try:
        if "question_type" in columns or "answer_type" in columns:
            from vlmeval.dataset.utils.mathvista import post_check
        else:
            from vlmeval.dataset.utils.mathv import post_check
    except Exception as err:
        raise RuntimeError(f"failed to import VLMEvalKit post_check helper: {err}") from err

    hit_map = {}
    for _, row in frame.iterrows():
        try:
            hit = bool(post_check(row, prefetch=False))
        except Exception:
            hit = False
        hit_map[str(row[index_col])] = hit
    return hit_map


def hit_map_from_prediction(prediction: Path) -> Tuple[Dict[str, bool], Any, Path]:
    frame, scored_path = find_scored_frame(prediction)
    if frame is None or scored_path is None:
        raise RuntimeError(
            f"cannot find a VLMEvalKit scored sidecar with per-sample result columns for {prediction}. "
            "Run evaluate_vlmeval.py on this prediction file first."
        )
    columns = lower_column_map(frame.columns)
    index_col = columns["index"]
    hit_col = columns.get("hit", columns.get("joker", columns.get("score")))
    if hit_col is not None:
        hit_map = {
            str(index): bool_hit(hit)
            for index, hit in zip(frame[index_col], frame[hit_col])
        }
    elif "res" in columns and "answer" in columns:
        hit_map = post_check_hit_map(frame, columns)
    else:
        raise RuntimeError(
            f"cannot derive per-sample hits from VLMEvalKit sidecar {scored_path}; "
            f"columns={list(frame.columns)}"
        )
    return hit_map, frame, scored_path


def candidate_files_by_idx(manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["candidate_idx"]): item for item in manifest.get("candidate_files", [])}


def major_files_by_k(manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["k"]): item for item in manifest.get("major_files", [])}


def wemath_official_from_hits(frame, hit_map: Dict[str, bool], primary: str) -> Tuple[float, Dict[str, Any]]:
    import pandas as pd

    columns = lower_column_map(frame.columns)
    index_col = columns["index"]
    id_col = columns.get("id")
    key_col = columns.get("key")
    if id_col is None or key_col is None:
        raise RuntimeError("WeMath scored frame must contain ID/id and key columns")

    data = frame.copy()
    data["_hit_bool"] = [bool(hit_map.get(str(index), False)) for index in data[index_col]]
    data["_key"] = data[key_col].astype(str)

    def process_steps(steps: int):
        parts = {}
        for step_idx in range(1, steps + 1):
            sub = data[data["_key"] == f"{steps}steps_{step_idx}"][[id_col, "_hit_bool"]].copy()
            sub.columns = ["ID", f"joker_{step_idx}"]
            parts[step_idx] = sub
        multi = data[data["_key"] == f"{steps}steps_multi"][[id_col, "_hit_bool"]].copy()
        multi.columns = ["ID", "joker_multi"]

        merged = parts[1]
        for step_idx in range(2, steps + 1):
            merged = pd.merge(merged, parts[step_idx], on="ID", how="left")
        merged = pd.merge(merged, multi, on="ID", how="left")
        for col in [f"joker_{idx}" for idx in range(1, steps + 1)] + ["joker_multi"]:
            merged[col] = merged[col].fillna(False).astype(bool)
        return merged

    merged_2 = process_steps(2)
    merged_3 = process_steps(3)

    j21, j22, jm2 = merged_2["joker_1"], merged_2["joker_2"], merged_2["joker_multi"]
    j31, j32, j33, jm3 = (
        merged_3["joker_1"],
        merged_3["joker_2"],
        merged_3["joker_3"],
        merged_3["joker_multi"],
    )

    counts = {
        "InadequateGeneralization": int(((j21 & j22 & ~jm2).sum()) + ((j31 & j32 & j33 & ~jm3).sum())),
        "InsufficientKnowledge": int((((~j21 | ~j22) & ~jm2).sum()) + (((~j31 | ~j32 | ~j33) & ~jm3).sum())),
        "CompleteMastery_loose": int(((j21 | j22) & jm2).sum() + ((j31 | j32 | j33) & jm3).sum()),
        "CompleteMastery_strict": int((j21 & j22 & jm2).sum() + (j31 & j32 & j33 & jm3).sum()),
        "RoteMemorization_loose": int(((~j21 & ~j22) & jm2).sum() + ((~j31 & ~j32 & ~j33) & jm3).sum()),
        "RoteMemorization_strict": int(((~j21 | ~j22) & jm2).sum() + ((~j31 | ~j32 | ~j33) & jm3).sum()),
    }
    total_count = 525
    strict = (
        total_count
        - 0.5 * counts["InadequateGeneralization"]
        - counts["RoteMemorization_strict"]
        - counts["InsufficientKnowledge"]
    ) / total_count
    loose = (
        total_count
        - 0.5 * counts["InadequateGeneralization"]
        - counts["RoteMemorization_loose"]
        - counts["InsufficientKnowledge"]
    ) / total_count
    row_acc = sum(hit_map.values()) / len(hit_map) if hit_map else 0.0
    details = {
        **counts,
        "Score (Strict)": strict,
        "Score (Loose)": loose,
        "row_accuracy": row_acc,
        "total_count": total_count,
    }
    if primary == "strict":
        return strict, details
    if primary == "loose":
        return loose, details
    return row_acc, details


def mean_hit(hit_map: Dict[str, bool]) -> float:
    return sum(hit_map.values()) / len(hit_map) if hit_map else 0.0


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    manifest_dir = manifest_path.parent
    dataset = manifest["dataset"]
    model_label = manifest.get("model_label", "")
    k_values = [int(k) for k in manifest["k_list"]]
    candidates = candidate_files_by_idx(manifest)
    majors = major_files_by_k(manifest)

    max_k = max(k_values)
    candidate_hit_maps: Dict[int, Dict[str, bool]] = {}
    candidate_scored_paths: Dict[int, str] = {}
    template_frame = None
    template_scored_path = None

    for idx in range(max_k):
        item = candidates.get(idx)
        if item is None:
            raise RuntimeError(f"manifest is missing candidate index {idx}")
        prediction = resolve_manifest_path(item["prediction"], manifest_dir)
        hit_map, frame, scored_path = hit_map_from_prediction(prediction)
        candidate_hit_maps[idx] = hit_map
        candidate_scored_paths[idx] = str(scored_path)
        if template_frame is None:
            template_frame = frame
            template_scored_path = scored_path

    metrics_by_k = {}
    for k in k_values:
        all_indices = set()
        for idx in range(k):
            all_indices.update(candidate_hit_maps[idx])
        pass_hit_map = {
            index: any(candidate_hit_maps[idx].get(index, False) for idx in range(k))
            for index in all_indices
        }

        if dataset == "WeMath":
            pass_value, pass_details = wemath_official_from_hits(template_frame, pass_hit_map, args.wemath_primary)
            pass_metric_name = f"WeMath {args.wemath_primary}"
        else:
            pass_value = mean_hit(pass_hit_map)
            pass_details = {"row_accuracy": pass_value}
            pass_metric_name = "row official hit mean"

        major_item = majors.get(k)
        if major_item is None:
            raise RuntimeError(f"manifest is missing Major@{k} file")
        major_eval = resolve_manifest_path(major_item["eval_output"], manifest_dir)
        major_metric_name, major_value = primary_metric_from_eval(major_eval, dataset, args.wemath_primary)
        major_details: Dict[str, Any] = {}
        if major_value is None:
            major_prediction = resolve_manifest_path(major_item["prediction"], manifest_dir)
            major_hit_map, major_frame, _ = hit_map_from_prediction(major_prediction)
            if dataset == "WeMath":
                major_value, major_details = wemath_official_from_hits(major_frame, major_hit_map, args.wemath_primary)
                major_metric_name = f"WeMath {args.wemath_primary}"
            else:
                major_value = mean_hit(major_hit_map)
                major_details = {"row_accuracy": major_value}
                major_metric_name = "row official hit mean"

        metrics_by_k[str(k)] = {
            "pass_at_k": pass_value,
            "major_at_k": major_value,
            "pass_metric_name": pass_metric_name,
            "major_metric_name": major_metric_name,
            "num_samples": len(pass_hit_map),
            "pass_details": pass_details,
            "major_details": major_details,
        }

    summary = {
        "dataset": dataset,
        "vlmeval_dataset": manifest.get("vlmeval_dataset"),
        "model_label": model_label,
        "metric_mode": "vlmeval_official",
        "manifest": str(manifest_path),
        "wemath_primary": args.wemath_primary,
        "candidate_scored_paths": candidate_scored_paths,
        "template_scored_path": str(template_scored_path) if template_scored_path else "",
        "k": metrics_by_k,
        "definition": {
            "pass_at_k": (
                "uses official VLMEvalKit per-candidate hits; WeMath is aggregated with VLMEvalKit Score "
                f"({args.wemath_primary.title()}) by default"
            ),
            "major_at_k": "evaluates the majority-voted final answer file with VLMEvalKit and extracts the primary score",
        },
    }
    write_json(Path(args.summary_output), summary)

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
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
                item = metrics_by_k[str(k)]
                writer.writerow(
                    {
                        "model_label": model_label,
                        "dataset": dataset,
                        "metric_mode": "vlmeval_official",
                        "k": k,
                        "pass_at_k": item["pass_at_k"],
                        "major_at_k": item["major_at_k"],
                        "pass_metric_name": item["pass_metric_name"],
                        "major_metric_name": item["major_metric_name"],
                        "num_samples": item["num_samples"],
                    }
                )

    print(json.dumps(metrics_by_k, ensure_ascii=False, indent=2))
    print(f"wrote official Pass/Major summary: {args.summary_output}")
    if args.csv_output:
        print(f"wrote csv: {args.csv_output}")


if __name__ == "__main__":
    main()
