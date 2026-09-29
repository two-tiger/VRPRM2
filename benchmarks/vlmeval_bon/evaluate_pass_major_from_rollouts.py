import argparse
import csv
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import load_jsonl, write_json


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
LETTER_RE = re.compile(r"^[\(\[\{]?\s*([A-I])\s*[\)\]\}]?\.?$", re.IGNORECASE)
PAREN_CHOICE_RE = re.compile(r"\(([A-I])\)\s*(.+?)(?=\s*\([A-I]\)\s*|\Z)", re.IGNORECASE | re.DOTALL)
LINE_CHOICE_RE = re.compile(r"(?:^|\n)\s*([A-I])[\)\].:]\s*(.+?)(?=\n\s*[A-I][\)\].:]|\Z)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][-+]?\d+)?")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Pass@K and Major@K directly from cached rollout candidates."
    )
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n<N>.jsonl file.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--k-list", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--summary-output", required=True, help="Per-dataset metrics JSON.")
    parser.add_argument("--details-output", required=True, help="Per-sample JSONL details.")
    parser.add_argument("--csv-output", default="", help="Optional per-dataset metrics CSV.")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Only inspect this many rollout candidates; 0 means all available.")
    parser.add_argument("--skip-errors", type=int, choices=(0, 1), default=1,
                        help="Ignore rollout candidates with non-empty generation errors.")
    parser.add_argument("--numeric-tol", type=float, default=1e-6)
    return parser.parse_args()


def parse_k_list(raw: str) -> List[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"K must be >= 1: {value}")
        values.append(value)
    if not values:
        raise ValueError("--k-list must contain at least one K")
    return sorted(set(values))


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


def extract_final_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    boxed = BOXED_RE.findall(text)
    if boxed:
        return cleanup_answer(boxed[-1])

    answer_match = ANSWER_RE.search(text)
    if answer_match:
        return cleanup_answer(answer_match.group(1))

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return ""
    return cleanup_answer(non_empty_lines[-1])


def cleanup_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\\boxed\{(.+)\}$", r"\1", text).strip()
    text = re.sub(r"^['\"`]+|['\"`]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_choice_map(question: str) -> Dict[str, str]:
    text = str(question or "")
    starts = []
    paren_match = re.search(r"\([A-I]\)", text, re.IGNORECASE)
    line_match = re.search(r"(?:^|\n)\s*[A-I][\)\].:]\s+", text, re.IGNORECASE)
    if paren_match:
        starts.append(paren_match.start())
    if line_match:
        starts.append(line_match.start())
    if starts:
        text = text[min(starts):]

    choices = {}
    for letter, value in PAREN_CHOICE_RE.findall(text):
        value = value.strip()
        value = re.sub(r"\s+", " ", value)
        choices[letter.upper()] = value
    if choices:
        return choices

    for letter, value in LINE_CHOICE_RE.findall(text):
        value = value.strip()
        value = re.sub(r"\s+", " ", value)
        choices[letter.upper()] = value
    return choices


def extract_letter(answer: str) -> str:
    answer = cleanup_answer(answer)
    match = LETTER_RE.match(answer)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([A-I])\s*[\)\].:,-]\s+.+", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def normalize_text(text: str) -> str:
    text = cleanup_answer(text).lower()
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(string.punctuation + " ")


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def number_value(text: str) -> Optional[float]:
    text = cleanup_answer(text).replace(",", "")
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def answer_keys(answer: str, choice_map: Dict[str, str]) -> List[str]:
    answer = cleanup_answer(answer)
    keys: List[str] = []
    letter = extract_letter(answer)
    if letter:
        keys.append(f"letter:{letter}")
        if letter in choice_map:
            keys.append(f"text:{compact_text(choice_map[letter])}")

    compact = compact_text(answer)
    if compact:
        keys.append(f"text:{compact}")
        for choice_letter, choice_text in choice_map.items():
            if compact == compact_text(choice_text):
                keys.append(f"letter:{choice_letter}")

    numeric = number_value(answer)
    if numeric is not None and math.isfinite(numeric):
        keys.append(f"num:{numeric:.12g}")

    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def is_correct(prediction: str, reference: Any, choice_map: Dict[str, str], numeric_tol: float) -> bool:
    pred = cleanup_answer(prediction)
    ref = cleanup_answer("" if reference is None else str(reference))
    if not pred or not ref:
        return False

    pred_keys = set(answer_keys(pred, choice_map))
    ref_keys = set(answer_keys(ref, choice_map))
    if pred_keys and ref_keys and pred_keys.intersection(ref_keys):
        return True

    pred_num = number_value(pred)
    ref_num = number_value(ref)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) <= numeric_tol * max(1.0, abs(ref_num))

    return compact_text(pred) == compact_text(ref)


def vote_key(answer: str, choice_map: Dict[str, str]) -> str:
    keys = answer_keys(answer, choice_map)
    for key in keys:
        if key.startswith("letter:"):
            return key
    for key in keys:
        if key.startswith("text:"):
            return key
    for key in keys:
        if key.startswith("num:"):
            return key
    return f"text:{compact_text(answer)}"


def valid_rollouts(row: Dict[str, Any], max_candidates: int, skip_errors: bool) -> List[Tuple[int, str]]:
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list):
        return []
    limit = len(rollouts) if max_candidates <= 0 else min(max_candidates, len(rollouts))
    valid = []
    for idx, rollout in enumerate(rollouts[:limit]):
        rollout = str(rollout or "")
        if not rollout.strip():
            continue
        if skip_errors and candidate_error(row, idx):
            continue
        valid.append((idx, rollout))
    return valid


def majority_answer(answers: Iterable[Tuple[int, str]], choice_map: Dict[str, str]) -> Tuple[str, str, int, int, bool]:
    counts: Counter[str] = Counter()
    first_seen: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for order, answer in answers:
        answer = cleanup_answer(answer)
        if not answer:
            continue
        key = vote_key(answer, choice_map)
        if not key or key == "text:":
            continue
        counts[key] += 1
        first_seen.setdefault(key, order)
        display.setdefault(key, answer)

    if not counts:
        return "", "", 0, 0, False

    max_count = max(counts.values())
    tied = [key for key, count in counts.items() if count == max_count]
    best_key = min(tied, key=lambda key: first_seen[key])
    return display[best_key], best_key, max_count, len(counts), len(tied) > 1


def evaluate_row(row: Dict[str, Any], k_values: List[int], args) -> Dict[str, Any]:
    choice_map = parse_choice_map(row.get("question") or row.get("prompt") or "")
    reference = row.get("answer")
    candidates = valid_rollouts(row, args.max_candidates, bool(args.skip_errors))
    extracted = []
    for candidate_idx, rollout in candidates:
        answer = extract_final_answer(rollout)
        correct = is_correct(answer, reference, choice_map, args.numeric_tol)
        extracted.append({
            "candidate_idx": candidate_idx,
            "answer": answer,
            "vote_key": vote_key(answer, choice_map) if answer else "",
            "correct": correct,
        })

    by_k = {}
    for k in k_values:
        subset = [item for item in extracted if item["candidate_idx"] < k]
        pass_correct = any(item["correct"] for item in subset)
        major_answer, major_key, vote_count, num_unique, tied = majority_answer(
            [(item["candidate_idx"], item["answer"]) for item in subset],
            choice_map,
        )
        major_correct = is_correct(major_answer, reference, choice_map, args.numeric_tol)
        by_k[str(k)] = {
            "pass_correct": pass_correct,
            "major_correct": major_correct,
            "major_answer": major_answer,
            "major_vote_key": major_key,
            "major_vote_count": vote_count,
            "num_unique_votes": num_unique,
            "major_tie": tied,
            "num_candidates_used": len(subset),
            "num_correct_candidates": sum(1 for item in subset if item["correct"]),
        }

    return {
        "index": str(row.get("index")),
        "dataset": row.get("dataset"),
        "vlmeval_dataset": row.get("vlmeval_dataset"),
        "dataset_type": row.get("dataset_type"),
        "reference": "" if reference is None else str(reference),
        "choice_map": choice_map,
        "num_rollouts": len(row.get("rollouts") or []),
        "num_valid_rollouts": len(candidates),
        "by_k": by_k,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, metrics: Dict[str, Any], k_values: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_label", "dataset", "metric_mode", "k", "pass_at_k", "major_at_k", "num_samples"],
        )
        writer.writeheader()
        for k in k_values:
            item = metrics["k"][str(k)]
            writer.writerow({
                "model_label": metrics.get("model_label", ""),
                "dataset": metrics["dataset"],
                "metric_mode": metrics["metric_mode"],
                "k": k,
                "pass_at_k": item["pass_at_k"],
                "major_at_k": item["major_at_k"],
                "num_samples": metrics["num_samples"],
            })


def main():
    args = parse_args()
    k_values = parse_k_list(args.k_list)
    rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    details = [evaluate_row(row, k_values, args) for row in rows]
    total = len(details)
    if total == 0:
        raise RuntimeError(f"no rollout rows found: {args.rollouts}")

    metrics_by_k = {}
    for k in k_values:
        key = str(k)
        pass_hits = sum(1 for row in details if row["by_k"][key]["pass_correct"])
        major_hits = sum(1 for row in details if row["by_k"][key]["major_correct"])
        metrics_by_k[key] = {
            "pass_at_k": pass_hits / total,
            "major_at_k": major_hits / total,
            "pass_correct": pass_hits,
            "major_correct": major_hits,
        }

    metrics = {
        "dataset": args.dataset,
        "model_label": args.model_label,
        "metric_mode": "local_exact",
        "rollouts": args.rollouts,
        "num_samples": total,
        "k": metrics_by_k,
        "definition": {
            "pass_at_k": (
                "local exact/numeric/choice matching only; for datasets such as WeMath this is not the "
                "official VLMEvalKit aggregate score"
            ),
            "major_at_k": (
                "local exact/numeric/choice matching on the majority-voted final answer only; for datasets "
                "such as WeMath this is not the official VLMEvalKit aggregate score"
            ),
            "major_tie_break": "ties are broken by the earliest candidate order among tied answers",
        },
    }

    write_json(Path(args.summary_output), metrics)
    write_jsonl(Path(args.details_output), details)
    if args.csv_output:
        write_metrics_csv(Path(args.csv_output), metrics, k_values)
    print(json.dumps(metrics_by_k, ensure_ascii=False, indent=2))
    print(f"wrote metrics: {args.summary_output}")
    print(f"wrote details: {args.details_output}")
    if args.csv_output:
        print(f"wrote csv: {args.csv_output}")


if __name__ == "__main__":
    main()
