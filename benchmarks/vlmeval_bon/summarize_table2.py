import argparse
import json
from pathlib import Path
from typing import Any, Optional

from common import TABLE2_DATASETS, parse_dataset_list


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize BoN evaluation outputs into a Table-2-style CSV.")
    parser.add_argument("--root", required=True, help="Pipeline output root.")
    parser.add_argument("--models", required=True, help="Comma-separated model labels.")
    parser.add_argument("--datasets", default=",".join(TABLE2_DATASETS))
    parser.add_argument("--bon", type=int, default=8)
    parser.add_argument("--rollout-n", type=int, default=128)
    parser.add_argument("--reward-label", default="prm")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def find_number(obj: Any) -> Optional[float]:
    preferred = ["Overall", "overall", "acc", "accuracy", "score", "average", "Average"]
    if isinstance(obj, dict):
        for key in preferred:
            if key in obj:
                value = obj[key]
                if isinstance(value, (int, float)):
                    return float(value)
                nested = find_number(value)
                if nested is not None:
                    return nested
        for value in obj.values():
            nested = find_number(value)
            if nested is not None:
                return nested
    if isinstance(obj, list):
        for item in obj:
            nested = find_number(item)
            if nested is not None:
                return nested
    return None


def load_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        return find_number(json.loads(path.read_text(encoding="utf-8")))
    return None


def main():
    import pandas as pd

    args = parse_args()
    root = Path(args.root)
    model_labels = [item.strip() for item in args.models.split(",") if item.strip()]
    datasets = parse_dataset_list(args.datasets)

    rows = []
    for model in model_labels:
        row = {"model": model}
        scores = []
        for dataset in datasets:
            result_path = root / model / dataset / f"bon{args.bon}_{args.reward_label}_from_n{args.rollout_n}_eval.json"
            if not result_path.exists():
                result_path = root / model / dataset / f"bon{args.bon}_{args.reward_label}_eval.json"
            score = load_score(result_path)
            row[dataset] = score
            if score is not None:
                scores.append(score)
        row["Overall"] = sum(scores) / len(scores) if scores else None
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote summary: {out}")


if __name__ == "__main__":
    main()
