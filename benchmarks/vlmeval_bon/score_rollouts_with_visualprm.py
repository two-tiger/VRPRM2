import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from common import append_jsonl, env_default, load_jsonl, response_score, split_steps

VISUALPROCESSBENCH_DIR = Path(__file__).resolve().parents[1] / "visualprocessbench"
import sys

if str(VISUALPROCESSBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALPROCESSBENCH_DIR))

from visualprm_paper_eval import (  # noqa: E402
    load_image_for_visualprm,
    load_visualprm_model,
    score_visualprm_steps,
    to_jsonable,
)


REWARD_MODE = "visualprm_soft_v1"


def candidate_cache_path(score_output: Path) -> Path:
    return score_output.with_suffix(".candidates.jsonl")


def latest_rows_by_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "index" in row:
            latest[str(row["index"])] = row
    return latest


def load_score_candidates(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    candidates = {}
    for row in load_jsonl(path):
        if "index" not in row or "candidate_idx" not in row:
            continue
        candidates[(str(row["index"]), int(row["candidate_idx"]))] = row
    return candidates


def seed_score_candidates_from_outputs(rows: List[Dict[str, Any]], candidates: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    for row in rows:
        index = str(row.get("index"))
        for candidate in row.get("candidates", []):
            if "candidate_idx" not in candidate:
                continue
            key = (index, int(candidate["candidate_idx"]))
            if key in candidates:
                continue
            seeded = dict(candidate)
            seeded.update({
                "dataset": row.get("dataset"),
                "vlmeval_dataset": row.get("vlmeval_dataset"),
                "index": index,
                "policy_model": row.get("policy_model"),
                "bon": row.get("bon"),
                "reward_mode": candidate.get("reward_mode", row.get("reward_mode")),
            })
            candidates[key] = seeded


def score_candidate_done(row: Dict[str, Any]) -> bool:
    return (
        row.get("reward_mode") == REWARD_MODE
        and row.get("score") is not None
        and row.get("step_scores") is not None
        and not row.get("error")
    )


def aggregate_scored_item(item: Dict[str, Any], args, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: int(row["candidate_idx"]))
    valid = [candidate for candidate in candidates if score_candidate_done(candidate)]
    best = max(valid, key=lambda row: row["score"]) if valid else {
        "candidate_idx": -1,
        "rollout": "",
        "score": float("-inf"),
    }
    return {
        "dataset": item["dataset"],
        "vlmeval_dataset": item["vlmeval_dataset"],
        "index": str(item["index"]),
        "policy_model": item.get("policy_model"),
        "bon": args.bon,
        "reward_mode": REWARD_MODE,
        "visualprm_model_path": args.model_path,
        "best_idx": best["candidate_idx"],
        "best_score": best["score"],
        "best_rollout": best["rollout"],
        "num_valid_candidates": len(valid),
        "candidates": candidates,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score cached BoN rollouts with local OpenGVLab/VisualPRM-8B."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n<N>.jsonl file.")
    parser.add_argument("--score-output", required=True, help="JSONL output with aggregate per-sample VisualPRM scores.")
    parser.add_argument("--model-path", default=env_default("VISUALPRM_MODEL_PATH", None),
                        help="Local or HF path for OpenGVLab/VisualPRM-8B.")
    parser.add_argument("--bon", type=int, default=int(env_default("ROLLOUT_N", 128)),
                        help="Number of rollouts per sample to score.")
    parser.add_argument("--dtype", default=env_default("VISUALPRM_DTYPE", "bfloat16"),
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--threshold", type=float, default=float(env_default("VISUALPRM_THRESHOLD", 0.85)),
                        help="Paper threshold for recording binary step labels; selection uses soft scores.")
    parser.add_argument("--limit", type=int, default=int(env_default("VISUALPRM_LIMIT", 0)))
    parser.add_argument("--log-every", type=int, default=int(env_default("VISUALPRM_LOG_EVERY", 10)))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_path_for_item(item: Dict[str, Any]) -> Optional[str]:
    images = item.get("images") or item.get("image") or []
    if isinstance(images, str):
        return images
    if images:
        return str(images[0])
    return None


def score_rollout(tokenizer, model, args, item: Dict[str, Any], rollout: str, pixel_values) -> Dict[str, Any]:
    start = time.perf_counter()
    steps = split_steps(rollout)
    if not steps:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "prm_step_score": float("-inf"),
            "step_scores": None,
            "step_binary_scores": None,
            "raw_response": "",
            "error": "empty rollout",
            "reward_mode": REWARD_MODE,
            "latency": time.perf_counter() - start,
        }

    try:
        scores, raw_scores = score_visualprm_steps(
            tokenizer,
            model,
            item.get("question") or item["prompt"],
            steps,
            pixel_values,
        )
        raw_scores = to_jsonable(raw_scores)
        if len(scores) != len(steps):
            return {
                "rollout": rollout,
                "score": float("-inf"),
                "prm_step_score": float("-inf"),
                "step_scores": None,
                "step_binary_scores": None,
                "raw_response": raw_scores,
                "num_steps": len(steps),
                "error": f"score length mismatch: got {len(scores)}, expected {len(steps)}",
                "reward_mode": REWARD_MODE,
                "latency": time.perf_counter() - start,
            }
        score = response_score(scores)
        return {
            "rollout": rollout,
            "score": score,
            "prm_step_score": score,
            "step_scores": scores,
            "step_binary_scores": [1 if score_item > args.threshold else 0 for score_item in scores],
            "raw_response": raw_scores,
            "num_steps": len(steps),
            "error": "",
            "reward_mode": REWARD_MODE,
            "latency": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "prm_step_score": float("-inf"),
            "step_scores": None,
            "step_binary_scores": None,
            "raw_response": "",
            "error": repr(exc),
            "reward_mode": REWARD_MODE,
            "latency": time.perf_counter() - start,
        }


def score_item(tokenizer, model, args, item: Dict[str, Any], existing_candidates: Dict[Tuple[str, int], Dict[str, Any]], candidate_cache: Path) -> Dict[str, Any]:
    index = str(item["index"])
    rollouts = (item.get("rollouts") or [])[: args.bon]
    image_path = image_path_for_item(item)
    candidates = []

    if image_path is None:
        for candidate_idx, rollout in enumerate(rollouts):
            scored = {
                "candidate_idx": candidate_idx,
                "rollout": rollout,
                "score": float("-inf"),
                "prm_step_score": float("-inf"),
                "step_scores": None,
                "step_binary_scores": None,
                "raw_response": "",
                "error": "missing image",
                "reward_mode": REWARD_MODE,
            }
            candidates.append(scored)
        return aggregate_scored_item(item, args, candidates)

    try:
        pixel_values = load_image_for_visualprm(image_path).to(getattr(torch, args.dtype)).cuda()
    except Exception as exc:
        for candidate_idx, rollout in enumerate(rollouts):
            scored = {
                "candidate_idx": candidate_idx,
                "rollout": rollout,
                "score": float("-inf"),
                "prm_step_score": float("-inf"),
                "step_scores": None,
                "step_binary_scores": None,
                "raw_response": "",
                "error": f"image_load_error={repr(exc)}",
                "reward_mode": REWARD_MODE,
            }
            candidates.append(scored)
        return aggregate_scored_item(item, args, candidates)

    try:
        for candidate_idx, rollout in enumerate(rollouts):
            key = (index, candidate_idx)
            if key in existing_candidates and score_candidate_done(existing_candidates[key]):
                candidates.append(existing_candidates[key])
                continue
            scored = score_rollout(tokenizer, model, args, item, rollout, pixel_values)
            scored["candidate_idx"] = candidate_idx
            scored.update({
                "dataset": item["dataset"],
                "vlmeval_dataset": item["vlmeval_dataset"],
                "index": index,
                "policy_model": item.get("policy_model"),
                "bon": args.bon,
                "reward_mode": REWARD_MODE,
                "visualprm_model_path": args.model_path,
                "image": image_path,
            })
            append_jsonl(candidate_cache, scored)
            existing_candidates[key] = scored
            candidates.append(scored)
    finally:
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return aggregate_scored_item(item, args, candidates)


def main():
    args = parse_args()
    if not args.model_path:
        raise ValueError("--model-path or VISUALPRM_MODEL_PATH is required")
    if args.bon < 1:
        raise ValueError("--bon must be >= 1")

    score_output = Path(args.score_output)
    candidate_cache = candidate_cache_path(score_output)
    if args.overwrite:
        score_output.unlink(missing_ok=True)
        candidate_cache.unlink(missing_ok=True)

    existing_output_rows = load_jsonl(score_output)
    existing_outputs = latest_rows_by_index(existing_output_rows)
    existing_candidates = load_score_candidates(candidate_cache)
    seed_score_candidates_from_outputs(existing_output_rows, existing_candidates)
    already_scored = {
        str(index)
        for index, row in existing_outputs.items()
        if (
            len(row.get("candidates") or []) >= args.bon
            and all(score_candidate_done(candidate) for candidate in (row.get("candidates") or [])[: args.bon])
        )
    }

    rollout_rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    items = [row for index, row in rollout_rows.items() if str(index) not in already_scored]
    if args.limit:
        items = items[: args.limit]

    tokenizer, model = load_visualprm_model(args.model_path, args.dtype)

    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(items), desc=f"VisualPRM {args.dataset}", dynamic_ncols=True)
        except ImportError:
            progress = None

    for completed, item in enumerate(items, start=1):
        result = score_item(tokenizer, model, args, item, existing_candidates, candidate_cache)
        append_jsonl(score_output, result)
        if progress is not None:
            progress.update(1)
        elif completed % args.log_every == 0 or completed == len(items):
            print(f"[VisualPRM] {args.dataset}: {completed}/{len(items)}", flush=True)

    if progress is not None:
        progress.close()

    print(f"wrote VisualPRM score cache: {score_output}")
    print(f"candidate cache: {candidate_cache}")


if __name__ == "__main__":
    main()
