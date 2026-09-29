import argparse
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

from common import (
    SFT_PRM_SYSTEM_PROMPT,
    append_jsonl,
    env_default,
    load_jsonl,
    openai_content,
    parse_score_array,
    response_score,
    split_steps,
)


STEPWISE_WARMUP_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, prepare concise reasoning context for judging each solution step.

Output requirements:
- Output exactly one concise <think>...</think> block.
- Focus on the visual evidence, problem constraints, reference answer, and the key checks needed for the candidate solution.
- Do not output per-step scores.
- Do not output final JSON."""

THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
TOKEN_SCORE_RE = re.compile(r"[01]")


def candidate_cache_path(score_output: Path) -> Path:
    return score_output.with_suffix(".candidates.jsonl")


def latest_rows_by_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "index" in row:
            latest[str(row["index"])] = row
    return latest


def load_score_candidates(path: Path) -> Dict[tuple, Dict[str, Any]]:
    candidates = {}
    for row in load_jsonl(path):
        if "index" not in row or "candidate_idx" not in row:
            continue
        candidates[(str(row["index"]), int(row["candidate_idx"]))] = row
    return candidates


def seed_score_candidates_from_outputs(rows: List[Dict[str, Any]], candidates: Dict[tuple, Dict[str, Any]]) -> None:
    for row in rows:
        index = str(row.get("index"))
        for candidate in row.get("candidates", []):
            if "candidate_idx" not in candidate:
                continue
            candidate_idx = int(candidate["candidate_idx"])
            key = (index, candidate_idx)
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


def score_candidate_ok(row: Dict[str, Any], reward_mode: Optional[str] = None) -> bool:
    if row.get("error") or row.get("step_scores") is None:
        return False
    if reward_mode is None:
        return True
    existing_mode = row.get("reward_mode")
    if existing_mode != reward_mode:
        return existing_mode is None and reward_mode == "full"
    return True


def aggregate_scored_item(item: Dict[str, Any], args, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: int(row["candidate_idx"]))
    best = max(candidates, key=lambda row: row["score"]) if candidates else {
        "candidate_idx": -1,
        "rollout": "",
        "score": float("-inf"),
        "step_scores": None,
        "raw_response": "",
        "error": "no candidates",
    }
    return {
        "dataset": item["dataset"],
        "vlmeval_dataset": item["vlmeval_dataset"],
        "index": str(item["index"]),
        "policy_model": item.get("policy_model"),
        "bon": args.bon,
        "reward_mode": args.reward_mode,
        "best_idx": best["candidate_idx"],
        "best_score": best["score"],
        "best_rollout": best["rollout"],
        "candidates": candidates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Score cached BoN rollouts with a PRM endpoint.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n128.jsonl file.")
    parser.add_argument("--score-output", required=True, help="JSONL cache with per-candidate PRM scores.")
    parser.add_argument("--prm-base-url", default=env_default("PRM_BASE_URL", "http://127.0.0.1:8000/v1"),
                        help="One or more comma-separated OpenAI-compatible PRM endpoints.")
    parser.add_argument("--prm-api-key", default=env_default("PRM_API_KEY", "EMPTY"))
    parser.add_argument("--prm-model", default=env_default("PRM_MODEL", "auto"),
                        help="One model name or comma-separated model names aligned with --prm-base-url.")
    parser.add_argument("--bon", type=int, default=int(env_default("ROLLOUT_N", 128)),
                        help="Number of rollouts per sample to score.")
    parser.add_argument("--concurrency", type=int, default=int(env_default("PRM_CONCURRENCY", 4)))
    parser.add_argument("--max-tokens", type=int, default=int(env_default("PRM_MAX_TOKENS", 2048)))
    parser.add_argument("--warmup-max-tokens", type=int, default=int(env_default("PRM_WARMUP_MAX_TOKENS", 1024)))
    parser.add_argument("--temperature", type=float, default=float(env_default("PRM_TEMPERATURE", 0.0)))
    parser.add_argument("--reward-mode", choices=("stepwise", "full"), default=env_default("PRM_REWARD_MODE", "stepwise"),
                        help="stepwise uses one-token per-step scores; full parses one Score JSON response.")
    parser.add_argument("--guided-choice", type=int, choices=(0, 1), default=int(env_default("PRM_GUIDED_CHOICE", 1)),
                        help="Use vLLM guided_choice for stepwise one-token scores. Disable for SGLang if unsupported.")
    parser.add_argument("--max-retries", type=int, default=int(env_default("PRM_MAX_RETRIES", 3)))
    parser.add_argument("--request-timeout", type=float, default=float(env_default("PRM_REQUEST_TIMEOUT", 300.0)),
                        help="Per-request timeout in seconds. Set <=0 to disable.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv_arg(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


async def resolve_prm_models(clients: List[AsyncOpenAI], model_arg: str) -> List[str]:
    models = split_csv_arg(model_arg)
    if len(models) == 1 and models[0] == "auto":
        resolved = []
        for client in clients:
            listing = await client.models.list()
            if not listing.data:
                raise RuntimeError("PRM endpoint returned no models")
            resolved.append(listing.data[0].id)
        return resolved
    if len(models) == 1 and len(clients) > 1:
        return models * len(clients)
    if len(models) != len(clients):
        raise ValueError("--prm-model must be one value, auto, or aligned with --prm-base-url")
    return models


def prm_user_text(question: str, steps: List[str], answer: Any) -> str:
    return (
        f"[Question]\n{question}\n"
        f"[Solution]\n{'<step split>'.join(steps)}\n"
        f"[Answer]\n{answer if answer is not None else 'None'}\n"
    )


async def await_with_timeout(request, args):
    if args.request_timeout > 0:
        return await asyncio.wait_for(request, timeout=args.request_timeout)
    return await request


def extract_think(text: str) -> str:
    match = THINK_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def parse_token_score(text: str) -> int:
    match = TOKEN_SCORE_RE.search(text or "")
    if not match:
        return -2
    return int(match.group(0))


def build_stepwise_seed_prompt(item: Dict[str, Any], think_text: str) -> str:
    prompt = ""
    prompt += f"[Question]\n{item.get('question') or item['prompt']}\n"
    prompt += f"[Answer]\n{item.get('answer') if item.get('answer') is not None else 'None'}\n"
    if think_text:
        prompt += f"<think>{think_text}</think>\n"
    prompt += "[Step judgment]\n"
    return prompt


async def get_step_score(client: AsyncOpenAI, model: str, args, prompt: str):
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            request_kwargs = {}
            if args.guided_choice:
                request_kwargs["extra_body"] = {"guided_choice": ["1", "0"]}
            request = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=1,
                **request_kwargs,
            )
            response = await await_with_timeout(request, args)
            text = response.choices[0].text or ""
            score = parse_token_score(text)
            if score in (0, 1):
                attempts.append({"attempt": attempt + 1, "raw": text, "error": None})
                return score, text, None, attempts
            error = f"Could not parse guided score token from {text!r}"
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        await asyncio.sleep(1.5 * (attempt + 1))
    return -2, text, error, attempts


async def score_rollout_full(client: AsyncOpenAI, model: str, args, item: Dict[str, Any], rollout: str) -> Dict[str, Any]:
    steps = split_steps(rollout)
    if not steps:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "step_scores": None,
            "raw_response": "",
            "error": "empty rollout",
            "reward_mode": args.reward_mode,
        }

    content = openai_content(prm_user_text(item.get("question") or item["prompt"], steps, item.get("answer")), item["images"])
    try:
        request = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SFT_PRM_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        response = await await_with_timeout(request, args)
        raw = response.choices[0].message.content or ""
        scores = parse_score_array(raw, len(steps))
        return {
            "rollout": rollout,
            "score": response_score(scores),
            "step_scores": scores,
            "raw_response": raw,
            "error": "" if scores is not None else "failed to parse Score JSON",
            "reward_mode": args.reward_mode,
        }
    except Exception as exc:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "step_scores": None,
            "raw_response": "",
            "error": repr(exc),
            "reward_mode": args.reward_mode,
        }


async def score_rollout_stepwise(client: AsyncOpenAI, model: str, args, item: Dict[str, Any], rollout: str) -> Dict[str, Any]:
    steps = split_steps(rollout)
    if not steps:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "step_scores": None,
            "raw_response": "",
            "error": "empty rollout",
            "reward_mode": args.reward_mode,
        }

    content = openai_content(prm_user_text(item.get("question") or item["prompt"], steps, item.get("answer")), item["images"])
    warmup_text = ""
    warmup_error = None
    warmup_attempts = []
    for attempt in range(args.max_retries):
        try:
            request = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STEPWISE_WARMUP_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=args.temperature,
                max_tokens=args.warmup_max_tokens,
            )
            response = await await_with_timeout(request, args)
            warmup_text = response.choices[0].message.content or ""
            warmup_error = None
            warmup_attempts.append({"attempt": attempt + 1, "max_tokens": args.warmup_max_tokens, "error": None})
            break
        except Exception as exc:
            warmup_error = repr(exc)
            warmup_attempts.append({"attempt": attempt + 1, "max_tokens": args.warmup_max_tokens, "error": warmup_error})
            await asyncio.sleep(1.5 * (attempt + 1))

    prompt = build_stepwise_seed_prompt(item, extract_think(warmup_text))
    step_scores = []
    step_raw_responses = []
    step_errors = []
    step_attempts = []
    for step_idx, step in enumerate(steps):
        prompt += f"Step {step_idx}: {step}\\boxed{{"
        score, raw_text, step_error, attempts = await get_step_score(client, model, args, prompt)
        step_scores.append(score)
        step_raw_responses.append(raw_text)
        step_errors.append(step_error)
        step_attempts.append(attempts)
        prompt += str(score if score in (0, 1) else 0)
        prompt += "}\n"

    failed_steps = [i for i, value in enumerate(step_scores) if value not in (0, 1)]
    error_parts = []
    if warmup_error:
        error_parts.append(f"warmup_error={warmup_error}")
    if failed_steps:
        error_parts.append(f"failed_steps={failed_steps}")
    error = "; ".join(error_parts)
    ok_scores = step_scores if not failed_steps else None
    return {
        "rollout": rollout,
        "score": response_score(ok_scores),
        "step_scores": ok_scores,
        "raw_response": warmup_text,
        "step_raw_responses": step_raw_responses,
        "step_attempts": step_attempts,
        "step_errors": step_errors,
        "warmup_attempts": warmup_attempts,
        "error": error,
        "reward_mode": args.reward_mode,
    }


async def score_rollout(client: AsyncOpenAI, model: str, args, item: Dict[str, Any], rollout: str) -> Dict[str, Any]:
    if args.reward_mode == "full":
        return await score_rollout_full(client, model, args, item, rollout)
    return await score_rollout_stepwise(client, model, args, item, rollout)


async def score_item_resumable(
    clients: List[AsyncOpenAI],
    models: List[str],
    args,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    candidate_cache: Path,
    existing_candidates: Dict[tuple, Dict[str, Any]],
) -> Dict[str, Any]:
    index = str(item["index"])
    rollouts = (item.get("rollouts") or [])[: args.bon]

    async def guarded(candidate_idx: int, rollout: str):
        key = (index, candidate_idx)
        if key in existing_candidates and score_candidate_ok(existing_candidates[key], args.reward_mode):
            return existing_candidates[key]
        async with semaphore:
            endpoint_idx = candidate_idx % len(clients)
            scored = await score_rollout(clients[endpoint_idx], models[endpoint_idx], args, item, rollout)
            scored["candidate_idx"] = candidate_idx
            scored.update({
                "dataset": item["dataset"],
                "vlmeval_dataset": item["vlmeval_dataset"],
                "index": index,
                "policy_model": item.get("policy_model"),
                "bon": args.bon,
                "reward_mode": args.reward_mode,
            })
            append_jsonl(candidate_cache, scored)
            existing_candidates[key] = scored
            return scored

    scored = await asyncio.gather(*[
        guarded(candidate_idx, rollout)
        for candidate_idx, rollout in enumerate(rollouts)
    ])
    return aggregate_scored_item(item, args, scored)


async def run(args):
    if args.bon < 1:
        raise ValueError("--bon must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.warmup_max_tokens < 1:
        raise ValueError("--warmup-max-tokens must be >= 1")

    score_output = Path(args.score_output)
    candidate_cache = candidate_cache_path(score_output)
    if args.overwrite and score_output.exists():
        score_output.unlink()
    if args.overwrite and candidate_cache.exists():
        candidate_cache.unlink()

    existing_output_rows = load_jsonl(score_output)
    existing_outputs = latest_rows_by_index(existing_output_rows)
    existing_candidates = load_score_candidates(candidate_cache)
    seed_score_candidates_from_outputs(existing_output_rows, existing_candidates)
    already_scored = {
        str(index)
        for index, row in existing_outputs.items()
        if (
            len(row.get("candidates") or []) >= args.bon
            and all(score_candidate_ok(candidate, args.reward_mode) for candidate in (row.get("candidates") or [])[: args.bon])
        )
    }
    rollout_rows = latest_rows_by_index(load_jsonl(Path(args.rollouts)))
    items = [row for index, row in rollout_rows.items() if str(index) not in already_scored]

    base_urls = split_csv_arg(args.prm_base_url)
    client_kwargs = {"api_key": args.prm_api_key}
    if args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout + 10
    clients = [AsyncOpenAI(base_url=url, **client_kwargs) for url in base_urls]
    models = await resolve_prm_models(clients, args.prm_model)

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = None if args.no_progress else tqdm(total=len(items), desc=f"score {args.dataset}", dynamic_ncols=True)
    try:
        completed = 0
        tasks = [
            score_item_resumable(clients, models, args, item, semaphore, candidate_cache, existing_candidates)
            for item in items
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            append_jsonl(score_output, result)
            completed += 1
            if progress is not None:
                progress.update(1)
            else:
                print(f"[score] {args.dataset}: {completed}/{len(items)}", flush=True)
    finally:
        if progress is not None:
            progress.close()

    print(f"wrote PRM score cache: {score_output}")


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
