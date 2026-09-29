import argparse
import asyncio
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

from common import append_jsonl, env_default, load_jsonl, openai_content, response_score, split_steps


GLOBAL_THINK_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, think before judging individual steps.

Output requirements:
- Output exactly one <think>...</think> block.
- In the thinking, identify the key visual evidence, the question goal, the reference-answer constraint, dependencies between steps, and likely first consequential error if any.
- You may include a compact step-level correctness overview if it helps make later step scoring consistent.
- Do not output text outside the <think>...</think> block."""


STEP_SCORE_SYSTEM_PROMPT = """You are a visual process reward model. Judge whether the current candidate solution step is correct in context.

Use the global thinking and previous step judgments as context.
Output only one token: 1 or 0.

Scoring policy:
- Output 1 only if the current step is fully supported by the image/problem context and remains logically and mathematically correct.
- Output 0 if the step is unsupported, visually mistaken, logically invalid, computationally wrong, contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery."""


THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
TOKEN_SCORE_RE = re.compile(r"[01]")
ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
REWARD_MODE = "global_think_stepwise_logprob_final_answer_v1"


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
            if key not in candidates:
                candidates[key] = dict(candidate)


def score_candidate_ok(row: Dict[str, Any]) -> bool:
    return (
        row.get("reward_mode") == REWARD_MODE
        and row.get("score") is not None
        and row.get("step_scores") is not None
    )


def score_candidate_done(row: Dict[str, Any]) -> bool:
    return (
        row.get("reward_mode") == REWARD_MODE
        and row.get("step_scores") is not None
    )


def split_csv_arg(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


async def await_with_timeout(request, args):
    if args.request_timeout > 0:
        return await asyncio.wait_for(request, timeout=args.request_timeout)
    return await request


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


def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_logprob_token(token: Any) -> str:
    return str(token or "").strip()


def extract_choice_logprobs(response: Any) -> Dict[str, float]:
    choice = response.choices[0]
    logprobs = get_attr_or_key(choice, "logprobs")
    content = get_attr_or_key(logprobs, "content", []) if logprobs is not None else []
    if not content:
        return {}
    first = content[0]
    values: Dict[str, float] = {}

    token = normalize_logprob_token(get_attr_or_key(first, "token"))
    logprob = get_attr_or_key(first, "logprob")
    if token in {"0", "1"} and logprob is not None:
        values[token] = float(logprob)

    for item in get_attr_or_key(first, "top_logprobs", []) or []:
        token = normalize_logprob_token(get_attr_or_key(item, "token"))
        logprob = get_attr_or_key(item, "logprob")
        if token in {"0", "1"} and logprob is not None:
            values[token] = float(logprob)
    return values


def logprob_score_for_one(response: Any, hard_pred: int) -> Tuple[float, Dict[str, float]]:
    logprobs = extract_choice_logprobs(response)
    lp0 = logprobs.get("0")
    lp1 = logprobs.get("1")
    if lp0 is not None and lp1 is not None:
        max_lp = max(lp0, lp1)
        p0 = math.exp(lp0 - max_lp)
        p1 = math.exp(lp1 - max_lp)
        return p1 / (p0 + p1), logprobs
    if hard_pred in (0, 1):
        return float(hard_pred), logprobs
    return float("-inf"), logprobs


def has_required_final_answer_format(text: str) -> bool:
    return bool(ANSWER_LINE_RE.search(text or "")) and "\\boxed{" in (text or "")


def extract_boxed_answers(text: str) -> List[str]:
    values = []
    raw = text or ""
    marker = "\\boxed{"
    start = 0
    while True:
        idx = raw.find(marker, start)
        if idx < 0:
            break
        pos = idx + len(marker)
        depth = 1
        chars = []
        while pos < len(raw) and depth > 0:
            ch = raw[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        if depth == 0:
            values.append("".join(chars).strip())
        start = max(pos, idx + len(marker))
    return values


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answers(text or "")
    if boxed:
        return boxed[-1].strip()
    match = ANSWER_LINE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return ""


def clean_answer_text(text: Any) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.strip().strip(" .,:;，。；：")
    return cleaned


def extract_option_letter(text: Any) -> str:
    cleaned = clean_answer_text(text).upper()
    match = re.fullmatch(r"\(?([A-Z])\)?[\).:]?", cleaned)
    if match:
        return match.group(1)
    match = re.match(r"^([A-Z])\s*[\).:]", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:OPTION|LETTER)\s*([A-Z])\b", cleaned)
    return match.group(1) if match else ""


def evaluate_final_answer(rollout: str, reference_answer: Any) -> Dict[str, Any]:
    extracted = extract_final_answer(rollout)
    has_format = has_required_final_answer_format(rollout)
    reference_letter = extract_option_letter(reference_answer)
    extracted_letter = extract_option_letter(extracted)

    comparable = bool(reference_letter)
    matched = None
    if comparable:
        matched = bool(extracted_letter and extracted_letter == reference_letter)

    return {
        "has_required_format": has_format,
        "extracted": extracted,
        "reference": reference_answer,
        "reference_letter": reference_letter,
        "extracted_letter": extracted_letter,
        "comparable": comparable,
        "matched": matched,
        "score": 1.0 if matched is True else 0.0,
    }


def numbered_steps(steps: List[str]) -> List[str]:
    numbered = []
    for idx, step in enumerate(steps):
        stripped = step.strip()
        if stripped.lower().startswith(f"step {idx}:"):
            numbered.append(stripped)
        else:
            numbered.append(f"Step {idx}: {stripped}")
    return numbered


def build_user_prompt(question: str, steps: List[str], answer: Any) -> str:
    return (
        f"[Question]\n{question}\n"
        f"[Reference Answer]\n{answer if answer is not None else 'None'}\n"
        f"[Candidate Solution]\n{chr(10).join(numbered_steps(steps))}\n"
    )


def build_global_think_messages(item: Dict[str, Any], steps: List[str]):
    question = item.get("question") or item.get("prompt") or ""
    user_text = build_user_prompt(question, steps, item.get("answer"))
    return [
        {"role": "system", "content": GLOBAL_THINK_SYSTEM_PROMPT},
        {"role": "user", "content": openai_content(user_text, item.get("images", []))},
    ]


def build_step_messages(item: Dict[str, Any], steps: List[str], think_text: str, previous_judgments: List[int], current_idx: int):
    question = item.get("question") or item.get("prompt") or ""
    step_lines = numbered_steps(steps)
    previous_text = "\n".join(
        f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_judgments)
    )
    if not previous_text:
        previous_text = "None"
    user_text = (
        f"[Question]\n{question}\n"
        f"[Reference Answer]\n{item.get('answer') if item.get('answer') is not None else 'None'}\n"
        f"[Candidate Solution]\n{chr(10).join(step_lines)}\n"
        f"[Global Thinking]\n<think>{think_text}</think>\n"
        f"[Previous Step Judgments]\n{previous_text}\n"
        f"[Current Step]\n{step_lines[current_idx]}\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )
    return [
        {"role": "system", "content": STEP_SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


async def request_chat_response(
    client: AsyncOpenAI,
    model: str,
    args,
    messages,
    max_tokens: int,
    guided_choice: Optional[List[str]] = None,
    logprobs: bool = False,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    if guided_choice and args.guided_choice:
        kwargs["extra_body"] = {"guided_choice": guided_choice}
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = args.top_logprobs
    api_attempts = args.api_error_retries + 1
    last_error = None
    for api_attempt in range(api_attempts):
        try:
            return await await_with_timeout(client.chat.completions.create(**kwargs), args)
        except Exception as exc:
            last_error = exc
            if api_attempt + 1 >= api_attempts:
                break
            await asyncio.sleep(1.5 * (api_attempt + 1))
    raise last_error


async def request_chat(client: AsyncOpenAI, model: str, args, messages, max_tokens: int, guided_choice: Optional[List[str]] = None):
    response = await request_chat_response(client, model, args, messages, max_tokens, guided_choice=guided_choice)
    return response.choices[0].message.content or ""


async def get_global_thinking(client: AsyncOpenAI, model: str, args, messages):
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            text = await request_chat(client, model, args, messages, args.think_max_tokens)
            attempts.append({"attempt": attempt + 1, "max_tokens": args.think_max_tokens, "error": None})
            return extract_think(text) or text.strip(), text, None, attempts
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "max_tokens": args.think_max_tokens, "error": error})
            await asyncio.sleep(1.5 * (attempt + 1))
    return "", text, error, attempts


async def get_step_score(client: AsyncOpenAI, model: str, args, messages):
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            response = await request_chat_response(
                client,
                model,
                args,
                messages,
                max_tokens=1,
                guided_choice=["1", "0"],
                logprobs=bool(args.use_logprob_score),
            )
            text = response.choices[0].message.content or ""
            pred = parse_token_score(text)
            if pred in (0, 1):
                score, choice_logprobs = logprob_score_for_one(response, pred) if args.use_logprob_score else (float(pred), {})
                attempts.append({
                    "attempt": attempt + 1,
                    "raw": text,
                    "binary": pred,
                    "score": score,
                    "choice_logprobs": choice_logprobs,
                    "error": None,
                })
                return pred, score, choice_logprobs, text, None, attempts
            error = f"Could not parse guided score token from {text!r}"
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        await asyncio.sleep(1.5 * (attempt + 1))
    return -2, float("-inf"), {}, text, error, attempts


async def score_rollout(client: AsyncOpenAI, model: str, args, item: Dict[str, Any], rollout: str) -> Dict[str, Any]:
    start = time.perf_counter()
    steps = split_steps(rollout)
    if not steps:
        return {
            "rollout": rollout,
            "score": float("-inf"),
            "step_scores": None,
            "raw_response": "",
            "global_thinking": "",
            "error": "empty rollout",
            "reward_mode": REWARD_MODE,
            "latency": time.perf_counter() - start,
        }

    think_messages = build_global_think_messages(item, steps)
    think_text, raw_think, think_error, think_attempts = await get_global_thinking(
        client, model, args, think_messages
    )

    step_binary_scores = []
    step_scores = []
    step_choice_logprobs = []
    step_raw_responses = []
    step_errors = []
    step_attempts = []
    for step_idx in range(len(steps)):
        messages = build_step_messages(item, steps, think_text, step_binary_scores, step_idx)
        binary_score, score, choice_logprobs, raw_text, step_error, attempts = await get_step_score(
            client, model, args, messages
        )
        step_binary_scores.append(binary_score)
        step_scores.append(score)
        step_choice_logprobs.append(choice_logprobs)
        step_raw_responses.append(raw_text)
        step_errors.append(step_error)
        step_attempts.append(attempts)

    failed_steps = [i for i, value in enumerate(step_binary_scores) if value not in (0, 1)]
    final_answer = evaluate_final_answer(rollout, item.get("answer"))
    error_parts = []
    if think_error:
        error_parts.append(f"think_error={think_error}")
    if failed_steps:
        error_parts.append(f"failed_steps={failed_steps}")
    error = "; ".join(error_parts)
    ok_scores = step_scores if not failed_steps else None
    prm_step_score = response_score(ok_scores)
    final_answer_score = float(final_answer["score"])
    final_answer_penalty = 0.0
    combined_score = prm_step_score
    return {
        "rollout": rollout,
        "score": combined_score,
        "prm_step_score": prm_step_score,
        "final_answer_score": final_answer_score,
        "final_answer_penalty": final_answer_penalty,
        "final_answer": final_answer,
        "step_scores": ok_scores,
        "step_binary_scores": step_binary_scores if not failed_steps else None,
        "step_choice_logprobs": step_choice_logprobs if not failed_steps else None,
        "num_steps": len(steps),
        "raw_response": raw_think,
        "global_thinking": think_text,
        "step_raw_responses": step_raw_responses,
        "step_attempts": step_attempts,
        "step_errors": step_errors,
        "attempts": think_attempts,
        "error": error,
        "reward_mode": REWARD_MODE,
        "latency": time.perf_counter() - start,
    }


def aggregate_scored_item(item: Dict[str, Any], args, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: int(row["candidate_idx"]))
    valid = [candidate for candidate in candidates if score_candidate_ok(candidate)]
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
        "best_idx": best["candidate_idx"],
        "best_score": best["score"],
        "best_rollout": best["rollout"],
        "num_valid_candidates": len(valid),
        "candidates": candidates,
    }


async def score_item_resumable(
    clients: List[AsyncOpenAI],
    models: List[str],
    base_urls: List[str],
    args,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    candidate_cache: Path,
    existing_candidates: Dict[Tuple[str, int], Dict[str, Any]],
    progress=None,
) -> Dict[str, Any]:
    index = str(item["index"])
    rollouts = (item.get("rollouts") or [])[: args.bon]

    async def guarded(candidate_idx: int, rollout: str):
        key = (index, candidate_idx)
        if key in existing_candidates and score_candidate_done(existing_candidates[key]):
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
                "reward_mode": REWARD_MODE,
                "prm_model": models[endpoint_idx],
                "prm_base_url": base_urls[endpoint_idx],
            })
            append_jsonl(candidate_cache, scored)
            existing_candidates[key] = scored
            if progress is not None:
                progress.update(1)
            return scored

    scored = await asyncio.gather(*[
        guarded(candidate_idx, rollout)
        for candidate_idx, rollout in enumerate(rollouts)
    ])
    return aggregate_scored_item(item, args, scored)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score cached BoN rollouts with VPB global-thinking + chat stepwise PRM logic."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollouts", required=True, help="Cached rollouts_n128.jsonl file.")
    parser.add_argument("--score-output", required=True, help="JSONL output with aggregate per-sample PRM scores.")
    parser.add_argument("--prm-base-url", default=env_default("PRM_BASE_URL", "http://127.0.0.1:8000/v1"),
                        help="One or more comma-separated OpenAI-compatible PRM endpoints.")
    parser.add_argument("--prm-api-key", default=env_default("PRM_API_KEY", "EMPTY"))
    parser.add_argument("--prm-model", default=env_default("PRM_MODEL", "auto"),
                        help="One model name, auto, or comma-separated model names aligned with --prm-base-url.")
    parser.add_argument("--bon", type=int, default=int(env_default("ROLLOUT_N", 128)))
    parser.add_argument("--concurrency", type=int, default=int(env_default("PRM_CONCURRENCY", 4)))
    parser.add_argument("--think-max-tokens", type=int, default=int(env_default("PRM_THINK_MAX_TOKENS", 1024)))
    parser.add_argument("--temperature", type=float, default=float(env_default("PRM_TEMPERATURE", 0.0)))
    parser.add_argument("--guided-choice", type=int, choices=(0, 1), default=int(env_default("PRM_GUIDED_CHOICE", 1)))
    parser.add_argument("--use-logprob-score", type=int, choices=(0, 1),
                        default=int(env_default("PRM_USE_LOGPROB_SCORE", 1)),
                        help="Use P(token=1) from 0/1 token logprobs as each step score.")
    parser.add_argument("--top-logprobs", type=int, default=int(env_default("PRM_TOP_LOGPROBS", 5)))
    parser.add_argument("--require-answer-box", type=int, choices=(0, 1),
                        default=int(env_default("PRM_REQUIRE_ANSWER_BOX", 0)),
                        help="Deprecated compatibility flag; candidate format is left to VLMEvalKit evaluation.")
    parser.add_argument("--final-answer-weight", type=float,
                        default=float(env_default("PRM_FINAL_ANSWER_WEIGHT", 0.0)),
                        help="Deprecated compatibility flag; PRM selection uses only the PRM step score.")
    parser.add_argument("--final-answer-mismatch-penalty", type=float,
                        default=float(env_default("PRM_FINAL_ANSWER_MISMATCH_PENALTY", 0.0)),
                        help="Deprecated compatibility flag; PRM selection uses only the PRM step score.")
    parser.add_argument("--max-retries", type=int, default=int(env_default("PRM_MAX_RETRIES", 3)))
    parser.add_argument("--api-error-retries", type=int, default=int(env_default("PRM_API_ERROR_RETRIES", 3)),
                        help="Retry each failed API request this many times before surfacing the error.")
    parser.add_argument("--request-timeout", type=float, default=float(env_default("PRM_REQUEST_TIMEOUT", 300.0)))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


async def run(args):
    if args.bon < 1:
        raise ValueError("--bon must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.api_error_retries < 0:
        raise ValueError("--api-error-retries must be >= 0")

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

    base_urls = split_csv_arg(args.prm_base_url)
    client_kwargs = {"api_key": args.prm_api_key}
    if args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout + 10
    clients = [AsyncOpenAI(base_url=url, **client_kwargs) for url in base_urls]
    models = await resolve_prm_models(clients, args.prm_model)

    total_pending_candidates = 0
    for item in items:
        index = str(item["index"])
        for candidate_idx, _ in enumerate((item.get("rollouts") or [])[: args.bon]):
            key = (index, candidate_idx)
            if not (key in existing_candidates and score_candidate_done(existing_candidates[key])):
                total_pending_candidates += 1

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = None
    if not args.no_progress and total_pending_candidates > 0:
        progress = tqdm(total=total_pending_candidates, desc=f"score {args.dataset} candidates", dynamic_ncols=True)
    try:
        completed = 0
        tasks = [
            score_item_resumable(
                clients,
                models,
                base_urls,
                args,
                item,
                semaphore,
                candidate_cache,
                existing_candidates,
                progress,
            )
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

    print(f"wrote VPB-style PRM score cache: {score_output}")
    print(f"candidate cache: {candidate_cache}")


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
