#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
TRAILING_SCORE_OBJECT_RE = re.compile(
    r"\s*(?:```(?:json)?\s*)?(?:\\boxed\{\s*)?(?:\[\s*)?\{[^{}\n]*['\"]?(?:Score|score)['\"]?\s*:[^{}\n]*\}(?:\s*\])?(?:\s*\})?\s*(?:```)?\s*$",
    re.IGNORECASE,
)
TRAILING_SCORE_LINE_RE = re.compile(
    r"(?im)\n+\s*(?:final\s+)?(?:step\s+)?(?:scores?|labels?)\s*[:=]\s*\[[^\]]*\]\s*$"
)
STEP_SCORE_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:Step\s*)?\d+\s*(?:[:.)\-]|=>)\s*(?:\\boxed\{)?[01](?:\})?\s*$"
)
ALL_CORRECT_RE = re.compile(
    r"\b(?:all|every|each)\b.{0,80}\b(?:correct|valid|sound|right)\b|"
    r"\bno\b.{0,40}\b(?:error|mistake|wrong|incorrect)\b",
    re.IGNORECASE | re.DOTALL,
)
ERROR_CLAIM_RE = re.compile(r"\b(?:error|mistake|wrong|incorrect|invalid)\b", re.IGNORECASE)
FIRST_ERROR_PATTERNS = [
    re.compile(
        r"\b(?:first|initial|earliest)\b.{0,80}\b(?:error|mistake|wrong|incorrect|invalid)\b.{0,80}\bStep\s*(\d+)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bStep\s*(\d+)\b.{0,80}\b(?:first|initial|earliest)\b.{0,80}\b(?:error|mistake|wrong|incorrect|invalid)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

GLOBAL_THINK_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, produce global thinking that will later be used for step-level process scoring.

Output requirements:
- Output exactly one <think>...</think> block.
- Focus on key visual evidence, the question goal, the reference-answer constraint, dependencies between candidate steps, and the likely first consequential error if any.
- You may include a compact step-level correctness overview if it helps later judging.
- Do not output final step scores.
- Do not output JSON or text outside the <think>...</think> block."""


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def encode_image(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_candidate_solution(steps: list[str]) -> str:
    lines = []
    for idx, step in enumerate(steps):
        text = str(step).strip()
        if text.lower().startswith(f"step {idx}:"):
            lines.append(text)
        else:
            lines.append(f"Step {idx}: {text}")
    return "\n".join(lines)


def build_global_think_text_prompt(row: dict[str, Any]) -> str:
    return (
        "[Question]\n"
        f"{row['question']}\n\n"
        "[Reference Answer]\n"
        f"{str(row.get('answer', '')).strip()}\n\n"
        "[Candidate Solution]\n"
        f"{build_candidate_solution(row['steps'])}\n\n"
        "[Task]\n"
        "Generate the global thinking for later step-level scoring."
    )


def image_full_path(image_dir: Path, rel_or_abs: str) -> Path | None:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path if path.is_file() else None
    full = image_dir / rel_or_abs
    return full if full.is_file() else None


def build_messages(args: argparse.Namespace, row: dict[str, Any]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": build_global_think_text_prompt(row)}]
    for rel_image in row.get("images", []):
        full_path = image_full_path(args.image_dir, rel_image)
        if full_path is not None:
            user_content.append(encode_image(full_path))
    return [
        {"role": "system", "content": GLOBAL_THINK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def strip_trailing_score_artifacts(text: str) -> str:
    stripped = (text or "").strip()
    previous = None
    while stripped and stripped != previous:
        previous = stripped
        stripped = TRAILING_SCORE_OBJECT_RE.sub("", stripped).strip()
        stripped = TRAILING_SCORE_LINE_RE.sub("", stripped).strip()
    return stripped


def extract_thinking(text: str) -> str:
    match = THINK_RE.search(text or "")
    if match:
        return strip_trailing_score_artifacts(match.group(1))

    raw = (text or "").strip()
    if not raw:
        return ""

    if THINK_CLOSE_RE.search(raw) and not THINK_OPEN_RE.search(raw):
        raw = THINK_CLOSE_RE.split(raw, maxsplit=1)[0]
    else:
        raw = THINK_OPEN_RE.sub("", raw)
        raw = THINK_CLOSE_RE.sub("", raw)
    return strip_trailing_score_artifacts(raw)


def has_acceptable_think_block(args: argparse.Namespace, raw_response: str, thinking: str) -> bool:
    if not args.require_think_block:
        return True
    raw_response = raw_response or ""
    if len(THINK_BLOCK_RE.findall(raw_response)) == 1:
        return True
    if args.allow_missing_think_open:
        return (
            len(THINK_OPEN_RE.findall(raw_response)) == 0
            and len(THINK_CLOSE_RE.findall(raw_response)) == 1
            and bool((thinking or "").strip())
        )
    return False


def first_negative_index(labels: list[int]) -> int | None:
    for idx, label in enumerate(labels):
        if label == 0:
            return idx
    return None


def extract_first_error_index(thinking: str) -> int | None:
    for pattern in FIRST_ERROR_PATTERNS:
        match = pattern.search(thinking or "")
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def thinking_filter_reasons(args: argparse.Namespace, thinking: str, raw_response: str, labels: list[int]) -> list[str]:
    reasons = []
    stripped = (thinking or "").strip()
    raw_response = raw_response or ""
    if not stripped:
        reasons.append("empty_thinking")
        return reasons
    if len(stripped) < args.min_think_chars:
        reasons.append("too_short")
    if args.max_think_chars > 0 and len(stripped) > args.max_think_chars:
        reasons.append("too_long")
    if not has_acceptable_think_block(args, raw_response, stripped):
        reasons.append("bad_think_block_count")
    if args.reject_step_scores_in_thinking and STEP_SCORE_LINE_RE.search(stripped):
        reasons.append("contains_step_scores")

    first_neg = first_negative_index(labels)
    says_all_correct = bool(ALL_CORRECT_RE.search(stripped))
    says_error = bool(ERROR_CLAIM_RE.search(stripped))
    claimed_first_error = extract_first_error_index(stripped)
    if args.reject_all_correct_mismatch and first_neg is not None and says_all_correct:
        reasons.append("all_correct_mismatch")
    if args.reject_error_mismatch and first_neg is None and says_error and not says_all_correct:
        reasons.append("error_mismatch_on_all_positive")
    if args.reject_first_error_mismatch and claimed_first_error is not None:
        if first_neg is None:
            reasons.append("first_error_on_all_positive")
        elif abs(claimed_first_error - first_neg) > args.first_error_tolerance:
            reasons.append("first_error_mismatch")
    return reasons


def score_thinking_candidate(args: argparse.Namespace, thinking: str, raw_response: str, labels: list[int]) -> dict[str, Any]:
    reasons = thinking_filter_reasons(args, thinking, raw_response, labels)
    length = len((thinking or "").strip())
    if length < args.min_think_chars:
        length_score = length / max(args.min_think_chars, 1)
    elif args.max_think_chars > 0 and length > args.max_think_chars:
        length_score = max(0.0, 1.0 - (length - args.max_think_chars) / max(args.max_think_chars, 1))
    else:
        length_score = 1.0

    quality = length_score
    if THINK_RE.search(raw_response or ""):
        quality += 0.15
    quality -= 0.35 * len(reasons)
    return {
        "passed": not reasons,
        "filter_reasons": reasons,
        "quality_score": float(max(0.0, quality)),
        "think_chars": length,
    }


def dropout_thinking(thinking: str, keep_ratio: float, rng: random.Random) -> str:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(thinking or "") if piece.strip()]
    if len(pieces) <= 1:
        text = (thinking or "").strip()
        keep_chars = max(1, int(len(text) * keep_ratio))
        return text[:keep_chars]

    keep_count = max(1, int(round(len(pieces) * keep_ratio)))
    kept_indices = sorted(rng.sample(range(len(pieces)), min(keep_count, len(pieces))))
    return " ".join(pieces[idx] for idx in kept_indices)


async def request_thinking(
    endpoint: dict[str, Any],
    args: argparse.Namespace,
    row: dict[str, Any],
    messages: list[dict[str, Any]] | None = None,
) -> tuple[str, str, str | None, list[dict[str, Any]]]:
    if messages is None:
        messages = build_messages(args, row)
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            request = endpoint["client"].chat.completions.create(
                model=endpoint["model"],
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            if args.request_timeout > 0:
                response = await asyncio.wait_for(request, timeout=args.request_timeout)
            else:
                response = await request
            text = response.choices[0].message.content or ""
            thinking = extract_thinking(text)
            if len(thinking) >= args.min_think_chars:
                attempts.append({"attempt": attempt + 1, "error": None, "raw_chars": len(text)})
                return thinking, text, None, attempts
            error = f"thinking too short: {len(thinking)} chars"
            attempts.append({"attempt": attempt + 1, "error": error, "raw_chars": len(text)})
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "error": error, "raw_chars": len(text)})
        await asyncio.sleep(1.5 * (attempt + 1))
    return extract_thinking(text), text, error, attempts


async def request_thinking_candidates(
    endpoint: dict[str, Any],
    args: argparse.Namespace,
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    labels = [int(label) for label in row.get("step_labels", [])]
    messages = build_messages(args, row)
    candidates = []
    last_error = None
    for sample_idx in range(max(1, args.thinking_samples)):
        thinking, raw_response, error, attempts = await request_thinking(endpoint, args, row, messages=messages)
        candidate = {
            "sample_idx": sample_idx,
            "global_thinking": thinking,
            "raw_response": raw_response,
            "error": error,
            "attempts": attempts,
            **score_thinking_candidate(args, thinking, raw_response, labels),
        }
        candidates.append(candidate)
        if error:
            last_error = error

    passed = [candidate for candidate in candidates if candidate["passed"]]
    if passed:
        return sorted(passed, key=lambda item: item["quality_score"], reverse=True), None

    best = sorted(candidates, key=lambda item: item["quality_score"], reverse=True)
    if args.keep_best_failed_thinking and best:
        best[0]["passed"] = True
        best[0]["filter_reasons"] = [*best[0].get("filter_reasons", []), "kept_best_failed"]
        return best[:1], None

    reason_counts = Counter(reason for candidate in candidates for reason in candidate.get("filter_reasons", []))
    reason_text = ",".join(f"{key}:{value}" for key, value in sorted(reason_counts.items()))
    return best, f"no_passed_thinking_candidates; {reason_text or last_error or 'unknown'}"


async def rollout_one(
    endpoint: dict[str, Any],
    args: argparse.Namespace,
    row: dict[str, Any],
    split: str,
    idx: int,
    global_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with global_semaphore, endpoint["semaphore"]:
        start = time.perf_counter()
        candidates, error = await request_thinking_candidates(endpoint, args, row)
        selected = candidates[0] if candidates and candidates[0].get("passed") else {}
        return {
            "id": row.get("id", f"{split}:{idx}"),
            "split": split,
            "source_index": idx,
            "endpoint": endpoint["base_url"],
            "model": endpoint["model"],
            "source_annotation": row.get("source_annotation"),
            "source_sample_id": row.get("source_sample_id"),
            "images": row.get("images", []),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "steps": row.get("steps", []),
            "source_scores": row.get("source_scores", []),
            "step_labels": row.get("step_labels", []),
            "negative_steps": row.get("negative_steps", 0),
            "positive_steps": row.get("positive_steps", 0),
            "negative_threshold": row.get("negative_threshold"),
            "positive_threshold": row.get("positive_threshold"),
            "global_thinking": selected.get("global_thinking", ""),
            "raw_response": selected.get("raw_response", ""),
            "thinking_candidates": candidates,
            "selected_thinking_sample_idx": selected.get("sample_idx"),
            "selected_thinking_quality": selected.get("quality_score", 0.0),
            "error": error,
            "attempts": selected.get("attempts", []),
            "latency": time.perf_counter() - start,
        }


async def build_endpoints(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_urls = split_csv(args.base_urls) if args.base_urls else split_csv(args.base_url)
    if not base_urls:
        raise ValueError("At least one base URL is required.")

    per_endpoint_concurrency = args.per_endpoint_concurrency
    if per_endpoint_concurrency <= 0:
        per_endpoint_concurrency = max(1, args.concurrency // len(base_urls))

    endpoints = []
    for base_url in base_urls:
        client_kwargs = {"base_url": base_url, "api_key": args.api_key}
        if args.request_timeout > 0:
            client_kwargs["timeout"] = args.request_timeout + 10
        client = AsyncOpenAI(**client_kwargs)
        model = args.model
        if model == "auto":
            models = await client.models.list()
            if not models.data:
                raise RuntimeError(f"No models returned by {base_url}/models")
            model = models.data[0].id
            print(f"Using deployed model from {base_url}: {model}")
        endpoints.append(
            {
                "base_url": base_url,
                "client": client,
                "model": model,
                "semaphore": asyncio.Semaphore(per_endpoint_concurrency),
            }
        )
    print(
        "Rollout endpoints: "
        + ", ".join(endpoint["base_url"] for endpoint in endpoints)
        + f"; total_concurrency={args.concurrency}; per_endpoint_concurrency={per_endpoint_concurrency}"
    )
    return endpoints


def step_counts(row: dict[str, Any]) -> tuple[int, int]:
    labels = [int(label) for label in row.get("step_labels", [])]
    return sum(label == 0 for label in labels), sum(label == 1 for label in labels)


def source_is_negative(row: dict[str, Any]) -> bool:
    return step_counts(row)[0] > 0


def source_bonus(row: dict[str, Any], source_bonus_regex: re.Pattern[str] | None, source_bonus: float) -> float:
    if source_bonus_regex is None or source_bonus <= 0:
        return 0.0
    text = " ".join(
        str(row.get(key, ""))
        for key in ("source_annotation", "source_sample_id", "id", "question")
    )
    return source_bonus if source_bonus_regex.search(text) else 0.0


def source_category(row: dict[str, Any]) -> str:
    neg, pos = step_counts(row)
    total = neg + pos
    if neg > 0 and pos > 0:
        return "hard_mixed"
    if neg > 0:
        return "medium_negative"
    if total >= 4:
        return "medium_positive"
    return "easy_positive"


def select_source_rows_for_rollout(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_source_ratio: float,
    target_negative_step_ratio: float,
    source_bonus_regex: re.Pattern[str] | None,
    source_bonus_value: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_size = max_samples if max_samples > 0 else len(rows)
    target_size = min(target_size, len(rows))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prefix = "bonus" if source_bonus(row, source_bonus_regex, source_bonus_value) > 0 else "plain"
        pools[f"{prefix}:{source_category(row)}"].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = {key: 0 for key in pools}

    selected = []
    selected_neg_sources = 0
    selected_neg_steps = 0
    selected_pos_steps = 0

    while len(selected) < target_size:
        candidates = []
        for pool_key, pool in pools.items():
            cursor = cursors[pool_key]
            if cursor >= len(pool):
                continue
            row = pool[cursor]
            row_neg, row_pos = step_counts(row)
            next_count = len(selected) + 1
            next_neg_sources = selected_neg_sources + int(row_neg > 0)
            next_neg_steps = selected_neg_steps + row_neg
            next_pos_steps = selected_pos_steps + row_pos
            source_ratio = next_neg_sources / next_count
            step_ratio = next_neg_steps / max(next_neg_steps + next_pos_steps, 1)
            cost = abs(source_ratio - target_negative_source_ratio) + abs(step_ratio - target_negative_step_ratio)
            if pool_key.startswith("bonus:"):
                cost -= source_bonus_value
            candidates.append((cost, pool_key, row))
        if not candidates:
            break

        _, pool_key, row = min(candidates, key=lambda item: item[0])
        cursors[pool_key] += 1
        row_neg, row_pos = step_counts(row)
        selected.append(row)
        selected_neg_sources += int(row_neg > 0)
        selected_neg_steps += row_neg
        selected_pos_steps += row_pos

    rng.shuffle(selected)
    selected_category_counts = Counter(source_category(row) for row in selected)
    selected_bonus_count = sum(source_bonus(row, source_bonus_regex, source_bonus_value) > 0 for row in selected)
    stats = {
        "source_rows_available": len(rows),
        "source_rows_selected": len(selected),
        "available_pools": {key: len(value) for key, value in sorted(pools.items())},
        "selected_categories": dict(selected_category_counts),
        "selected_bonus_rows": selected_bonus_count,
        "selected_bonus_ratio": selected_bonus_count / len(selected) if selected else 0.0,
        "selected_negative_source_ratio": selected_neg_sources / len(selected) if selected else 0.0,
        "selected_negative_steps": selected_neg_steps,
        "selected_positive_steps": selected_pos_steps,
        "selected_negative_step_ratio": (
            selected_neg_steps / max(selected_neg_steps + selected_pos_steps, 1)
        ),
        "target_negative_source_ratio": target_negative_source_ratio,
        "target_negative_step_ratio": target_negative_step_ratio,
        "source_bonus_regex": source_bonus_regex.pattern if source_bonus_regex else "",
        "source_bonus": source_bonus_value,
    }
    return selected, stats


async def rollout_split(args: argparse.Namespace, split: str, rows: list[dict[str, Any]]) -> None:
    cache_path = args.output_dir / f"{split}_thinking_cache.jsonl"
    if args.overwrite_cache and cache_path.exists():
        cache_path.unlink()

    rng = random.Random(args.seed + (0 if split == "train" else 1000003))
    max_source_samples = args.max_train_source_rollout if split == "train" else args.max_val_source_rollout
    bonus_regex = re.compile(args.source_bonus_regex, re.IGNORECASE) if args.source_bonus_regex else None
    rows, source_selection = select_source_rows_for_rollout(
        rows,
        max_samples=max_source_samples,
        target_negative_source_ratio=args.target_negative_source_ratio,
        target_negative_step_ratio=args.target_negative_step_ratio,
        source_bonus_regex=bonus_regex,
        source_bonus_value=args.source_bonus,
        rng=rng,
    )
    write_json(args.output_dir / f"{split}_source_rollout_selection.json", source_selection)
    print(f"{split}: selected {len(rows)} source rows for thinking rollout from {source_selection['source_rows_available']}")

    cached = load_jsonl(cache_path)
    cached_ids = {row.get("id") for row in cached if row.get("id") and not row.get("error")}
    pending = [
        (idx, row)
        for idx, row in enumerate(rows)
        if row.get("id", f"{split}:{idx}") not in cached_ids
    ]
    if not pending:
        print(f"{split}: no pending thinking rollout rows; cache={cache_path}")
        return

    endpoints = await build_endpoints(args)
    queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=max(args.concurrency * 2, 1))
    completed = errors = 0
    worker_count = min(max(args.concurrency, 1), len(pending))
    progress = tqdm(total=len(pending), desc=f"Rollout {split}", dynamic_ncols=True) if tqdm else None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    cache_file = cache_path.open("a", encoding="utf-8")

    async def producer() -> None:
        for item in pending:
            await queue.put(item)
        for _ in range(worker_count):
            await queue.put(None)

    async def worker(worker_idx: int) -> None:
        nonlocal completed, errors
        endpoint = endpoints[worker_idx % len(endpoints)]
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                idx, row = item
                result = await rollout_one(endpoint, args, row, split, idx, asyncio.Semaphore(1))
                async with write_lock:
                    next_completed = completed + 1
                    cache_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if args.cache_flush_every > 0 and next_completed % args.cache_flush_every == 0:
                        cache_file.flush()
                    if args.cache_fsync_every > 0 and next_completed % args.cache_fsync_every == 0:
                        cache_file.flush()
                        os.fsync(cache_file.fileno())
                    completed = next_completed
                    errors += int(bool(result.get("error")))
                if progress:
                    progress.update(1)
                    progress.set_postfix(errors=errors)
                elif completed % args.log_every == 0 or completed == len(pending):
                    print(f"{split}: completed {completed}/{len(pending)} errors={errors}")
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(worker_idx)) for worker_idx in range(worker_count)]
    producer_task = asyncio.create_task(producer())
    try:
        await producer_task
        await queue.join()
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        if progress:
            progress.close()
        cache_file.flush()
        if args.cache_fsync_every > 0:
            os.fsync(cache_file.fileno())
        cache_file.close()


def build_previous_text(previous_scores: list[int]) -> str:
    if not previous_scores:
        return "None"
    return "\n".join(f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_scores))


def build_step_prompt(
    row: dict[str, Any],
    current_idx: int,
    previous_scores: list[int],
    thinking_text: str,
    thinking_mode: str,
) -> str:
    steps = row["steps"]
    if thinking_mode == "no_thinking":
        thinking_block = "<think>No cached global thinking is provided. Judge from the image, question, reference answer, candidate solution, previous judgments, and current step.</think>"
    else:
        thinking_block = f"<think>{thinking_text.strip()}</think>"
    return (
        "[Question]\n"
        f"{row['question']}\n\n"
        "[Reference Answer]\n"
        f"{str(row.get('answer', '')).strip()}\n\n"
        "[Candidate Solution]\n"
        f"{build_candidate_solution(steps)}\n\n"
        "[Global Thinking]\n"
        f"{thinking_block}\n\n"
        "[Previous Step Judgments]\n"
        f"{build_previous_text(previous_scores)}\n\n"
        "[Current Step]\n"
        f"{steps[current_idx]}\n\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )


def rescore_thinking_candidates(
    source: dict[str, Any],
    args: argparse.Namespace,
    labels: list[int],
) -> list[dict[str, Any]]:
    rescored = []
    for candidate in source.get("thinking_candidates", []):
        raw_response = str(candidate.get("raw_response") or candidate.get("global_thinking") or "")
        thinking = extract_thinking(raw_response)
        if not thinking and candidate.get("global_thinking"):
            thinking = strip_trailing_score_artifacts(str(candidate.get("global_thinking", "")))
        if not raw_response and thinking:
            raw_response = f"<think>{thinking}</think>"
        score = score_thinking_candidate(args, thinking, raw_response, labels)
        rescored.append(
            {
                **candidate,
                "global_thinking": thinking,
                "raw_response": raw_response,
                **score,
                "original_passed": candidate.get("passed"),
                "original_filter_reasons": candidate.get("filter_reasons", []),
            }
        )
    return rescored


def has_recoverable_thinking(source: dict[str, Any], args: argparse.Namespace) -> bool:
    labels = [int(label) for label in source.get("step_labels", [])]
    for candidate in rescore_thinking_candidates(source, args, labels):
        if candidate.get("passed") and candidate.get("global_thinking"):
            return True
    if source.get("global_thinking"):
        raw_response = str(source.get("raw_response", ""))
        thinking = extract_thinking(raw_response) if raw_response else str(source.get("global_thinking", ""))
        score = score_thinking_candidate(args, thinking, raw_response or f"<think>{thinking}</think>", labels)
        return bool(score["passed"] or args.keep_best_failed_thinking)
    return False


def selected_thinking_variants(source: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    labels = [int(label) for label in source.get("step_labels", [])]
    candidates = [
        candidate for candidate in rescore_thinking_candidates(source, args, labels)
        if candidate.get("passed") and candidate.get("global_thinking")
    ]
    if not candidates and source.get("global_thinking"):
        raw_response = str(source.get("raw_response", ""))
        thinking = extract_thinking(raw_response) if raw_response else str(source.get("global_thinking", ""))
        legacy_score = score_thinking_candidate(
            args,
            thinking,
            raw_response or f"<think>{thinking}</think>",
            labels,
        )
        if not legacy_score["passed"] and not args.keep_best_failed_thinking:
            return []
        candidates = [
            {
                "sample_idx": source.get("selected_thinking_sample_idx", 0),
                "global_thinking": thinking,
                "quality_score": source.get("selected_thinking_quality", legacy_score["quality_score"]),
                "filter_reasons": legacy_score["filter_reasons"],
                "passed": legacy_score["passed"] or args.keep_best_failed_thinking,
            }
        ]
    candidates = sorted(candidates, key=lambda item: item.get("quality_score", 0.0), reverse=True)
    candidates = candidates[: max(1, args.max_thinking_variants_per_source)]

    variants = []
    for rank, candidate in enumerate(candidates):
        thinking = str(candidate.get("global_thinking", "")).strip()
        if not thinking:
            continue
        base_variant_id = f"think{candidate.get('sample_idx', rank)}"
        variants.append(
            {
                "thinking_mode": "cached",
                "thinking_variant_id": base_variant_id,
                "global_thinking": thinking,
                "thinking_quality": candidate.get("quality_score", 0.0),
                "thinking_filter_reasons": candidate.get("filter_reasons", []),
            }
        )
        if args.thinking_dropout_ratio > 0 and rng.random() < args.thinking_dropout_ratio:
            variants.append(
                {
                    "thinking_mode": "dropout",
                    "thinking_variant_id": f"{base_variant_id}:dropout",
                    "global_thinking": dropout_thinking(thinking, args.thinking_sentence_keep_ratio, rng),
                    "thinking_quality": candidate.get("quality_score", 0.0),
                    "thinking_filter_reasons": candidate.get("filter_reasons", []),
                }
            )
        if args.no_thinking_ratio > 0 and rng.random() < args.no_thinking_ratio:
            variants.append(
                {
                    "thinking_mode": "no_thinking",
                    "thinking_variant_id": f"{base_variant_id}:no_thinking",
                    "global_thinking": "",
                    "thinking_quality": candidate.get("quality_score", 0.0),
                    "thinking_filter_reasons": candidate.get("filter_reasons", []),
                }
            )
    return variants


def make_step_rows(cached_rows: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for source in cached_rows:
        if source.get("error") and not (source.get("thinking_candidates") or source.get("global_thinking")):
            continue
        variants = selected_thinking_variants(source, args, rng)
        if not variants:
            continue
        labels = [int(label) for label in source.get("step_labels", [])]
        for variant in variants:
            previous_scores: list[int] = []
            group_id = f"{source.get('id')}::{variant['thinking_variant_id']}"
            for idx, label in enumerate(labels):
                if label not in (0, 1):
                    previous_scores.append(0)
                    continue
                ground_truth = {
                    "label": label,
                    "step_index": idx,
                    "num_steps": len(labels),
                    "source_scores": source.get("source_scores", []),
                    "step_labels": labels,
                    "negative_threshold": source.get("negative_threshold"),
                    "positive_threshold": source.get("positive_threshold"),
                    "source_id": source.get("id"),
                    "source_annotation": source.get("source_annotation"),
                    "source_sample_id": source.get("source_sample_id"),
                    "answer": source.get("answer", ""),
                    "thinking_mode": variant["thinking_mode"],
                    "thinking_variant_id": variant["thinking_variant_id"],
                    "source_group_id": group_id,
                    "thinking_quality": variant["thinking_quality"],
                    "thinking_filter_reasons": variant["thinking_filter_reasons"],
                }
                rows.append(
                    {
                        "prompt": build_step_prompt(
                            source,
                            idx,
                            previous_scores,
                            thinking_text=variant["global_thinking"],
                            thinking_mode=variant["thinking_mode"],
                        ),
                        "images": source.get("images", []),
                        "ground_truth": json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":")),
                        "label": label,
                        "source_id": source.get("id"),
                        "source_group_id": group_id,
                        "source_step_index": idx,
                        "thinking_mode": variant["thinking_mode"],
                    }
                )
                previous_scores.append(label)
    return rows


def select_step_rows(
    rows: list[dict[str, Any]],
    max_samples: int,
    target_negative_ratio: float,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source_group_id") or row.get("source_id"))].append(row)
    group_rows = []
    for group_id, group in groups.items():
        group.sort(key=lambda item: int(item.get("source_step_index", 0)))
        neg = sum(row["label"] == 0 for row in group)
        pos = sum(row["label"] == 1 for row in group)
        group_rows.append((group_id, group, neg, pos))
    rng.shuffle(group_rows)

    target_size = max_samples if max_samples > 0 else len(rows)
    target_size = min(target_size, len(rows))
    selected_groups = []
    selected_steps = selected_neg = selected_pos = 0
    while selected_steps < target_size and group_rows:
        best_idx = 0
        best_cost = float("inf")
        for idx, (_, group, neg, pos) in enumerate(group_rows[: min(len(group_rows), 2048)]):
            if selected_steps + len(group) > target_size and selected_groups:
                continue
            next_neg = selected_neg + neg
            next_pos = selected_pos + pos
            next_ratio = next_neg / max(next_neg + next_pos, 1)
            cost = abs(next_ratio - target_negative_ratio)
            if cost < best_cost:
                best_idx = idx
                best_cost = cost
        _, group, neg, pos = group_rows.pop(best_idx)
        selected_groups.append(group)
        selected_steps += len(group)
        selected_neg += neg
        selected_pos += pos

    if args.shuffle_source_groups:
        rng.shuffle(selected_groups)
    selected = [row for group in selected_groups for row in group]
    neg = sum(row["label"] == 0 for row in selected)
    pos = sum(row["label"] == 1 for row in selected)
    mode_counts = Counter(row.get("thinking_mode", "unknown") for row in selected)
    stats = {
        "available_step_rows": len(rows),
        "available_source_groups": len(groups),
        "selected_source_groups": len(selected_groups),
        "available_negative_steps": sum(row["label"] == 0 for row in rows),
        "available_positive_steps": sum(row["label"] == 1 for row in rows),
        "selected_negative_steps": neg,
        "selected_positive_steps": pos,
        "target_negative_ratio": target_negative_ratio,
        "selected_negative_ratio": neg / (neg + pos) if neg + pos else 0.0,
        "selected_thinking_modes": dict(mode_counts),
    }
    return selected, stats


def strip_private_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "label"} for row in rows]


def process_cached_data(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    train_cache = load_jsonl(args.output_dir / "train_thinking_cache.jsonl")
    test_cache = load_jsonl(args.output_dir / "test_thinking_cache.jsonl")
    train_candidates = make_step_rows(train_cache, args=args, rng=rng)
    test_candidates = make_step_rows(test_cache, args=args, rng=random.Random(args.seed + 17))

    train_rows, train_selection = select_step_rows(
        train_candidates,
        max_samples=args.max_train_step_samples,
        target_negative_ratio=args.target_negative_step_ratio,
        rng=rng,
        args=args,
    )
    test_rows, test_selection = select_step_rows(
        test_candidates,
        max_samples=args.max_val_step_samples,
        target_negative_ratio=args.target_negative_step_ratio,
        rng=random.Random(args.seed + 29),
        args=args,
    )

    write_jsonl(args.output_dir / "train.jsonl", strip_private_fields(train_rows))
    write_jsonl(args.output_dir / "test.jsonl", strip_private_fields(test_rows))
    write_json(
        args.output_dir / "dataset_meta.json",
        {
            "seed": args.seed,
            "source_data_dir": str(args.input_dir),
            "max_train_step_samples": args.max_train_step_samples,
            "max_val_step_samples": args.max_val_step_samples,
            "target_negative_step_ratio": args.target_negative_step_ratio,
            "thinking_samples": args.thinking_samples,
            "max_thinking_variants_per_source": args.max_thinking_variants_per_source,
            "thinking_dropout_ratio": args.thinking_dropout_ratio,
            "no_thinking_ratio": args.no_thinking_ratio,
            "thinking_sentence_keep_ratio": args.thinking_sentence_keep_ratio,
            "filter_config": {
                "min_think_chars": args.min_think_chars,
                "max_think_chars": args.max_think_chars,
                "require_think_block": args.require_think_block,
                "allow_missing_think_open": args.allow_missing_think_open,
                "reject_step_scores_in_thinking": args.reject_step_scores_in_thinking,
                "reject_all_correct_mismatch": args.reject_all_correct_mismatch,
                "reject_error_mismatch": args.reject_error_mismatch,
                "reject_first_error_mismatch": args.reject_first_error_mismatch,
                "first_error_tolerance": args.first_error_tolerance,
                "keep_best_failed_thinking": args.keep_best_failed_thinking,
            },
            "train_cache_rows": len(train_cache),
            "test_cache_rows": len(test_cache),
            "train_passed_cache_rows": sum(has_recoverable_thinking(row, args) for row in train_cache),
            "test_passed_cache_rows": sum(has_recoverable_thinking(row, args) for row in test_cache),
            "train_step_candidates": len(train_candidates),
            "test_step_candidates": len(test_candidates),
            "selection": train_selection,
            "val_selection": test_selection,
            "note": "Each row is one current-step RL prompt with cached SFT global thinking in the prompt and score-only response expected.",
        },
    )
    (args.output_dir / ".visualprm_cached_think_step_paths_v1").write_text("ok\n", encoding="utf-8")
    print(f"Wrote {len(train_rows)} train RL step rows: {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(test_rows)} val RL step rows: {args.output_dir / 'test.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out SFT global thinking for filtered VisualPRM source cases and build step-level RL data."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL. May also be comma-separated for DP replicas.",
    )
    parser.add_argument(
        "--base-urls",
        default="",
        help="Comma-separated OpenAI-compatible base URLs. Overrides --base-url when set.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--max-tokens", default=1024, type=int)
    parser.add_argument("--min-think-chars", default=80, type=int)
    parser.add_argument("--max-think-chars", default=1800, type=int)
    parser.add_argument("--thinking-samples", default=1, type=int)
    parser.add_argument("--max-thinking-variants-per-source", default=1, type=int)
    parser.add_argument("--thinking-dropout-ratio", default=0.0, type=float)
    parser.add_argument("--no-thinking-ratio", default=0.0, type=float)
    parser.add_argument("--thinking-sentence-keep-ratio", default=0.65, type=float)
    parser.add_argument("--require-think-block", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-missing-think-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accept responses with exactly one closing </think> but no opening <think>, which some Qwen/vLLM chat templates emit.",
    )
    parser.add_argument("--reject-step-scores-in-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-all-correct-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-error-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reject-first-error-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-error-tolerance", default=1, type=int)
    parser.add_argument("--keep-best-failed-thinking", action="store_true")
    parser.add_argument("--concurrency", default=64, type=int, help="Total rollout concurrency across all endpoints.")
    parser.add_argument(
        "--per-endpoint-concurrency",
        default=16,
        type=int,
        help="Concurrency cap per base URL. Use 0 to derive concurrency/num_endpoints.",
    )
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument("--request-timeout", default=300.0, type=float)
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--cache-flush-every", default=20, type=int)
    parser.add_argument("--cache-fsync-every", default=0, type=int, help="0 disables fsync during rollout.")
    parser.add_argument("--splits", default="train,test", help="Comma-separated splits to roll out: train,test")
    parser.add_argument("--skip-rollout", action="store_true", help="Only process existing thinking caches.")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--max-train-source-rollout", default=0, type=int)
    parser.add_argument("--max-val-source-rollout", default=0, type=int)
    parser.add_argument("--max-train-step-samples", default=50000, type=int)
    parser.add_argument("--max-val-step-samples", default=2000, type=int)
    parser.add_argument("--target-negative-source-ratio", default=0.50, type=float)
    parser.add_argument("--target-negative-step-ratio", default=0.35, type=float)
    parser.add_argument(
        "--source-bonus-regex",
        default="",
        help="Optional regex. Matching source rows get a lower selection cost, useful for WeMath-like math/geo data.",
    )
    parser.add_argument("--source-bonus", default=0.0, type=float)
    parser.add_argument("--shuffle-source-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_rollout:
        split_to_file = {
            "train": args.input_dir / "train_source.jsonl",
            "test": args.input_dir / "test_source.jsonl",
        }
        for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
            if split not in split_to_file:
                raise ValueError(f"Unknown split: {split}")
            rows = load_jsonl(split_to_file[split])
            if not rows:
                raise FileNotFoundError(f"No rows loaded from {split_to_file[split]}")
            await rollout_split(args, split, rows)
    process_cached_data(args)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
