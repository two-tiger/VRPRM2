import argparse
import asyncio
import base64
import csv
import json
import math
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from metrics import compute_metrics, write_json


DEFAULT_BENCH_DIR = Path(__file__).resolve().parent / "VisualProcessBench"
SCORE_RE = re.compile(r'\{\s*"Score"\s*:\s*\[([^\]]*)\]\s*\}')
THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
TOKEN_SCORE_RE = re.compile(r"[01]")
MAX_TOKENS_ERROR_RE = re.compile(r"(\d+)\s*>\s*(\d+)\s*-\s*(\d+)")

SFT_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, and a step-by-step candidate solution, generate SFT-quality supervision for a visual reasoning process reward model.

For each solution step, output 1 only when the step is fully supported by the image/problem context and remains logically and mathematically correct. Output 0 when the step is unsupported, contains a visual misunderstanding, uses invalid logic, has a calculation error, contradicts earlier valid reasoning, or depends on a previous incorrect step.

Output requirements:
- Start with [Preliminary thinking] and include exactly one <think>...</think> block.
- Then provide [Step Analysis] with one line per received step. Each line must end with \\boxed{0} or \\boxed{1}.
- Then provide [Final Scores] as strict JSON: {"Score": [comma-separated integer scores]}.
- Then provide [Final Judgment] as strict JSON: {"Judge": 0 or 1}.
- The Score array length must exactly equal the number of received steps.
- Do not add any text after the final JSON object."""

DIRECT_SCORE_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, and a step-by-step candidate solution, judge each solution step.

For each solution step, output 1 only when the step is fully supported by the image/problem context and remains logically and mathematically correct. Output 0 when the step is unsupported, contains a visual misunderstanding, uses invalid logic, has a calculation error, contradicts earlier valid reasoning, or depends on a previous incorrect step.

Output requirements:
- Do not write reasoning.
- Do not write step analysis.
- Output only strict JSON: {"Score": [comma-separated integer scores]}.
- The Score array length must exactly equal the number of received steps."""

NO_THINK_SINGLE_PASS_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, and a step-by-step candidate solution, judge every solution step.

Do not reason out loud. Do not explain. Do not output <think>...</think>.

Scoring policy:
- Output 1 only if the step is fully supported by the image/problem context and remains logically and mathematically correct.
- Output 0 if the step is unsupported, visually mistaken, logically invalid, computationally wrong, contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery.

Output requirements:
- Output only strict JSON: {"Score": [comma-separated integer scores]}.
- The Score array length must exactly equal the number of received steps.
- Do not add any other text."""

NO_THINK_SINGLE_PASS_WITH_REFERENCE_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, judge every solution step.

Do not reason out loud. Do not explain. Do not output <think>...</think>.

Scoring policy:
- Output 1 only if the step is fully supported by the image/problem context and remains logically and mathematically correct.
- Output 0 if the step is unsupported, visually mistaken, logically invalid, computationally wrong, contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery.

Output requirements:
- Output only strict JSON: {"Score": [comma-separated integer scores]}.
- The Score array length must exactly equal the number of received steps.
- Do not add any other text."""

STEPWISE_WARMUP_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, and a step-by-step candidate solution, prepare concise reasoning context for judging each solution step.

Output requirements:
- Output exactly one concise <think>...</think> block.
- Focus on the visual evidence, problem constraints, and the key checks needed for the candidate solution.
- Do not output per-step scores.
- Do not output final JSON."""

GLOBAL_THINK_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, and a complete candidate solution, think before judging individual steps.

Output requirements:
- Output exactly one <think>...</think> block.
- In the thinking, identify the key visual evidence, the question goal, dependencies between steps, and likely first consequential error if any.
- You may include a compact step-level correctness overview if it helps make later step scoring consistent.
- Do not output text outside the <think>...</think> block."""

GLOBAL_THINK_WITH_REFERENCE_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, think before judging individual steps.

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


def env_default(name: str, default: Any) -> Any:
    return os.environ.get(name, default)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def split_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def encode_image(path: Path) -> Dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def build_solution_text(steps: List[str]) -> str:
    return "\n".join(
        step if str(step).strip().lower().startswith(f"step {idx}:") else f"Step {idx}: {step}"
        for idx, step in enumerate(steps)
    )


def numbered_steps(steps: List[str]) -> List[str]:
    return [
        step if str(step).strip().lower().startswith(f"step {idx}:") else f"Step {idx}: {step}"
        for idx, step in enumerate(steps)
    ]


def build_user_prompt(question: str, steps: List[str], answer: str, use_reference_answer: bool = False) -> str:
    text = f"[Question]\n{question}\n"
    if use_reference_answer:
        text += f"[Reference Answer]\n{answer}\n"
    text += f"[Candidate Solution]\n{build_solution_text(steps)}\n"
    return text


def build_full_user_prompt(question: str, steps: List[str], answer: str, use_reference_answer: bool = False) -> str:
    text = (
        f"[Question]\n{question}\n"
        f"[Solution]\n{'<step split>'.join(steps)}\n"
    )
    if use_reference_answer:
        text += f"[Answer]\n{answer}\n"
    return text


def build_no_think_user_prompt(question: str, steps: List[str], answer: str, use_reference_answer: bool = False) -> str:
    step_texts = numbered_steps(steps)
    text = f"[Question]\n{question}\n"
    if use_reference_answer:
        text += f"[Reference Answer]\n{answer}\n"
    text += (
        f"[Candidate Solution]\n{chr(10).join(step_texts)}\n"
        f"[Number of Steps]\n{len(step_texts)}\n"
        'Directly output strict JSON in this exact form: {"Score": [0 or 1 for each step]}.'
    )
    return text


def build_multimodal_messages(system_prompt: str, user_text: str, image_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_content = [{"type": "text", "text": user_text}]
    user_content.extend(image_contents)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_step_messages(
    row: Dict[str, Any],
    steps: List[str],
    think_text: str,
    previous_judgments: List[int],
    current_idx: int,
    use_reference_answer: bool = False,
):
    step_texts = numbered_steps(steps)
    previous_text = "\n".join(
        f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_judgments)
    ) or "None"
    user_text = f"[Question]\n{row['question']}\n"
    if use_reference_answer:
        user_text += f"[Reference Answer]\n{row['answer']}\n"
    user_text += (
        f"[Candidate Solution]\n{chr(10).join(step_texts)}\n"
        f"[Global Thinking]\n<think>{think_text}</think>\n"
        f"[Previous Step Judgments]\n{previous_text}\n"
        f"[Current Step]\n{step_texts[current_idx]}\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )
    return [
        {"role": "system", "content": STEP_SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def extract_think(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def parse_token_score(text: str) -> int:
    match = TOKEN_SCORE_RE.search(text or "")
    return int(match.group(0)) if match else -2


def parse_scores(text: str, expected_len: int) -> List[int]:
    match = SCORE_RE.search(text or "")
    if not match:
        return [-2] * expected_len
    raw_items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    scores = []
    for item in raw_items:
        try:
            scores.append(1 if float(item) > 0 else 0)
        except ValueError:
            scores.append(-2)
    return scores if len(scores) == expected_len else [-2] * expected_len


def infer_retry_max_tokens(error: str) -> Optional[int]:
    match = MAX_TOKENS_ERROR_RE.search(error or "")
    if not match:
        return None
    _requested, model_len, input_len = [int(item) for item in match.groups()]
    return max(1, model_len - input_len)


def load_images(args, row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], float]:
    start = time.perf_counter()
    image_contents = []
    image_paths = []
    for rel_image in row.get("image", []):
        image_path = Path(args.benchmark_dir) / rel_image
        if image_path.exists():
            image_contents.append(encode_image(image_path))
            image_paths.append(str(image_path))
    return image_contents, image_paths, time.perf_counter() - start


def usage_to_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: getattr(usage, key) for key in keys if hasattr(usage, key)}


def usage_number(usage: Dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def add_usage(target: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = target.get(key, 0) + usage_number(usage, key)
    if usage:
        target["usage_available_requests"] = target.get("usage_available_requests", 0) + 1


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    chunks = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    chunks.append("<image>")
    return "\n".join(chunks)


def count_text_tokens(tokenizer: Any, text: str) -> Optional[int]:
    if tokenizer is None or not text:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        return len(encoded.get("input_ids", []))
    except Exception:
        return None


def load_optional_tokenizer(path: str) -> Any:
    if not path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    except Exception as exc:
        print(
            f"[warn] failed to load optional tokenizer from {path!r}: {exc!r}. "
            "Continue without local token estimates; API usage tokens will still be recorded if vLLM returns them.",
            flush=True,
        )
        return None


async def await_with_timeout(coro, timeout: float):
    if timeout and timeout > 0:
        return await asyncio.wait_for(coro, timeout=timeout)
    return await coro


async def request_chat(client, args, messages, max_tokens: int, phase: str, tokenizer=None, guided_choice=None):
    attempts = []
    text = ""
    current_max_tokens = max_tokens
    prompt_token_estimate = count_text_tokens(tokenizer, messages_to_text(messages))
    for attempt in range(args.sft_max_retries):
        start = time.perf_counter()
        try:
            kwargs = {
                "model": args.sft_model,
                "messages": messages,
                "temperature": args.sft_temperature,
                "max_tokens": current_max_tokens,
            }
            if guided_choice:
                kwargs["extra_body"] = {"guided_choice": guided_choice}
            response = await await_with_timeout(client.chat.completions.create(**kwargs), args.sft_request_timeout)
            latency = time.perf_counter() - start
            text = response.choices[0].message.content or ""
            usage = usage_to_dict(response)
            attempts.append({
                "attempt": attempt + 1,
                "latency_sec": latency,
                "max_tokens": current_max_tokens,
                "usage": usage,
                "output_chars": len(text),
                "output_token_estimate": count_text_tokens(tokenizer, text),
                "error": None,
            })
            return text, {
                "phase": phase,
                "api": "chat.completions",
                "latency_sec": latency,
                "max_tokens": current_max_tokens,
                "usage": usage,
                "prompt_token_estimate": prompt_token_estimate,
                "output_chars": len(text),
                "output_token_estimate": count_text_tokens(tokenizer, text),
                "attempts": attempts,
                "error": None,
            }, None
        except Exception as exc:
            latency = time.perf_counter() - start
            error = repr(exc)
            attempts.append({
                "attempt": attempt + 1,
                "latency_sec": latency,
                "max_tokens": current_max_tokens,
                "usage": {},
                "output_chars": 0,
                "output_token_estimate": 0,
                "error": error,
            })
            retry_max_tokens = infer_retry_max_tokens(error)
            if retry_max_tokens is not None and retry_max_tokens < current_max_tokens:
                current_max_tokens = retry_max_tokens
                continue
            await asyncio.sleep(1.5 * (attempt + 1))
    return text, {
        "phase": phase,
        "api": "chat.completions",
        "latency_sec": sum(item["latency_sec"] for item in attempts),
        "max_tokens": current_max_tokens,
        "usage": {},
        "prompt_token_estimate": prompt_token_estimate,
        "output_chars": len(text),
        "output_token_estimate": count_text_tokens(tokenizer, text) or 0,
        "attempts": attempts,
        "error": attempts[-1]["error"] if attempts else "request failed",
    }, attempts[-1]["error"] if attempts else "request failed"


async def request_completion(client, args, prompt: str, phase: str, tokenizer=None, guided_choice=None):
    attempts = []
    text = ""
    prompt_token_estimate = count_text_tokens(tokenizer, prompt)
    for attempt in range(args.sft_max_retries):
        start = time.perf_counter()
        try:
            kwargs = {
                "model": args.sft_model,
                "prompt": prompt,
                "temperature": args.sft_temperature,
                "max_tokens": 1,
            }
            if guided_choice:
                kwargs["extra_body"] = {"guided_choice": guided_choice}
            response = await await_with_timeout(client.completions.create(**kwargs), args.sft_request_timeout)
            latency = time.perf_counter() - start
            text = response.choices[0].text or ""
            usage = usage_to_dict(response)
            attempts.append({
                "attempt": attempt + 1,
                "latency_sec": latency,
                "max_tokens": 1,
                "usage": usage,
                "output_chars": len(text),
                "output_token_estimate": count_text_tokens(tokenizer, text),
                "error": None,
            })
            return text, {
                "phase": phase,
                "api": "completions",
                "latency_sec": latency,
                "max_tokens": 1,
                "usage": usage,
                "prompt_token_estimate": prompt_token_estimate,
                "output_chars": len(text),
                "output_token_estimate": count_text_tokens(tokenizer, text),
                "attempts": attempts,
                "error": None,
            }, None
        except Exception as exc:
            latency = time.perf_counter() - start
            attempts.append({
                "attempt": attempt + 1,
                "latency_sec": latency,
                "max_tokens": 1,
                "usage": {},
                "output_chars": 0,
                "output_token_estimate": 0,
                "error": repr(exc),
            })
            await asyncio.sleep(1.5 * (attempt + 1))
    return text, {
        "phase": phase,
        "api": "completions",
        "latency_sec": sum(item["latency_sec"] for item in attempts),
        "max_tokens": 1,
        "usage": {},
        "prompt_token_estimate": prompt_token_estimate,
        "output_chars": len(text),
        "output_token_estimate": count_text_tokens(tokenizer, text) or 0,
        "attempts": attempts,
        "error": attempts[-1]["error"] if attempts else "request failed",
    }, attempts[-1]["error"] if attempts else "request failed"


def finalize_sft_row(args, row, idx, pred, request_details, start, image_paths, image_latency, error, parse_error):
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usage_available_requests": 0}
    output_token_estimate = 0
    prompt_token_estimate = 0
    for detail in request_details:
        add_usage(usage, detail.get("usage") or {})
        output_token_estimate += int(detail.get("output_token_estimate") or 0)
        prompt_token_estimate += int(detail.get("prompt_token_estimate") or 0)
    if usage["completion_tokens"]:
        generated_tokens = usage["completion_tokens"]
        generated_tokens_source = "api_usage"
    elif output_token_estimate:
        generated_tokens = output_token_estimate
        generated_tokens_source = "tokenizer_estimate"
    else:
        generated_tokens = 0
        generated_tokens_source = "unavailable"
    return {
        "backend": "sft",
        "mode": args.sft_mode,
        "id": idx,
        "data_source": row.get("data_source"),
        "policy_model": row.get("policy_model"),
        "image": row.get("image", []),
        "image_paths": image_paths,
        "num_images": len(image_paths),
        "num_steps": len(row["response"]["steps"]),
        "ground_truth": row["response"]["process_correctness"],
        "pred": pred,
        "latency_sec": time.perf_counter() - start,
        "image_encode_latency_sec": image_latency,
        "num_requests": len(request_details),
        "num_step_scores": len(pred),
        "use_reference_answer": bool(getattr(args, "sft_use_reference_answer", False)),
        "usage": usage,
        "generated_tokens": generated_tokens,
        "generated_tokens_source": generated_tokens_source,
        "text_input_tokens": usage["prompt_tokens"] or prompt_token_estimate,
        "text_input_tokens_source": "api_usage" if usage["prompt_tokens"] else "tokenizer_estimate",
        "output_token_estimate": output_token_estimate,
        "request_details": request_details,
        "error": error,
        "parse_error": parse_error,
    }


async def evaluate_sft_full(client, args, row, idx, tokenizer):
    start = time.perf_counter()
    image_contents, image_paths, image_latency = load_images(args, row)
    steps = row["response"]["steps"]
    messages = build_multimodal_messages(
        SFT_SYSTEM_PROMPT,
        build_full_user_prompt(
            row["question"],
            steps,
            row["answer"],
            use_reference_answer=bool(args.sft_use_reference_answer),
        ),
        image_contents,
    )
    text, detail, error = await request_chat(client, args, messages, args.sft_full_max_tokens, "full_score", tokenizer)
    pred = parse_scores(text, len(steps))
    parse_error = any(value == -2 for value in pred)
    return finalize_sft_row(args, row, idx, pred, [detail], start, image_paths, image_latency, error, parse_error)


async def evaluate_sft_direct_scores(client, args, row, idx, tokenizer):
    start = time.perf_counter()
    image_contents, image_paths, image_latency = load_images(args, row)
    steps = row["response"]["steps"]
    messages = build_multimodal_messages(
        DIRECT_SCORE_SYSTEM_PROMPT,
        build_full_user_prompt(
            row["question"],
            steps,
            row["answer"],
            use_reference_answer=bool(args.sft_use_reference_answer),
        ),
        image_contents,
    )
    text, detail, error = await request_chat(
        client,
        args,
        messages,
        args.sft_direct_max_tokens,
        "direct_scores",
        tokenizer,
    )
    pred = parse_scores(text, len(steps))
    parse_error = any(value == -2 for value in pred)
    return finalize_sft_row(args, row, idx, pred, [detail], start, image_paths, image_latency, error, parse_error)


async def evaluate_sft_no_think_single_pass(client, args, row, idx, tokenizer):
    start = time.perf_counter()
    image_contents, image_paths, image_latency = load_images(args, row)
    steps = row["response"]["steps"]
    messages = build_multimodal_messages(
        (
            NO_THINK_SINGLE_PASS_WITH_REFERENCE_SYSTEM_PROMPT
            if args.sft_use_reference_answer
            else NO_THINK_SINGLE_PASS_SYSTEM_PROMPT
        ),
        build_no_think_user_prompt(
            row["question"],
            steps,
            row["answer"],
            use_reference_answer=bool(args.sft_use_reference_answer),
        ),
        image_contents,
    )
    text, detail, error = await request_chat(
        client,
        args,
        messages,
        args.sft_direct_max_tokens,
        "no_think_single_pass",
        tokenizer,
    )
    pred = parse_scores(text, len(steps))
    parse_error = any(value == -2 for value in pred)
    return finalize_sft_row(args, row, idx, pred, [detail], start, image_paths, image_latency, error, parse_error)


async def evaluate_sft_stepwise(client, args, row, idx, tokenizer):
    start = time.perf_counter()
    image_contents, image_paths, image_latency = load_images(args, row)
    steps = row["response"]["steps"]
    messages = build_multimodal_messages(
        STEPWISE_WARMUP_SYSTEM_PROMPT,
        build_full_user_prompt(
            row["question"],
            steps,
            row["answer"],
            use_reference_answer=bool(args.sft_use_reference_answer),
        ),
        image_contents,
    )
    warmup_text, warmup_detail, warmup_error = await request_chat(
        client,
        args,
        messages,
        args.sft_warmup_max_tokens,
        "warmup_think",
        tokenizer,
    )
    think_text = extract_think(warmup_text)
    prompt = f"[Question]\n{row['question']}\n"
    if args.sft_use_reference_answer:
        prompt += f"[Answer]\n{row['answer']}\n"
    prompt += f"<think>{think_text}</think>\n[Step judgment]\n"
    pred = []
    request_details = [warmup_detail]
    step_errors = []
    for step_idx, step in enumerate(steps):
        prompt += f"Step {step_idx}: {step}\\boxed{{"
        text, detail, step_error = await request_completion(
            client,
            args,
            prompt,
            f"step_{step_idx}",
            tokenizer,
            guided_choice=["1", "0"],
        )
        score = parse_token_score(text)
        pred.append(score)
        request_details.append(detail)
        step_errors.append(step_error)
        prompt += str(score if score in (0, 1) else 0)
        prompt += "}\n"
    failed_steps = [i for i, value in enumerate(pred) if value not in (0, 1)]
    errors = [f"warmup_error={warmup_error}"] if warmup_error else []
    errors.extend(f"step_{idx}_error={err}" for idx, err in enumerate(step_errors) if err)
    error = "; ".join(errors) if errors else None
    return finalize_sft_row(args, row, idx, pred, request_details, start, image_paths, image_latency, error, bool(failed_steps))


async def evaluate_sft_global_think_stepwise(client, args, row, idx, tokenizer):
    start = time.perf_counter()
    image_contents, image_paths, image_latency = load_images(args, row)
    steps = row["response"]["steps"]
    messages = build_multimodal_messages(
        GLOBAL_THINK_WITH_REFERENCE_SYSTEM_PROMPT if args.sft_use_reference_answer else GLOBAL_THINK_SYSTEM_PROMPT,
        build_user_prompt(
            row["question"],
            steps,
            row["answer"],
            use_reference_answer=bool(args.sft_use_reference_answer),
        ),
        image_contents,
    )
    think_text_raw, think_detail, think_error = await request_chat(
        client,
        args,
        messages,
        args.sft_think_max_tokens,
        "global_think",
        tokenizer,
    )
    think_text = extract_think(think_text_raw) or think_text_raw.strip()
    pred = []
    request_details = [think_detail]
    step_errors = []
    for step_idx in range(len(steps)):
        step_messages = build_step_messages(
            row,
            steps,
            think_text,
            pred,
            step_idx,
            use_reference_answer=bool(args.sft_use_reference_answer),
        )
        text, detail, step_error = await request_chat(
            client,
            args,
            step_messages,
            1,
            f"step_{step_idx}",
            tokenizer,
            guided_choice=["1", "0"],
        )
        score = parse_token_score(text)
        pred.append(score)
        request_details.append(detail)
        step_errors.append(step_error)
    failed_steps = [i for i, value in enumerate(pred) if value not in (0, 1)]
    errors = [f"think_error={think_error}"] if think_error else []
    errors.extend(f"step_{idx}_error={err}" for idx, err in enumerate(step_errors) if err)
    error = "; ".join(errors) if errors else None
    return finalize_sft_row(args, row, idx, pred, request_details, start, image_paths, image_latency, error, bool(failed_steps))


async def safe_evaluate_sft(client, args, row, idx, semaphore, tokenizer):
    async with semaphore:
        try:
            if args.sft_mode == "full":
                return await evaluate_sft_full(client, args, row, idx, tokenizer)
            if args.sft_mode == "direct_scores":
                return await evaluate_sft_direct_scores(client, args, row, idx, tokenizer)
            if args.sft_mode == "no_think_single_pass":
                return await evaluate_sft_no_think_single_pass(client, args, row, idx, tokenizer)
            if args.sft_mode == "stepwise":
                return await evaluate_sft_stepwise(client, args, row, idx, tokenizer)
            return await evaluate_sft_global_think_stepwise(client, args, row, idx, tokenizer)
        except Exception as exc:
            response = row.get("response", {}) if isinstance(row, dict) else {}
            steps = response.get("steps", [])
            return {
                "backend": "sft",
                "mode": args.sft_mode,
                "id": idx,
                "data_source": row.get("data_source") if isinstance(row, dict) else None,
                "policy_model": row.get("policy_model") if isinstance(row, dict) else None,
                "image": row.get("image", []) if isinstance(row, dict) else [],
                "image_paths": [],
                "num_images": 0,
                "num_steps": len(steps),
                "ground_truth": response.get("process_correctness", []),
                "pred": [-2] * len(steps),
                "latency_sec": 0.0,
                "image_encode_latency_sec": 0.0,
                "num_requests": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usage_available_requests": 0},
                "generated_tokens": 0,
                "generated_tokens_source": "none",
                "output_token_estimate": 0,
                "request_details": [],
                "error": repr(exc),
                "parse_error": True,
            }


async def run_sft(args, rows_with_ids: List[Tuple[int, Dict[str, Any]]], output_path: Path):
    from openai import AsyncOpenAI

    base_urls = split_csv(args.sft_base_url)
    client_kwargs = {"api_key": args.sft_api_key}
    if args.sft_request_timeout > 0:
        client_kwargs["timeout"] = args.sft_request_timeout + 10
    clients = [AsyncOpenAI(base_url=url, **client_kwargs) for url in base_urls]
    if args.sft_model == "auto":
        models = await clients[0].models.list()
        if not models.data:
            raise RuntimeError(f"No models returned by {base_urls[0]}/models")
        args.sft_model = models.data[0].id
        print(f"Using deployed SFT model: {args.sft_model}")

    tokenizer = load_optional_tokenizer(args.sft_tokenizer_path)
    semaphore = asyncio.Semaphore(args.sft_concurrency)
    tasks = [
        safe_evaluate_sft(clients[pos % len(clients)], args, row, idx, semaphore, tokenizer)
        for pos, (idx, row) in enumerate(rows_with_ids)
    ]

    use_progress = tqdm is not None and not args.no_progress
    progress = tqdm(total=len(tasks), desc=f"SFT cost ({args.sft_mode})", dynamic_ncols=True) if use_progress else None
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        append_jsonl(result, output_path)
        completed += 1
        if progress is not None:
            progress.update(1)
        elif completed % args.log_every == 0 or completed == len(tasks):
            print(f"[sft] completed {completed}/{len(tasks)}", flush=True)
    if progress is not None:
        progress.close()


def import_visualprm_helpers():
    from visualprm_paper_eval import (
        load_image_for_visualprm,
        load_visualprm_model,
        score_visualprm_steps,
        to_jsonable,
    )

    return load_image_for_visualprm, load_visualprm_model, score_visualprm_steps, to_jsonable


def visualprm_token_count(tokenizer, row: Dict[str, Any]) -> Dict[str, int]:
    steps = row["response"]["steps"]
    question_tokens = count_text_tokens(tokenizer, row.get("question", "")) or 0
    answer_tokens = count_text_tokens(tokenizer, row.get("answer", "")) or 0
    response_tokens = count_text_tokens(tokenizer, "\n\n".join(str(step) for step in steps)) or 0
    return {
        "question_tokens": question_tokens,
        "answer_tokens": answer_tokens,
        "response_tokens": response_tokens,
        "text_input_tokens": question_tokens + response_tokens,
    }


def run_visualprm(args, rows_with_ids: List[Tuple[int, Dict[str, Any]]], output_path: Path):
    import torch

    (
        load_image_for_visualprm,
        load_visualprm_model,
        score_visualprm_steps,
        to_jsonable,
    ) = import_visualprm_helpers()

    tokenizer, model = load_visualprm_model(args.visualprm_model_path, args.visualprm_dtype)
    use_progress = tqdm is not None and not args.no_progress
    progress = tqdm(total=len(rows_with_ids), desc="VisualPRM cost", dynamic_ncols=True) if use_progress else None
    completed = 0
    for idx, row in rows_with_ids:
        start = time.perf_counter()
        error = None
        scores = []
        raw_scores = []
        pred = [-2] * len(row["response"]["steps"])
        image_latency = 0.0
        score_latency = 0.0
        gpu_allocated_mb = None
        gpu_reserved_mb = None
        image_path = Path(args.benchmark_dir) / row["image"][0]
        token_counts = visualprm_token_count(tokenizer, row)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            image_start = time.perf_counter()
            pixel_values = load_image_for_visualprm(image_path).to(getattr(torch, args.visualprm_dtype)).cuda()
            image_latency = time.perf_counter() - image_start
            score_start = time.perf_counter()
            scores, raw_scores = score_visualprm_steps(
                tokenizer,
                model,
                row["question"],
                row["response"]["steps"],
                pixel_values,
            )
            score_latency = time.perf_counter() - score_start
            raw_scores = to_jsonable(raw_scores)
            if len(scores) == len(row["response"]["steps"]):
                pred = [1 if score > args.visualprm_threshold else 0 for score in scores]
            else:
                error = f"score length mismatch: got {len(scores)}, expected {len(row['response']['steps'])}"
            if torch.cuda.is_available():
                gpu_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            error = repr(exc)

        result = {
            "backend": "visualprm",
            "mode": "paper_soft_score",
            "id": idx,
            "data_source": row.get("data_source"),
            "policy_model": row.get("policy_model"),
            "image": row.get("image", []),
            "image_paths": [str(image_path)],
            "num_images": 1,
            "num_steps": len(row["response"]["steps"]),
            "ground_truth": row["response"]["process_correctness"],
            "pred": pred,
            "scores": scores,
            "raw_response": raw_scores,
            "latency_sec": time.perf_counter() - start,
            "image_preprocess_latency_sec": image_latency,
            "score_latency_sec": score_latency,
            "num_requests": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usage_available_requests": 0},
            "generated_tokens": 0,
            "generated_tokens_source": "not_applicable_visualprm_soft_score",
            "num_step_scores": len(scores),
            **token_counts,
            "gpu_max_memory_allocated_mb": gpu_allocated_mb,
            "gpu_max_memory_reserved_mb": gpu_reserved_mb,
            "error": error,
            "parse_error": any(value == -2 for value in pred),
        }
        append_jsonl(result, output_path)
        completed += 1
        if progress is not None:
            progress.update(1)
        elif completed % args.log_every == 0 or completed == len(rows_with_ids):
            print(f"[visualprm] completed {completed}/{len(rows_with_ids)}", flush=True)
    if progress is not None:
        progress.close()


def latest_rows_by_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "id" in row:
            latest[int(row["id"])] = row
    return [latest[idx] for idx in sorted(latest)]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return {"sum": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "sum": sum(vals),
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
    }


def summarize_backend(rows: List[Dict[str, Any]], wall_time_sec: float) -> Dict[str, Any]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usage_available_requests": 0}
    for row in rows:
        row_usage = row.get("usage") or {}
        for key in usage:
            usage[key] += int(row_usage.get(key) or 0)
    num_samples = len(rows)
    num_steps = sum(int(row.get("num_steps") or 0) for row in rows)
    score_outputs = sum(
        int(row.get("num_step_scores") if row.get("num_step_scores") is not None else len(row.get("pred") or []))
        for row in rows
    )
    generated_tokens = sum(int(row.get("generated_tokens") or 0) for row in rows)
    text_input_tokens = sum(int(row.get("text_input_tokens") or 0) for row in rows)
    latency = numeric_summary(row.get("latency_sec", 0.0) for row in rows)
    image_latency = numeric_summary(
        row.get("image_encode_latency_sec", row.get("image_preprocess_latency_sec", 0.0)) for row in rows
    )
    score_latency = numeric_summary(row.get("score_latency_sec", 0.0) for row in rows)
    request_counts = numeric_summary(row.get("num_requests", 0) for row in rows)
    errors = sum(1 for row in rows if row.get("error"))
    parse_errors = sum(1 for row in rows if row.get("parse_error"))
    backend = rows[0].get("backend") if rows else "unknown"
    mode = rows[0].get("mode") if rows else "unknown"
    summary = {
        "backend": backend,
        "mode": mode,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "score_outputs": score_outputs,
        "score_outputs_per_sample": score_outputs / num_samples if num_samples else 0.0,
        "score_outputs_per_step": score_outputs / num_steps if num_steps else 0.0,
        "num_errors": errors,
        "num_parse_errors": parse_errors,
        "wall_time_sec": wall_time_sec,
        "throughput_samples_per_sec": num_samples / wall_time_sec if wall_time_sec > 0 else 0.0,
        "throughput_steps_per_sec": num_steps / wall_time_sec if wall_time_sec > 0 else 0.0,
        "latency_sec": latency,
        "image_latency_sec": image_latency,
        "score_latency_sec": score_latency,
        "requests_per_sample": request_counts,
        "usage": usage,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_tokens / num_samples if num_samples else 0.0,
        "generated_tokens_per_step": generated_tokens / num_steps if num_steps else 0.0,
        "generated_tokens_per_sec": generated_tokens / wall_time_sec if wall_time_sec > 0 else 0.0,
        "text_input_tokens": text_input_tokens,
        "text_input_tokens_per_sample": text_input_tokens / num_samples if num_samples else 0.0,
    }
    try:
        summary["quality_metrics"] = compute_metrics(rows)
    except Exception as exc:
        summary["quality_metrics_error"] = repr(exc)
    return summary


def write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "mode",
        "num_samples",
        "num_steps",
        "score_outputs",
        "score_outputs_per_sample",
        "score_outputs_per_step",
        "num_errors",
        "num_parse_errors",
        "wall_time_sec",
        "throughput_samples_per_sec",
        "throughput_steps_per_sec",
        "latency_mean_sec",
        "latency_p50_sec",
        "latency_p90_sec",
        "latency_p95_sec",
        "latency_p99_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generated_tokens",
        "generated_tokens_per_sample",
        "generated_tokens_per_step",
        "generated_tokens_per_sec",
        "text_input_tokens",
        "text_input_tokens_per_sample",
        "usage_available_requests",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            usage = item.get("usage", {})
            writer.writerow({
                "backend": item.get("backend"),
                "mode": item.get("mode"),
                "num_samples": item.get("num_samples"),
                "num_steps": item.get("num_steps"),
                "score_outputs": item.get("score_outputs"),
                "score_outputs_per_sample": item.get("score_outputs_per_sample"),
                "score_outputs_per_step": item.get("score_outputs_per_step"),
                "num_errors": item.get("num_errors"),
                "num_parse_errors": item.get("num_parse_errors"),
                "wall_time_sec": item.get("wall_time_sec"),
                "throughput_samples_per_sec": item.get("throughput_samples_per_sec"),
                "throughput_steps_per_sec": item.get("throughput_steps_per_sec"),
                "latency_mean_sec": item.get("latency_sec", {}).get("mean"),
                "latency_p50_sec": item.get("latency_sec", {}).get("p50"),
                "latency_p90_sec": item.get("latency_sec", {}).get("p90"),
                "latency_p95_sec": item.get("latency_sec", {}).get("p95"),
                "latency_p99_sec": item.get("latency_sec", {}).get("p99"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "generated_tokens": item.get("generated_tokens"),
                "generated_tokens_per_sample": item.get("generated_tokens_per_sample"),
                "generated_tokens_per_step": item.get("generated_tokens_per_step"),
                "generated_tokens_per_sec": item.get("generated_tokens_per_sec"),
                "text_input_tokens": item.get("text_input_tokens"),
                "text_input_tokens_per_sample": item.get("text_input_tokens_per_sample"),
                "usage_available_requests": usage.get("usage_available_requests"),
            })


def write_sample_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "mode",
        "id",
        "data_source",
        "num_images",
        "num_steps",
        "latency_sec",
        "image_latency_sec",
        "score_latency_sec",
        "num_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generated_tokens",
        "generated_tokens_source",
        "text_input_tokens",
        "num_step_scores",
        "gpu_max_memory_allocated_mb",
        "gpu_max_memory_reserved_mb",
        "error",
        "parse_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage", {})
            writer.writerow({
                "backend": row.get("backend"),
                "mode": row.get("mode"),
                "id": row.get("id"),
                "data_source": row.get("data_source"),
                "num_images": row.get("num_images"),
                "num_steps": row.get("num_steps"),
                "latency_sec": row.get("latency_sec"),
                "image_latency_sec": row.get("image_encode_latency_sec", row.get("image_preprocess_latency_sec", 0.0)),
                "score_latency_sec": row.get("score_latency_sec", 0.0),
                "num_requests": row.get("num_requests"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "generated_tokens": row.get("generated_tokens"),
                "generated_tokens_source": row.get("generated_tokens_source"),
                "text_input_tokens": row.get("text_input_tokens", 0),
                "num_step_scores": row.get("num_step_scores", 0),
                "gpu_max_memory_allocated_mb": row.get("gpu_max_memory_allocated_mb"),
                "gpu_max_memory_reserved_mb": row.get("gpu_max_memory_reserved_mb"),
                "error": row.get("error"),
                "parse_error": row.get("parse_error"),
            })


def write_request_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "phase",
        "api",
        "latency_sec",
        "max_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_token_estimate",
        "output_token_estimate",
        "output_chars",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for detail in row.get("request_details", []):
                usage = detail.get("usage", {})
                writer.writerow({
                    "sample_id": row.get("id"),
                    "phase": detail.get("phase"),
                    "api": detail.get("api"),
                    "latency_sec": detail.get("latency_sec"),
                    "max_tokens": detail.get("max_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "prompt_token_estimate": detail.get("prompt_token_estimate"),
                    "output_token_estimate": detail.get("output_token_estimate"),
                    "output_chars": detail.get("output_chars"),
                    "error": detail.get("error"),
                })


def runtime_path_for_samples(path: Path) -> Path:
    return path.with_name(f"{path.stem}.runtime.json")


def write_runtime(path: Path, wall_time_sec: float) -> None:
    write_json({"wall_time_sec": wall_time_sec}, runtime_path_for_samples(path))


def load_runtime(path: Path, rows: List[Dict[str, Any]]) -> float:
    runtime_path = runtime_path_for_samples(path)
    if runtime_path.exists():
        try:
            with runtime_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            value = float(data.get("wall_time_sec", 0.0))
            if value > 0:
                return value
        except Exception:
            pass
    return sum(float(row.get("latency_sec") or 0.0) for row in rows)


def collect_existing_backend_rows(run_dir: Path) -> List[Tuple[Path, List[Dict[str, Any]], float]]:
    items = []
    sample_paths = sorted(run_dir.glob("sft_*_samples.jsonl"))
    visualprm_path = run_dir / "visualprm_samples.jsonl"
    if visualprm_path.exists():
        sample_paths.append(visualprm_path)
    for path in sample_paths:
        rows = latest_rows_by_id(load_jsonl(path))
        if not rows:
            continue
        items.append((path, rows, load_runtime(path, rows)))
    return items


def sample_rows(args) -> List[Tuple[int, Dict[str, Any]]]:
    rows = load_jsonl(Path(args.benchmark_dir) / "test.jsonl")
    indexed = list(enumerate(rows))
    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(indexed)
    if args.limit and args.limit > 0:
        indexed = indexed[: args.limit]
    return indexed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure compute cost of SFT PRM API and VisualPRM-8B on VisualProcessBench."
    )
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--backends", default=env_default("VPB_BACKENDS", "sft,visualprm"),
                        help="Comma-separated: sft,visualprm")
    parser.add_argument("--limit", type=int, default=int(env_default("VPB_LIMIT", 128)),
                        help="Number of samples. Use 0 for full VisualProcessBench.")
    parser.add_argument("--sample-strategy", choices=("first", "random"), default=env_default("VPB_SAMPLE_STRATEGY", "random"))
    parser.add_argument("--seed", type=int, default=int(env_default("VPB_SEED", 42)))
    parser.add_argument("--output-dir", default=env_default(
        "VPB_COST_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "outputs" / "compute_cost"),
    ))
    parser.add_argument("--run-name", default=env_default("VPB_RUN_NAME", time.strftime("run_%Y%m%d_%H%M%S")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=int(env_default("VPB_LOG_EVERY", 10)))

    parser.add_argument("--sft-base-url", default=env_default("SFT_BASE_URL", env_default("VPB_BASE_URL", "http://127.0.0.1:8000/v1")))
    parser.add_argument("--sft-api-key", default=env_default("SFT_API_KEY", env_default("VPB_API_KEY", "EMPTY")))
    parser.add_argument("--sft-model", default=env_default("SFT_MODEL", env_default("VPB_MODEL", "auto")))
    parser.add_argument("--sft-mode", choices=("global_think_stepwise", "stepwise", "full", "direct_scores", "no_think_single_pass"),
                        default=env_default("SFT_MODE", "global_think_stepwise"))
    parser.add_argument("--sft-concurrency", type=int, default=int(env_default("SFT_CONCURRENCY", env_default("VPB_CONCURRENCY", 1))))
    parser.add_argument("--sft-temperature", type=float, default=float(env_default("SFT_TEMPERATURE", 0.0)))
    parser.add_argument("--sft-use-reference-answer", type=int, choices=(0, 1),
                        default=int(env_default("SFT_USE_REFERENCE_ANSWER", env_default("VPB_USE_REFERENCE_ANSWER", 0))),
                        help="Whether to include VPB reference answers in SFT/VRPRM prompts. Default 0 avoids answer leakage.")
    parser.add_argument("--sft-think-max-tokens", type=int, default=int(env_default("SFT_THINK_MAX_TOKENS", 1024)))
    parser.add_argument("--sft-warmup-max-tokens", type=int, default=int(env_default("SFT_WARMUP_MAX_TOKENS", 1024)))
    parser.add_argument("--sft-full-max-tokens", type=int, default=int(env_default("SFT_FULL_MAX_TOKENS", 2048)))
    parser.add_argument("--sft-direct-max-tokens", type=int, default=int(env_default("SFT_DIRECT_MAX_TOKENS", 512)))
    parser.add_argument("--sft-max-retries", type=int, default=int(env_default("SFT_MAX_RETRIES", 3)))
    parser.add_argument("--sft-request-timeout", type=float, default=float(env_default("SFT_REQUEST_TIMEOUT", 300)))
    parser.add_argument("--sft-tokenizer-path", default=env_default("SFT_TOKENIZER_PATH", ""))

    parser.add_argument("--visualprm-model-path", default=env_default(
        "VISUALPRM_MODEL_PATH",
        "VisualPRM/VisualPRM-8B",
    ))
    parser.add_argument("--visualprm-dtype", choices=("bfloat16", "float16", "float32"),
                        default=env_default("VISUALPRM_DTYPE", env_default("VPB_DTYPE", "bfloat16")))
    parser.add_argument("--visualprm-threshold", type=float,
                        default=float(env_default("VISUALPRM_THRESHOLD", env_default("VPB_THRESHOLD", 0.85))))
    return parser.parse_args()


def main():
    args = parse_args()
    backends = split_csv(args.backends)
    if not backends:
        raise ValueError("--backends must contain at least one backend")
    rows_with_ids = sample_rows(args)
    if not rows_with_ids:
        raise RuntimeError("No VisualProcessBench samples selected")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in run_dir.glob("*"):
            if path.is_file():
                path.unlink()

    metadata = {
        "benchmark_dir": str(args.benchmark_dir),
        "num_selected_samples": len(rows_with_ids),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "selected_ids": [idx for idx, _ in rows_with_ids],
        "backends": backends,
        "sft_base_url": args.sft_base_url,
        "sft_model": args.sft_model,
        "sft_mode": args.sft_mode,
        "sft_concurrency": args.sft_concurrency,
        "sft_use_reference_answer": bool(args.sft_use_reference_answer),
        "visualprm_model_path": args.visualprm_model_path,
        "visualprm_dtype": args.visualprm_dtype,
        "visualprm_threshold": args.visualprm_threshold,
    }
    write_json(metadata, run_dir / "metadata.json")

    if "sft" in backends:
        sft_output = run_dir / f"sft_{args.sft_mode}_samples.jsonl"
        if sft_output.exists() and not args.overwrite:
            raise FileExistsError(f"{sft_output} exists. Use --overwrite.")
        start = time.perf_counter()
        asyncio.run(run_sft(args, rows_with_ids, sft_output))
        wall = time.perf_counter() - start
        write_runtime(sft_output, wall)
        rows = latest_rows_by_id(load_jsonl(sft_output))
        write_sample_csv(run_dir / f"sft_{args.sft_mode}_samples.csv", rows)
        write_request_csv(run_dir / f"sft_{args.sft_mode}_requests.csv", rows)

    if "visualprm" in backends:
        visualprm_output = run_dir / "visualprm_samples.jsonl"
        if visualprm_output.exists() and not args.overwrite:
            raise FileExistsError(f"{visualprm_output} exists. Use --overwrite.")
        start = time.perf_counter()
        run_visualprm(args, rows_with_ids, visualprm_output)
        wall = time.perf_counter() - start
        write_runtime(visualprm_output, wall)
        rows = latest_rows_by_id(load_jsonl(visualprm_output))
        write_sample_csv(run_dir / "visualprm_samples.csv", rows)

    unknown = sorted(set(backends) - {"sft", "visualprm"})
    if unknown:
        raise ValueError(f"Unknown backends: {unknown}")

    summaries = []
    all_rows = []
    for _path, rows, wall_time_sec in collect_existing_backend_rows(run_dir):
        all_rows.extend(rows)
        summaries.append(summarize_backend(rows, wall_time_sec))

    summary = {
        "metadata": metadata,
        "summaries": summaries,
    }
    write_json(summary, run_dir / "summary.json")
    write_summary_csv(run_dir / "summary.csv", summaries)
    write_sample_csv(run_dir / "all_samples.csv", all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"cost benchmark outputs: {run_dir}")


if __name__ == "__main__":
    main()
