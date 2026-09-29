import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

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

SFT_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, generate SFT-quality supervision for a visual reasoning process reward model.

For each solution step, output 1 only when the step is fully supported by the image/problem context and remains logically and mathematically correct. Output 0 when the step is unsupported, contains a visual misunderstanding, uses invalid logic, has a calculation error, contradicts earlier valid reasoning, or depends on a previous incorrect step.

Output requirements:
- Start with [Preliminary thinking] and include exactly one <think>...</think> block.
- Then provide [Step Analysis] with one line per received step. Each line must end with \\boxed{0} or \\boxed{1}.
- Then provide [Final Scores] as strict JSON: {"Score": [comma-separated integer scores]}.
- Then provide [Final Judgment] as strict JSON: {"Judge": 0 or 1}.
- The Score array length must exactly equal the number of received steps.
- Do not add any text after the final JSON object."""

STEPWISE_WARMUP_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, prepare concise reasoning context for judging each solution step.

Output requirements:
- Output exactly one concise <think>...</think> block.
- Focus on the visual evidence, problem constraints, reference answer, and the key checks needed for the candidate solution.
- Do not output per-step scores.
- Do not output final JSON."""


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def encode_image(path):
    path = Path(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{data}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def build_user_prompt(question, steps, answer):
    return (
        f"[Question]\n{question}\n"
        f"[Solution]\n{'<step split>'.join(steps)}\n"
        f"[Answer]\n{answer}\n"
    )


def parse_scores(text, expected_len):
    match = SCORE_RE.search(text or "")
    if not match:
        return [-2] * expected_len
    raw_items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    scores = []
    for item in raw_items:
        try:
            value = float(item)
        except ValueError:
            scores.append(-2)
            continue
        scores.append(1 if value > 0 else 0)
    if len(scores) != expected_len:
        return [-2] * expected_len
    return scores


def parse_scores_detail(text, expected_len):
    scores = parse_scores(text, expected_len)
    parse_error = any(value == -2 for value in scores)
    return scores, parse_error


def extract_think(text):
    match = THINK_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def parse_token_score(text):
    match = TOKEN_SCORE_RE.search(text or "")
    if not match:
        return -2
    return int(match.group(0))


def infer_retry_max_tokens(error):
    match = MAX_TOKENS_ERROR_RE.search(error or "")
    if not match:
        return None
    _requested, model_len, input_len = [int(item) for item in match.groups()]
    return max(1, model_len - input_len)


async def await_with_timeout(request, args):
    if args.request_timeout > 0:
        return await asyncio.wait_for(request, timeout=args.request_timeout)
    return await request


def build_messages(args, row, steps, image_contents, system_prompt):
    user_content = [{"type": "text", "text": build_user_prompt(row["question"], steps, row["answer"])}]
    user_content.extend(image_contents)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_images(args, row):
    image_contents = []
    image_paths = []
    for rel_image in row.get("image", []):
        image_path = Path(args.benchmark_dir) / rel_image
        if image_path.exists():
            image_contents.append(encode_image(image_path))
            image_paths.append(str(image_path))
    return image_contents, image_paths


async def evaluate_one_full(client, args, row, idx, image_contents, image_paths):
    steps = row["response"]["steps"]
    messages = build_messages(args, row, steps, image_contents, SFT_SYSTEM_PROMPT)

    start = time.perf_counter()
    error = None
    text = ""
    max_tokens = args.max_tokens
    attempts = []
    for attempt in range(args.max_retries):
        try:
            request = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=max_tokens,
            )
            response = await await_with_timeout(request, args)
            text = response.choices[0].message.content or ""
            error = None
            attempts.append({"attempt": attempt + 1, "max_tokens": max_tokens, "error": None})
            break
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "max_tokens": max_tokens, "error": error})
            retry_max_tokens = infer_retry_max_tokens(error)
            if retry_max_tokens is not None and retry_max_tokens < max_tokens:
                max_tokens = retry_max_tokens
                continue
            await asyncio.sleep(1.5 * (attempt + 1))

    pred, parse_error = parse_scores_detail(text, len(steps))
    return {
        "id": idx,
        "data_source": row.get("data_source"),
        "policy_model": row.get("policy_model"),
        "image": row.get("image", []),
        "image_paths": image_paths,
        "num_images": len(image_contents),
        "ground_truth": row["response"]["process_correctness"],
        "pred": pred,
        "raw_response": text,
        "latency": time.perf_counter() - start,
        "error": error,
        "parse_error": parse_error,
        "max_tokens": max_tokens,
        "attempts": attempts,
        "reward_mode": "full",
    }


async def get_step_score(client, args, prompt):
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            request = client.completions.create(
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=1,
                extra_body={"guided_choice": ["1", "0"]},
            )
            response = await await_with_timeout(request, args)
            text = response.choices[0].text or ""
            pred = parse_token_score(text)
            if pred in (0, 1):
                attempts.append({"attempt": attempt + 1, "raw": text, "error": None})
                return pred, text, None, attempts
            error = f"Could not parse guided score token from {text!r}"
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        except Exception as exc:
            error = repr(exc)
            attempts.append({"attempt": attempt + 1, "raw": text, "error": error})
        await asyncio.sleep(1.5 * (attempt + 1))
    return -2, text, error, attempts


def build_stepwise_seed_prompt(row, think_text):
    prompt = ""
    prompt += f"[Question]\n{row['question']}\n"
    prompt += f"[Answer]\n{row['answer']}\n"
    if think_text:
        prompt += f"<think>{think_text}</think>\n"
    prompt += "[Step judgment]\n"
    return prompt


async def evaluate_one_stepwise(client, args, row, idx, image_contents, image_paths):
    steps = row["response"]["steps"]
    messages = build_messages(args, row, steps, image_contents, STEPWISE_WARMUP_SYSTEM_PROMPT)

    start = time.perf_counter()
    warmup_error = None
    warmup_text = ""
    warmup_attempts = []
    for attempt in range(args.max_retries):
        try:
            request = client.chat.completions.create(
                model=args.model,
                messages=messages,
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

    think_text = extract_think(warmup_text)
    prompt = build_stepwise_seed_prompt(row, think_text)
    pred = []
    step_raw_responses = []
    step_errors = []
    step_attempts = []
    for step_idx, step in enumerate(steps):
        prompt += f"Step {step_idx}: {step}\\boxed{{"
        score, raw_text, step_error, attempts = await get_step_score(client, args, prompt)
        pred.append(score)
        step_raw_responses.append(raw_text)
        step_errors.append(step_error)
        step_attempts.append(attempts)
        prompt += str(score if score in (0, 1) else 0)
        prompt += "}\n"

    failed_steps = [i for i, value in enumerate(pred) if value not in (0, 1)]
    parse_error = bool(failed_steps)
    error_parts = []
    if warmup_error:
        error_parts.append(f"warmup_error={warmup_error}")
    if failed_steps:
        error_parts.append(f"failed_steps={failed_steps}")
    error = "; ".join(error_parts) if error_parts else None
    return {
        "id": idx,
        "data_source": row.get("data_source"),
        "policy_model": row.get("policy_model"),
        "image": row.get("image", []),
        "image_paths": image_paths,
        "num_images": len(image_contents),
        "ground_truth": row["response"]["process_correctness"],
        "pred": pred,
        "raw_response": warmup_text,
        "step_raw_responses": step_raw_responses,
        "latency": time.perf_counter() - start,
        "error": error,
        "parse_error": parse_error,
        "max_tokens": args.warmup_max_tokens,
        "attempts": warmup_attempts,
        "step_attempts": step_attempts,
        "step_errors": step_errors,
        "reward_mode": "stepwise",
    }


async def evaluate_one(client, args, row, idx, semaphore):
    async with semaphore:
        image_contents, image_paths = load_images(args, row)
        if args.reward_mode == "full":
            return await evaluate_one_full(client, args, row, idx, image_contents, image_paths)
        return await evaluate_one_stepwise(client, args, row, idx, image_contents, image_paths)


async def safe_evaluate_one(client, args, row, idx, semaphore):
    try:
        return await evaluate_one(client, args, row, idx, semaphore)
    except Exception as exc:
        response = row.get("response", {}) if isinstance(row, dict) else {}
        steps = response.get("steps", [])
        return {
            "id": idx,
            "data_source": row.get("data_source") if isinstance(row, dict) else None,
            "policy_model": row.get("policy_model") if isinstance(row, dict) else None,
            "image": row.get("image", []) if isinstance(row, dict) else [],
            "image_paths": [],
            "num_images": 0,
            "ground_truth": response.get("process_correctness", []),
            "pred": [-2] * len(steps),
            "raw_response": "",
            "latency": 0.0,
            "error": repr(exc),
            "parse_error": True,
            "max_tokens": args.max_tokens,
            "attempts": [],
            "reward_mode": args.reward_mode,
        }


def latest_rows_by_id(rows):
    by_id = {}
    duplicates = 0
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            continue
        duplicates += int(row_id in by_id)
        by_id[int(row_id)] = row
    return by_id, duplicates


def should_skip_existing(row, args):
    if row is None:
        return False
    existing_mode = row.get("reward_mode")
    if existing_mode != args.reward_mode:
        if not (existing_mode is None and args.reward_mode == "full"):
            return False
    if args.retry_errors and row.get("error"):
        return False
    if args.retry_invalid and row.get("parse_error"):
        return False
    return True


def write_status(args, status_path, **status):
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["base_url"] = args.base_url
    status["model"] = args.model
    status["reward_mode"] = args.reward_mode
    status["output"] = str(args.output)
    write_json(status, status_path)


def log_error(result, error_log_path):
    if not error_log_path:
        return
    if not result.get("error") and not result.get("parse_error"):
        return
    append_jsonl({
        "id": result.get("id"),
        "data_source": result.get("data_source"),
        "image": result.get("image", []),
        "num_images": result.get("num_images"),
        "error": result.get("error"),
        "parse_error": result.get("parse_error"),
        "pred": result.get("pred"),
        "latency": result.get("latency"),
        "max_tokens": result.get("max_tokens"),
        "attempts": result.get("attempts", []),
        "step_attempts": result.get("step_attempts", []),
        "step_errors": result.get("step_errors", []),
        "reward_mode": result.get("reward_mode"),
        "raw_response": result.get("raw_response", ""),
        "step_raw_responses": result.get("step_raw_responses", []),
    }, error_log_path)


async def run(args):
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.warmup_max_tokens < 1:
        raise ValueError("--warmup-max-tokens must be >= 1")
    if args.status_every < 1:
        raise ValueError("--status-every must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    rows = load_jsonl(Path(args.benchmark_dir) / "test.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    error_log_path = Path(args.error_log) if args.error_log else output_path.with_suffix(".errors.jsonl")
    status_path = Path(args.status_file) if args.status_file else output_path.with_suffix(".status.json")

    if args.overwrite:
        for path in (output_path, error_log_path, status_path):
            if path.exists():
                path.unlink()
    elif output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

    existing_by_id, duplicates = latest_rows_by_id(load_jsonl(output_path))
    pending = [
        (idx, row)
        for idx, row in enumerate(rows)
        if not should_skip_existing(existing_by_id.get(idx), args)
    ]
    skipped = len(rows) - len(pending)

    client_kwargs = {"base_url": args.base_url, "api_key": args.api_key}
    if args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout + 10
    client = AsyncOpenAI(**client_kwargs)
    if args.model == "auto":
        models = await client.models.list()
        if not models.data:
            raise RuntimeError(f"No models returned by {args.base_url}/models")
        args.model = models.data[0].id
        print(f"Using deployed model: {args.model}")

    write_status(
        args,
        status_path,
        total=len(rows),
        skipped=skipped,
        pending=len(pending),
        completed=0,
        errors=0,
        parse_errors=0,
        duplicates_in_existing=duplicates,
        running=True,
    )

    print(
        f"Loaded {len(rows)} samples, skipped {skipped}, pending {len(pending)}. "
        f"Output: {output_path}"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [safe_evaluate_one(client, args, row, idx, semaphore) for idx, row in pending]

    completed = 0
    errors = 0
    parse_errors = 0
    use_progress = tqdm is not None and not args.no_progress
    progress = tqdm(total=len(tasks), desc="Evaluating", dynamic_ncols=True) if use_progress else None
    for task in asyncio.as_completed(tasks):
        result = await task
        append_jsonl(result, output_path)
        log_error(result, error_log_path)
        completed += 1
        errors += int(bool(result.get("error")))
        parse_errors += int(bool(result.get("parse_error")))
        if progress is not None:
            progress.update(1)
            progress.set_postfix(errors=errors, parse_errors=parse_errors, skipped=skipped)
        if not use_progress and (completed % args.log_every == 0 or completed == len(tasks)):
            print(f"completed {completed}/{len(tasks)} errors={errors} parse_errors={parse_errors}")
        if completed % args.status_every == 0 or completed == len(tasks):
            write_status(
                args,
                status_path,
                total=len(rows),
                skipped=skipped,
                pending=len(pending),
                completed=completed,
                errors=errors,
                parse_errors=parse_errors,
                duplicates_in_existing=duplicates,
                running=completed < len(tasks),
            )
    if progress is not None:
        progress.close()

    pred_by_id, final_duplicates = latest_rows_by_id(load_jsonl(output_path))
    pred_rows = [pred_by_id[idx] for idx in sorted(pred_by_id) if idx < len(rows)]
    metrics = compute_metrics(pred_rows)
    metrics_path = output_path.with_suffix(".metrics.json")
    write_json(metrics, metrics_path)
    write_status(
        args,
        status_path,
        total=len(rows),
        skipped=skipped,
        pending=len(pending),
        completed=completed,
        completed_total=len(pred_rows),
        errors=errors,
        parse_errors=parse_errors,
        duplicates_in_existing=duplicates,
        duplicates_final=final_duplicates,
        running=False,
        metrics=str(metrics_path),
        error_log=str(error_log_path),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"predictions: {output_path}")
    print(f"errors: {error_log_path}")
    print(f"status: {status_path}")
    print(f"metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a generative SFT PRM on VisualProcessBench via OpenAI API.")
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="auto", help='Model id. Use "auto" to read the first id from /v1/models.')
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "sft_stepwise_predictions.jsonl"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup-max-tokens", type=int, default=1024)
    parser.add_argument("--reward-mode", choices=("stepwise", "full"), default="stepwise")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Per-request timeout in seconds. Set <=0 to disable.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the output JSONL.")
    parser.add_argument("--retry-errors", action="store_true", help="When resuming, retry rows whose previous result has error.")
    parser.add_argument("--retry-invalid", action="store_true", help="When resuming, retry rows whose previous result failed Score parsing.")
    parser.add_argument("--error-log", default=None, help="JSONL path for error and parse-error records.")
    parser.add_argument("--status-file", default=None, help="JSON status checkpoint path.")
    return parser.parse_args()


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
