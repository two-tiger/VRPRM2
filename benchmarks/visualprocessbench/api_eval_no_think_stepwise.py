import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from api_eval_sft import (  # noqa: E402
    DEFAULT_BENCH_DIR,
    append_jsonl,
    await_with_timeout,
    encode_image,
    infer_retry_max_tokens,
    latest_rows_by_id,
    load_jsonl,
    log_error,
    parse_scores_detail,
    parse_token_score,
    should_skip_existing,
    write_status,
)
from metrics import compute_metrics, write_json  # noqa: E402


REWARD_MODE_PREFIX = "no_think"


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


STEP_SCORE_SYSTEM_PROMPT = """You are a visual process reward model. Judge whether the current candidate solution step is correct in context.

Do not reason out loud. Do not explain.
Output only one token: 1 or 0.

Scoring policy:
- Output 1 only if the current step is fully supported by the image/problem context and remains logically and mathematically correct.
- Output 0 if the step is unsupported, visually mistaken, logically invalid, computationally wrong, contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery."""


def split_base_urls(base_url: str) -> list[str]:
    urls = [url.strip() for url in str(base_url).split(",") if url.strip()]
    return urls or ["http://localhost:8000/v1"]


def load_images(args, row):
    image_contents = []
    image_paths = []
    for rel_image in row.get("image", []):
        image_path = Path(args.benchmark_dir) / rel_image
        if image_path.exists():
            image_contents.append(encode_image(image_path))
            image_paths.append(str(image_path))
    return image_contents, image_paths


def numbered_steps(steps):
    return [
        step if str(step).strip().lower().startswith(f"step {idx}:") else f"Step {idx}: {step}"
        for idx, step in enumerate(steps)
    ]


def build_single_pass_messages(row, steps, image_contents, use_reference_answer=False):
    step_texts = numbered_steps(steps)
    user_text = f"[Question]\n{row['question']}\n"
    if use_reference_answer:
        user_text += f"[Reference Answer]\n{row['answer']}\n"
    user_text += (
        f"[Candidate Solution]\n{chr(10).join(step_texts)}\n"
        f"[Number of Steps]\n{len(step_texts)}\n"
        'Directly output strict JSON in this exact form: {"Score": [0 or 1 for each step]}.'
    )
    user_content = [{"type": "text", "text": user_text}]
    user_content.extend(image_contents)
    return [
        {
            "role": "system",
            "content": (
                NO_THINK_SINGLE_PASS_WITH_REFERENCE_SYSTEM_PROMPT
                if use_reference_answer
                else NO_THINK_SINGLE_PASS_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_step_messages(
    row,
    steps,
    previous_judgments,
    current_idx,
    image_contents,
    include_previous=True,
    step_image_mode="never",
    use_reference_answer=False,
):
    step_texts = numbered_steps(steps)
    previous_text = "None"
    if include_previous and previous_judgments:
        previous_text = "\n".join(
            f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_judgments)
        )

    user_text = f"[Question]\n{row['question']}\n"
    if use_reference_answer:
        user_text += f"[Reference Answer]\n{row['answer']}\n"
    user_text += f"[Candidate Solution]\n{chr(10).join(step_texts)}\n"
    if include_previous:
        user_text += f"[Previous Step Judgments]\n{previous_text}\n"
    user_text += (
        f"[Current Step]\n{step_texts[current_idx]}\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )

    user_content = user_text
    if step_image_mode == "each_step":
        user_content = [{"type": "text", "text": user_text}]
        user_content.extend(image_contents)
    return [
        {"role": "system", "content": STEP_SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def request_chat(client, args, messages, max_tokens=1, guided_choice=None):
    kwargs = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    if guided_choice:
        kwargs["extra_body"] = {"guided_choice": guided_choice}
    response = await await_with_timeout(client.chat.completions.create(**kwargs), args)
    return response.choices[0].message.content or ""


async def get_step_score(client, args, messages):
    text = ""
    error = None
    attempts = []
    for attempt in range(args.max_retries):
        try:
            text = await request_chat(
                client,
                args,
                messages,
                max_tokens=1,
                guided_choice=["1", "0"],
            )
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


async def get_single_pass_scores(client, args, messages, expected_len):
    text = ""
    error = None
    attempts = []
    max_tokens = args.max_tokens
    for attempt in range(args.max_retries):
        try:
            text = await request_chat(
                client,
                args,
                messages,
                max_tokens=max_tokens,
            )
            pred, parse_error = parse_scores_detail(text, expected_len)
            error = None
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "max_tokens": max_tokens,
                    "raw": text,
                    "parse_error": parse_error,
                    "error": None,
                }
            )
            if not parse_error:
                return pred, text, None, False, max_tokens, attempts
            error = f"Could not parse Score array with length {expected_len}"
        except Exception as exc:
            error = repr(exc)
            retry_max_tokens = infer_retry_max_tokens(error)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "max_tokens": max_tokens,
                    "raw": text,
                    "parse_error": True,
                    "error": error,
                }
            )
            if retry_max_tokens is not None and retry_max_tokens < max_tokens:
                max_tokens = retry_max_tokens
                continue
        await asyncio.sleep(1.5 * (attempt + 1))
    pred, parse_error = parse_scores_detail(text, expected_len)
    if parse_error:
        pred = [-2] * expected_len
    return pred, text, error, parse_error, max_tokens, attempts


async def evaluate_one_single_pass(client, args, row, idx):
    start = time.perf_counter()
    steps = row["response"]["steps"]
    image_contents, image_paths = load_images(args, row)
    messages = build_single_pass_messages(
        row,
        steps,
        image_contents,
        use_reference_answer=bool(args.use_reference_answer),
    )
    pred, raw_text, error, parse_error, max_tokens, attempts = await get_single_pass_scores(
        client,
        args,
        messages,
        len(steps),
    )
    return {
        "id": idx,
        "data_source": row.get("data_source"),
        "policy_model": row.get("policy_model"),
        "image": row.get("image", []),
        "image_paths": image_paths,
        "num_images": len(image_contents),
        "ground_truth": row["response"]["process_correctness"],
        "pred": pred,
        "raw_response": raw_text,
        "global_thinking": "",
        "step_raw_responses": [],
        "latency": time.perf_counter() - start,
        "error": error,
        "parse_error": parse_error,
        "max_tokens": max_tokens,
        "attempts": attempts,
        "step_attempts": [],
        "step_errors": [],
        "include_previous_judgments": False,
        "step_image_mode": "once",
        "no_think_mode": "single_pass",
        "use_reference_answer": bool(args.use_reference_answer),
        "reward_mode": args.reward_mode,
    }


async def evaluate_one_stepwise(client, args, row, idx):
    start = time.perf_counter()
    steps = row["response"]["steps"]
    if args.step_image_mode == "each_step":
        image_contents, image_paths = load_images(args, row)
    else:
        image_contents, image_paths = [], []

    pred = []
    step_raw_responses = []
    step_errors = []
    step_attempts = []
    for step_idx in range(len(steps)):
        messages = build_step_messages(
            row,
            steps,
            pred,
            step_idx,
            image_contents,
                include_previous=bool(args.include_previous_judgments),
                step_image_mode=args.step_image_mode,
                use_reference_answer=bool(args.use_reference_answer),
            )
        score, raw_text, step_error, attempts = await get_step_score(client, args, messages)
        pred.append(score)
        step_raw_responses.append(raw_text)
        step_errors.append(step_error)
        step_attempts.append(attempts)

    failed_steps = [i for i, value in enumerate(pred) if value not in (0, 1)]
    error_parts = []
    if failed_steps:
        error_parts.append(f"failed_steps={failed_steps}")
    error_parts.extend(f"step_{idx}_error={err}" for idx, err in enumerate(step_errors) if err)
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
        "raw_response": "",
        "global_thinking": "",
        "step_raw_responses": step_raw_responses,
        "latency": time.perf_counter() - start,
        "error": error,
        "parse_error": bool(failed_steps),
        "max_tokens": 1,
        "attempts": [],
        "step_attempts": step_attempts,
        "step_errors": step_errors,
        "include_previous_judgments": bool(args.include_previous_judgments),
        "step_image_mode": args.step_image_mode,
        "no_think_mode": "stepwise",
        "use_reference_answer": bool(args.use_reference_answer),
        "reward_mode": args.reward_mode,
    }


async def evaluate_one(client, args, row, idx, semaphore):
    async with semaphore:
        if args.no_think_mode == "single_pass":
            return await evaluate_one_single_pass(client, args, row, idx)
        return await evaluate_one_stepwise(client, args, row, idx)


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
            "global_thinking": "",
            "step_raw_responses": [],
            "latency": 0.0,
            "error": repr(exc),
            "parse_error": True,
            "max_tokens": args.max_tokens if args.no_think_mode == "single_pass" else 1,
            "attempts": [],
            "step_attempts": [],
            "step_errors": [],
            "include_previous_judgments": bool(args.include_previous_judgments),
            "step_image_mode": "once" if args.no_think_mode == "single_pass" else args.step_image_mode,
            "no_think_mode": args.no_think_mode,
            "use_reference_answer": bool(args.use_reference_answer),
            "reward_mode": args.reward_mode,
        }


def should_skip_no_think_existing(row, args):
    if row is None:
        return False
    if row.get("reward_mode") != args.reward_mode:
        return False
    if row.get("no_think_mode") != args.no_think_mode:
        return False
    if bool(row.get("use_reference_answer", True)) != bool(args.use_reference_answer):
        return False
    if args.no_think_mode == "stepwise":
        if bool(row.get("include_previous_judgments", True)) != bool(args.include_previous_judgments):
            return False
        if row.get("step_image_mode") != args.step_image_mode:
            return False
    return should_skip_existing(row, args)


async def run(args):
    args.reward_mode = f"{REWARD_MODE_PREFIX}_{args.no_think_mode}"
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")
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
        if not should_skip_no_think_existing(existing_by_id.get(idx), args)
    ]
    skipped = len(rows) - len(pending)

    base_urls = split_base_urls(args.base_url)
    client_kwargs = {"api_key": args.api_key}
    if args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout + 10
    clients = [AsyncOpenAI(base_url=base_url, **client_kwargs) for base_url in base_urls]
    if args.model == "auto":
        models = await clients[0].models.list()
        if not models.data:
            raise RuntimeError(f"No models returned by {base_urls[0]}/models")
        args.model = models.data[0].id
        print(f"Using deployed model: {args.model}")
    if len(base_urls) > 1:
        print(f"Using {len(base_urls)} OpenAI-compatible endpoints: {', '.join(base_urls)}")

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
    tasks = [
        safe_evaluate_one(clients[idx % len(clients)], args, row, idx, semaphore)
        for idx, row in pending
    ]

    completed = errors = parse_errors = 0
    use_progress = tqdm is not None and not args.no_progress
    progress = tqdm(total=len(tasks), desc="NoThinkStepwise", dynamic_ncols=True) if use_progress else None
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
        elif completed % args.log_every == 0 or completed == len(tasks):
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
    parser = argparse.ArgumentParser(
        description="Evaluate VisualProcessBench with no-thinking scoring."
    )
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="auto", help='Model id. Use "auto" to read the first id from /v1/models.')
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "outputs" / "no_think_single_pass_no_ref_predictions.jsonl"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument(
        "--use-reference-answer",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether to include the VPB reference answer in model prompts. Default 0 avoids answer leakage.",
    )
    parser.add_argument(
        "--no-think-mode",
        choices=("single_pass", "stepwise"),
        default="single_pass",
        help=(
            "single_pass sends one multimodal request per sample and directly outputs all step scores. "
            "stepwise keeps the older one-request-per-step ablation."
        ),
    )
    parser.add_argument("--include-previous-judgments", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--step-image-mode",
        choices=("never", "each_step"),
        default="never",
        help=(
            "Only used by --no-think-mode stepwise. Whether to attach image(s) to every "
            "step-scoring request."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--error-log", default=None)
    parser.add_argument("--status-file", default=None)
    return parser.parse_args()


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
