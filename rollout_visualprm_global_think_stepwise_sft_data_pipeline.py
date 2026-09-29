import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rollout_sft_data_pipeline import (  # noqa: E402
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OUTPUT_DIR,
    allocate_even_quotas,
    extract_reasoning_from_message,
    iter_input_files,
    load_json,
    normalize_samples,
    parse_api_extra_body,
    sample_records,
    save_filed_sample_json,
    save_json,
    truncate_reasoning_text,
)


HIDDEN_REASONING_SYSTEM_PROMPT = """You are a visual process reward model teacher. Analyze the image(s), question, reference answer, and complete candidate solution before any step-level supervision is created.

Focus your internal reasoning on:
- key visual evidence needed to solve the problem;
- the question goal and reference-answer constraint;
- dependencies between candidate steps;
- the first consequential incorrect step if one exists;
- later steps that are correct independently or are invalid because they rely on an earlier error.

After reasoning, output strict JSON only:
{"Score": [one integer 0 or 1 for each candidate step]}

The Score array length must exactly equal the number of candidate steps."""


GLOBAL_THINK_COMPRESS_SYSTEM_PROMPT = """You compress a teacher model's hidden reasoning into one concise global thinking block for SFT.

Output requirements:
- Output exactly one <think>...</think> block.
- Preserve the key visual evidence, question goal, reference-answer constraint, dependencies between steps, and the main correctness issue.
- Do not output step scores.
- Do not output JSON.
- Keep the thinking focused and no more than {max_words} words."""


GLOBAL_THINK_SFT_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, first produce concise global thinking for later step-level scoring.

Output requirements:
- Output exactly one <think>...</think> block.
- Identify the key visual evidence, the question goal, the reference-answer constraint, dependencies between candidate steps, and the main correctness issue.
- Do not output step scores.
- Do not output JSON."""


STEP_LEVEL_SFT_SYSTEM_PROMPT = """You are a visual process reward model. Judge whether the current candidate solution step is correct in context.

Use the image(s), problem, reference answer, complete candidate solution, global thinking, and previous step judgments as context.

Output only one token: 1 or 0.

Scoring policy:
- Output 1 only if the current step is fully supported by the image/problem context and remains logically and mathematically correct.
- Output 0 if the step is unsupported, visually mistaken, logically invalid, computationally wrong, contradicts earlier valid reasoning, or relies on a previous incorrect step without recovery.
- Do not output <think> tags, explanations, JSON, or extra text."""


THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
SCORE_RE = re.compile(r'\{\s*"Score"\s*:\s*\[([^\]]*)\]\s*\}')
READY_RE = re.compile(r"\bREADY\b", re.IGNORECASE)


def step_scores(raw_sample):
    scores = []
    for item in raw_sample.get("steps_with_score") or []:
        if not isinstance(item, dict):
            continue
        try:
            scores.append(float(item.get("score", 0)))
        except (TypeError, ValueError):
            continue
    return scores


def is_negative_sample(raw_sample, mode, negative_threshold):
    scores = step_scores(raw_sample)
    if not scores:
        return False

    if mode == "any_step":
        return any(score <= negative_threshold for score in scores)
    if mode == "final_step":
        return scores[-1] <= negative_threshold
    if mode == "all_steps":
        return all(score <= negative_threshold for score in scores)
    raise ValueError(f"Unknown negative mode: {mode}")


def is_positive_sample(raw_sample, positive_threshold):
    scores = step_scores(raw_sample)
    return bool(scores) and all(score >= positive_threshold for score in scores)


def filter_samples(raw_samples, polarity, negative_mode, negative_threshold, positive_threshold):
    if polarity == "all":
        return [
            sample
            for sample in raw_samples
            if any(score <= negative_threshold or score >= positive_threshold for score in step_scores(sample))
        ]
    if polarity == "negative":
        return [
            sample
            for sample in raw_samples
            if is_negative_sample(sample, mode=negative_mode, negative_threshold=negative_threshold)
        ]
    if polarity == "positive":
        return [
            sample
            for sample in raw_samples
            if is_positive_sample(sample, positive_threshold=positive_threshold)
        ]
    raise ValueError(f"Unknown polarity: {polarity}")


def attach_source_scores(sample):
    item = sample.copy()
    messages = [message.copy() for message in item.get("messages", [])]
    if len(messages) < 3:
        return item

    try:
        payload = json.loads(messages[2].get("content", "").replace("'", '"'))
        scores = [float(score) for score in payload["Score"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return item

    item["source_scores"] = scores
    item["messages"] = messages
    return item


def load_selected_samples(
    input_path,
    image_root,
    sample_total,
    samples_per_annotation,
    seed,
    negative_mode,
    negative_score_threshold,
    positive_score_threshold,
    polarity,
):
    groups = []
    total_raw = 0
    total_selected_raw = 0

    for file_path in iter_input_files(input_path):
        raw_samples = load_json(file_path)
        selected_raw = filter_samples(
            raw_samples,
            polarity=polarity,
            negative_mode=negative_mode,
            negative_threshold=negative_score_threshold,
            positive_threshold=positive_score_threshold,
        )
        normalized = normalize_samples(
            selected_raw,
            image_root,
            source_annotation=str(file_path),
        )
        normalized = [attach_source_scores(sample) for sample in normalized]
        total_raw += len(raw_samples)
        total_selected_raw += len(selected_raw)
        groups.append((file_path, raw_samples, selected_raw, normalized))

    if samples_per_annotation is not None and samples_per_annotation > 0:
        quotas = [min(len(normalized), samples_per_annotation) for _, _, _, normalized in groups]
    else:
        quotas = allocate_even_quotas([len(normalized) for _, _, _, normalized in groups], sample_total)

    data = []
    selected_total = 0
    for (file_path, raw_samples, selected_raw, normalized), quota in zip(groups, quotas):
        rng = random.Random(f"{seed}:{file_path}")
        selected = [] if quota <= 0 else sample_records(normalized, quota, rng)
        selected_total += len(selected)
        print(
            f"Loaded {file_path}: raw={len(raw_samples)}, "
            f"{polarity}_raw={len(selected_raw)}, normalized={len(normalized)}, "
            f"quota={quota}, selected={len(selected)}"
        )
        data.extend(selected)

    random.Random(seed).shuffle(data)
    if total_raw:
        print(
            f"Total raw samples: {total_raw}; selected raw samples: {total_selected_raw} "
            f"({total_selected_raw / total_raw * 100:.2f}%)"
        )
    print(f"Total selected samples: {selected_total}")
    return data


def split_user_content(user_text):
    try:
        question = user_text.split("[Question]\n", 1)[1].split("\n[Solution]\n", 1)[0].strip()
        solution = user_text.split("\n[Solution]\n", 1)[1].split("\n[Answer]\n", 1)[0].strip()
        answer = user_text.split("\n[Answer]\n", 1)[1].strip()
    except IndexError as exc:
        raise ValueError("User content does not match [Question]/[Solution]/[Answer] format.") from exc

    steps = [step.strip() for step in solution.split("<step split>") if step.strip()]
    return question, steps, answer


def load_source_scores(sample):
    if sample.get("source_scores"):
        return [float(score) for score in sample["source_scores"]]
    try:
        payload = json.loads(sample["messages"][2]["content"].replace("'", '"'))
        return [float(score) for score in payload["Score"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Could not parse source Score labels from sample.") from exc


def build_conservative_labels(scores, negative_threshold, positive_threshold):
    labels = []
    for score in scores:
        if score <= negative_threshold:
            labels.append(0)
        elif score >= positive_threshold:
            labels.append(1)
        else:
            labels.append(None)
    return labels


def parse_score_list(text, expected_len):
    match = SCORE_RE.search(text or "")
    if not match:
        raise ValueError("missing_kimi_score_json")
    raw_items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    scores = []
    for item in raw_items:
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"bad_kimi_score_item:{item!r}") from exc
        if value not in (0, 1):
            raise ValueError(f"bad_kimi_score_value:{value}")
        scores.append(int(value))
    if len(scores) != expected_len:
        raise ValueError(f"kimi_score_length_mismatch:{len(scores)}!={expected_len}")
    return scores


def validate_kimi_scores(kimi_scores, mc_labels):
    mismatches = []
    confident_count = 0
    for idx, label in enumerate(mc_labels):
        if label is None:
            continue
        confident_count += 1
        if int(kimi_scores[idx]) != int(label):
            mismatches.append({
                "step_index": idx,
                "kimi_score": int(kimi_scores[idx]),
                "mc_label": int(label),
            })
    if confident_count == 0:
        raise ValueError("no_confident_mc_steps")
    if mismatches:
        raise ValueError(f"kimi_mc_score_mismatch:{json.dumps(mismatches, ensure_ascii=False)}")
    return confident_count


def numbered_steps(steps):
    result = []
    for idx, step in enumerate(steps):
        if step.strip().lower().startswith(f"step {idx}:"):
            result.append(step.strip())
        else:
            result.append(f"Step {idx}: {step.strip()}")
    return result


def build_full_problem_text(question, steps, answer):
    return (
        f"[Question]\n{question}\n"
        f"[Reference Answer]\n{answer}\n"
        f"[Candidate Solution]\n{chr(10).join(numbered_steps(steps))}\n"
    )


def extract_think_text(text):
    match = THINK_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def normalize_global_thinking(text, max_chars):
    thinking = extract_think_text(text) or (text or "").strip()
    thinking = re.sub(r"</?think>", "", thinking).strip()
    thinking = READY_RE.sub("", thinking).strip()
    thinking = re.sub(r"\n{3,}", "\n\n", thinking)
    thinking = truncate_reasoning_text(thinking, max_chars=max_chars)
    if not thinking:
        return ""
    return thinking


def validate_global_thinking(global_thinking, min_chars):
    normalized = (global_thinking or "").strip()
    if normalized.upper() == "READY":
        raise ValueError("global_thinking_is_ready")
    if "READY" in normalized.upper():
        raise ValueError("global_thinking_contains_ready")
    if len(normalized) < min_chars:
        raise ValueError(f"global_thinking_too_short:{len(normalized)}<{min_chars}")


def build_step_user_text(question, steps, answer, global_thinking, previous_labels, step_idx):
    step_lines = numbered_steps(steps)
    previous = "\n".join(
        f"Step {idx}: \\boxed{{{label}}}" for idx, label in enumerate(previous_labels)
    )
    if not previous:
        previous = "None"
    return (
        f"[Question]\n{question}\n"
        f"[Reference Answer]\n{answer}\n"
        f"[Candidate Solution]\n{chr(10).join(step_lines)}\n"
        f"[Global Thinking]\n<think>{global_thinking}</think>\n"
        f"[Previous Step Judgments]\n{previous}\n"
        f"[Current Step]\n{step_lines[step_idx]}\n"
        "Is the current step correct in context? Output only one token: 1 or 0."
    )


def apply_step_balance(samples, target_negative_ratio, seed):
    if target_negative_ratio is None or target_negative_ratio <= 0:
        return samples

    global_samples = [sample for sample in samples if sample.get("task_type") == "global_thinking"]
    step_samples = [sample for sample in samples if sample.get("task_type") == "step_score"]
    other_samples = [
        sample
        for sample in samples
        if sample.get("task_type") not in {"global_thinking", "step_score"}
    ]

    negative = [sample for sample in step_samples if sample.get("step_label") == 0]
    positive = [sample for sample in step_samples if sample.get("step_label") == 1]
    if not negative or not positive:
        return samples

    target_negative_ratio = min(max(target_negative_ratio, 0.01), 0.99)
    max_positive = int(len(negative) * (1 - target_negative_ratio) / target_negative_ratio)
    if max_positive <= 0 or len(positive) <= max_positive:
        return samples

    rng = random.Random(seed)
    kept_positive = rng.sample(positive, max_positive)
    balanced_steps = negative + kept_positive
    rng.shuffle(balanced_steps)
    return global_samples + balanced_steps + other_samples


class AsyncGlobalThinkStepwiseRolloutor:
    def __init__(self, args):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Please install the openai package before running rollout generation.") from exc

        self.args = args
        self.client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        self.log_lock = asyncio.Lock()

    def format_api_error(self, error, attempt=None):
        error_info = {
            "attempt": attempt,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        for attr in ("status_code", "code", "type", "param", "request_id"):
            value = getattr(error, attr, None)
            if value is not None:
                error_info[attr] = value
        body = getattr(error, "body", None)
        if body is not None:
            error_info["body"] = body
        response = getattr(error, "response", None)
        if response is not None:
            error_info["response_status_code"] = getattr(response, "status_code", None)
            error_info["response_headers"] = dict(getattr(response, "headers", {}) or {})
            try:
                error_info["response_json"] = response.json()
            except Exception:
                text = getattr(response, "text", None)
                if text is not None:
                    error_info["response_text"] = text
        return error_info

    async def append_log(self, sample_id, image_path, ground_truth, event):
        log_path = Path(self.args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.log_lock:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("=" * 50 + "\n")
                log_file.write(f"Sample ID: {sample_id}\n")
                log_file.write(f"Image: {image_path}\n")
                log_file.write(f"Ground Truth:\n{json.dumps(ground_truth, ensure_ascii=False)}\n")
                log_file.write(json.dumps(event, ensure_ascii=False, indent=2, default=str))
                log_file.write("\n\n")

    def make_image_content(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            print(f"Failed to encode image {image_path}: {exc}")
            return None
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }

    def build_user_content(self, text, image_paths):
        user_content = [{"type": "text", "text": text}]
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            image_content = self.make_image_content(image_path)
            if image_content is not None:
                user_content.append(image_content)
        return user_content

    async def chat_completion(self, messages, max_tokens, temperature, extra_body_json):
        kwargs = {
            "model": self.args.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra_body = parse_api_extra_body(extra_body_json)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return await self.client.chat.completions.create(**kwargs)

    async def request_hidden_reasoning(self, question, steps, answer, image_paths):
        text = build_full_problem_text(question, steps, answer)
        messages = [
            {"role": "system", "content": HIDDEN_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_content(text, image_paths)},
        ]
        response = await self.chat_completion(
            messages,
            max_tokens=self.args.analysis_max_tokens,
            temperature=self.args.analysis_temperature,
            extra_body_json=self.args.analysis_extra_body_json,
        )
        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        hidden_reasoning = extract_reasoning_from_message(message)
        return {
            "content": content,
            "hidden_reasoning": hidden_reasoning,
            "finish_reason": getattr(choice, "finish_reason", None),
        }

    async def compress_reasoning(self, question, steps, answer, reasoning_text):
        compressed_input = truncate_reasoning_text(
            reasoning_text,
            max_chars=self.args.reasoning_input_max_chars,
        )
        user_text = (
            f"{build_full_problem_text(question, steps, answer)}\n"
            f"[Teacher Hidden Reasoning]\n{compressed_input}\n"
        )
        messages = [
            {
                "role": "system",
                "content": GLOBAL_THINK_COMPRESS_SYSTEM_PROMPT.format(
                    max_words=self.args.global_thinking_max_words
                ),
            },
            {"role": "user", "content": user_text},
        ]
        response = await self.chat_completion(
            messages,
            max_tokens=self.args.compression_max_tokens,
            temperature=self.args.compression_temperature,
            extra_body_json=self.args.compression_extra_body_json,
        )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        return {
            "content": content,
            "finish_reason": getattr(choice, "finish_reason", None),
        }

    def build_sft_samples(self, sample, sample_id, question, steps, answer, labels, kimi_scores, global_thinking):
        source_scores = sample.get("source_scores", [])
        common = {
            "images": sample.get("images", []),
            "source_image": sample.get("source_image"),
            "source_annotation": sample.get("source_annotation"),
            "source_scores": source_scores,
            "negative_score_threshold": self.args.negative_score_threshold,
            "positive_score_threshold": self.args.positive_score_threshold,
            "kimi_scores": kimi_scores,
            "source_sample_id": sample_id,
            "num_steps": len(steps),
        }

        samples = []
        if self.args.include_global_thinking_task:
            samples.append({
                **common,
                "task_type": "global_thinking",
                "messages": [
                    {"role": "system", "content": GLOBAL_THINK_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": build_full_problem_text(question, steps, answer)},
                    {"role": "assistant", "content": f"<think>{global_thinking}</think>"},
                ],
            })

        previous_labels = []
        for step_idx, label in enumerate(labels):
            if label is None:
                previous_labels.append(int(kimi_scores[step_idx]))
                continue
            samples.append({
                **common,
                "task_type": "step_score",
                "step_index": step_idx,
                "step_label": int(label),
                "messages": [
                    {"role": "system", "content": STEP_LEVEL_SFT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_step_user_text(
                            question,
                            steps,
                            answer,
                            global_thinking,
                            previous_labels,
                            step_idx,
                        ),
                    },
                    {"role": "assistant", "content": str(int(label))},
                ],
            })
            previous_labels.append(int(label))

        return samples

    async def process_sample(self, sample, sample_id):
        user_text = sample["messages"][1]["content"]
        question, steps, answer = split_user_content(user_text)
        source_scores = load_source_scores(sample)
        labels = build_conservative_labels(
            source_scores,
            negative_threshold=self.args.negative_score_threshold,
            positive_threshold=self.args.positive_score_threshold,
        )
        if len(labels) != len(steps):
            raise ValueError(f"Score length mismatch: {len(labels)}!={len(steps)}")
        if not any(label is not None for label in labels):
            raise ValueError("no_confident_mc_steps")

        image_path = sample.get("images", [""])[0] if sample.get("images") else ""
        hidden_info = None
        compressed_info = None
        kimi_scores = None
        for attempt in range(self.args.max_retries + 1):
            try:
                hidden_info = await self.request_hidden_reasoning(
                    question,
                    steps,
                    answer,
                    sample.get("images", []),
                )
                reasoning_source = (hidden_info.get("hidden_reasoning") or "").strip()
                visible_content = (hidden_info.get("content") or "").strip()
                kimi_scores = parse_score_list(visible_content, expected_len=len(steps))
                confident_count = validate_kimi_scores(kimi_scores, labels)
                if not reasoning_source:
                    raise ValueError("missing_hidden_reasoning")

                compressed_info = await self.compress_reasoning(
                    question,
                    steps,
                    answer,
                    reasoning_source,
                )
                global_thinking = normalize_global_thinking(
                    compressed_info.get("content", ""),
                    max_chars=self.args.global_thinking_max_chars,
                )
                validate_global_thinking(
                    global_thinking,
                    min_chars=self.args.global_thinking_min_chars,
                )

                sft_samples = self.build_sft_samples(
                    sample,
                    sample_id,
                    question,
                    steps,
                    answer,
                    labels,
                    kimi_scores,
                    global_thinking,
                )
                await self.append_log(
                    sample_id,
                    image_path,
                    labels,
                    {
                        "status": "accepted",
                        "attempt": attempt + 1,
                        "num_steps": len(steps),
                        "num_confident_steps": confident_count,
                        "num_sft_samples": len(sft_samples),
                        "kimi_scores": kimi_scores,
                        "conservative_labels": labels,
                        "source_scores": source_scores,
                        "hidden_content_chars": len(hidden_info.get("content", "")),
                        "hidden_reasoning_chars": len(hidden_info.get("hidden_reasoning", "")),
                        "compressed_chars": len(compressed_info.get("content", "")),
                    },
                )
                return sft_samples, None
            except Exception as exc:
                error_info = self.format_api_error(exc, attempt=attempt + 1)
                await self.append_log(
                    sample_id,
                    image_path,
                    labels,
                    {
                        "status": "failed_attempt",
                        "attempt": attempt + 1,
                        "error": error_info,
                        "kimi_scores": kimi_scores,
                        "conservative_labels": labels,
                        "source_scores": source_scores,
                        "hidden_info": hidden_info,
                        "compressed_info": compressed_info,
                    },
                )
                if attempt < self.args.max_retries:
                    await asyncio.sleep(self.args.retry_sleep * (attempt + 1))

        failed = sample.copy()
        failed["failure_reason"] = "global_think_stepwise_rollout_failed"
        failed["kimi_scores"] = kimi_scores
        failed["conservative_labels"] = labels
        failed["source_scores"] = source_scores
        failed["hidden_info"] = hidden_info
        failed["compressed_info"] = compressed_info
        return None, failed


async def process_sample_async(rollout, sample, sample_id, semaphore):
    async with semaphore:
        try:
            return await rollout.process_sample(sample, sample_id)
        except Exception as exc:
            failed = sample.copy()
            failed["failure_reason"] = "pipeline_exception"
            failed["error"] = repr(exc)
            return None, failed


async def run_rollout(data, rollout, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(process_sample_async(rollout, sample, idx, semaphore))
        for idx, sample in enumerate(data)
    ]
    train_data = []
    failed_samples = []
    iterator = asyncio.as_completed(tasks)
    progress = tqdm(total=len(tasks), desc="Processing samples") if tqdm is not None else None
    completed = 0
    for task in iterator:
        sft_samples, failed = await task
        completed += 1
        if sft_samples is not None:
            train_data.extend(sft_samples)
        else:
            failed_samples.append(failed)
        if progress is not None:
            progress.update(1)
            progress.set_description(
                f"Processed: {len(train_data)} sft rows, {len(failed_samples)} failed"
            )
        elif completed % 20 == 0 or completed == len(tasks):
            print(f"completed {completed}/{len(tasks)} sft_rows={len(train_data)} failed={len(failed_samples)}")
    if progress is not None:
        progress.close()
    return train_data, failed_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create global-thinking + step-level SFT data from VisualPRM400K-v1.1-Raw. "
            "Kimi hidden reasoning is compressed into global thinking, then each solution "
            "step becomes an independent scoring sample."
        )
    )
    parser.add_argument("--input-path", default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--failed-output-path", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--polarity", choices=("positive", "negative", "all"), default="negative")
    parser.add_argument("--negative-mode", choices=("any_step", "final_step", "all_steps"), default="any_step")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --negative-score-threshold. Kept for old commands."
        ),
    )
    parser.add_argument(
        "--negative-score-threshold",
        type=float,
        default=0.125,
        help="Conservative negative label threshold. A step is negative when MC score <= this value.",
    )
    parser.add_argument(
        "--positive-score-threshold",
        type=float,
        default=0.75,
        help="Conservative positive label threshold. A step is positive when MC score >= this value.",
    )
    parser.add_argument("--model-name", default="kimi-k26-w4a8")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible chat endpoint. Can also be set with OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the endpoint. Can also be set with OPENAI_API_KEY.",
    )
    parser.add_argument("--analysis-max-tokens", type=int, default=2048)
    parser.add_argument("--analysis-temperature", type=float, default=0.2)
    parser.add_argument(
        "--analysis-extra-body-json",
        default="",
        help=(
            "Extra JSON object for the first Kimi call. Default is empty so hidden "
            "reasoning is not disabled. If your server requires it, pass "
            '\'{"chat_template_kwargs":{"thinking":true}}\'.'
        ),
    )
    parser.add_argument("--reasoning-input-max-chars", type=int, default=12000)
    parser.add_argument("--compression-max-tokens", type=int, default=512)
    parser.add_argument("--compression-temperature", type=float, default=0.0)
    parser.add_argument(
        "--compression-extra-body-json",
        default='{"chat_template_kwargs":{"thinking":false}}',
        help="Extra JSON object for the compression call. Default disables hidden thinking.",
    )
    parser.add_argument("--global-thinking-max-words", type=int, default=180)
    parser.add_argument("--global-thinking-max-chars", type=int, default=2000)
    parser.add_argument(
        "--global-thinking-min-chars",
        type=int,
        default=100,
        help="Reject compressed global thinking shorter than this many characters.",
    )
    parser.add_argument(
        "--include-global-thinking-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save one global-thinking generation sample per original problem.",
    )
    parser.add_argument(
        "--target-negative-step-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional positive-step downsampling target for step_score rows. "
            "Example: 0.4 keeps at most 60/40 positive/negative among step rows. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--samples-per-annotation", type=int, default=None)
    parser.add_argument("--sample-total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.score_threshold is not None:
        args.negative_score_threshold = args.score_threshold
    if args.negative_score_threshold >= args.positive_score_threshold:
        raise ValueError("--negative-score-threshold must be smaller than --positive-score-threshold.")

    if args.output_path is None:
        args.output_path = str(
            DEFAULT_OUTPUT_DIR / f"visualprm400k_{args.polarity}_global_think_stepwise_sft_success.json"
        )
    if args.failed_output_path is None:
        args.failed_output_path = str(
            DEFAULT_OUTPUT_DIR / f"visualprm400k_{args.polarity}_global_think_stepwise_sft_failed.json"
        )
    if args.log_path is None:
        args.log_path = str(
            DEFAULT_OUTPUT_DIR / f"visualprm400k_{args.polarity}_global_think_stepwise_processing_log.txt"
        )

    if args.overwrite:
        for path in (args.output_path, args.failed_output_path, args.log_path):
            path = Path(path)
            if path.exists():
                path.unlink()

    data = load_selected_samples(
        args.input_path,
        args.image_root,
        sample_total=args.sample_total,
        samples_per_annotation=args.samples_per_annotation,
        seed=args.seed,
        negative_mode=args.negative_mode,
        negative_score_threshold=args.negative_score_threshold,
        positive_score_threshold=args.positive_score_threshold,
        polarity=args.polarity,
    )
    if args.limit is not None:
        data = data[: args.limit]

    if args.dry_run:
        step_labels = []
        ignored_steps = 0
        for sample in data:
            try:
                labels = build_conservative_labels(
                    load_source_scores(sample),
                    negative_threshold=args.negative_score_threshold,
                    positive_threshold=args.positive_score_threshold,
                )
                step_labels.extend([label for label in labels if label is not None])
                ignored_steps += sum(1 for label in labels if label is None)
            except ValueError:
                pass
        neg = sum(1 for label in step_labels if label == 0)
        pos = sum(1 for label in step_labels if label == 1)
        print(f"Dry run selected source samples: {len(data)}")
        print(
            f"Dry run confident step labels: positive={pos}, negative={neg}, "
            f"confident_total={len(step_labels)}, ignored_middle={ignored_steps}"
        )
        return

    if not data:
        print("No samples to process.")
        save_json([], args.output_path)
        save_filed_sample_json([], args.failed_output_path)
        return

    start_time = time.time()
    rollout = AsyncGlobalThinkStepwiseRolloutor(args)
    train_data, failed_samples = asyncio.run(
        run_rollout(data, rollout, concurrency=max(1, args.num_workers))
    )
    before_balance = len(train_data)
    train_data = apply_step_balance(train_data, args.target_negative_step_ratio, args.seed)

    print("\nProcessing completed!")
    print(f"Source samples processed: {len(data)}")
    print(f"Valid SFT rows before balancing: {before_balance}")
    print(f"Valid SFT rows after balancing: {len(train_data)}")
    print(f"Failed source samples: {len(failed_samples)}")
    print(f"Source success rate: {(len(data) - len(failed_samples)) / len(data) * 100:.2f}%")
    print(f"Total run time: {time.time() - start_time:.2f} seconds")
    save_json(train_data, args.output_path)
    save_filed_sample_json(failed_samples, args.failed_output_path)


if __name__ == "__main__":
    main()
