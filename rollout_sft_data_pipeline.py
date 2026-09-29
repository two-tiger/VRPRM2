import os
import re
import json
import base64
import time
import argparse
import mimetypes
import traceback
import random
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RAW_DATA_DIR = DATA_ROOT / "VisualPRM400K-v1.1-Raw"
DEFAULT_ANNOTATION_PATH = DEFAULT_RAW_DATA_DIR / "annotations"
DEFAULT_IMAGE_ROOT = DEFAULT_RAW_DATA_DIR / "images"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "rollout_outputs"


SFT_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, generate SFT-quality supervision for a visual reasoning process reward model.

For each solution step, output 1 only when the step is fully supported by the image/problem context and remains logically and mathematically correct. Output 0 when the step is unsupported, contains a visual misunderstanding, uses invalid logic, has a calculation error, contradicts earlier valid reasoning, or depends on a previous incorrect step.

Evaluation policy:
- Use the image(s) as primary evidence whenever the question depends on visual content.
- Before scoring, identify the relevant visual facts, the mathematical/logical goal, and how the reference answer constrains the solution.
- Evaluate each step in sequence. A step receives 1 only if it is correct in context and all required premises are valid.
- A step receives 0 if it introduces an error, misses necessary visual evidence, hallucinates image content, makes an invalid inference, performs an incorrect computation, or relies on a previous incorrect step.
- After the first consequential wrong step, later steps should receive 0 unless they explicitly recover with correct independent reasoning.
- Do not give partial credit. Use only integer scores 0 or 1.
- The overall Judge is 1 only when the complete solution reaches the reference answer through valid reasoning; otherwise it is 0.

Output requirements:
- Start the visible answer with exactly one <think>...</think> block. In this block, write the model's reasoning about the problem: summarize the key visual evidence, the question goal, the reference-answer constraint, and the main correctness issue in the candidate steps. Keep it focused.
- Provide [Step Analysis] with one line per received step. Each line must be no more than 50 words and end with \\boxed{0} or \\boxed{1}.
- Provide [Final Scores] as strict JSON: {"Score": [comma-separated integer scores]}.
- Provide [Final Judgment] as strict JSON: {"Judge": 0 or 1}.
- The Score array length must exactly equal the number of received steps.
- Do not add any text after the final JSON object.

Output format:
<think>Reason about the problem, image evidence, reference answer, and the candidate solution. Identify the decisive facts and the first major step error if any.</think>
[Step Analysis]
Step 0: Brief evidence-grounded assessment. \\boxed{1}
Step 1: Brief evidence-grounded assessment. \\boxed{0}
[Final Scores]
{"Score": [1, 0]}
[Final Judgment]
{"Judge": 0}"""

def build_first_content(question, answer):
    return f"""{SFT_SYSTEM_PROMPT}

Task:
You will receive the image(s), the question, the reference answer, and then the candidate solution steps one by one. Generate SFT-quality supervision for a visual reasoning process reward model.

Question:
{question}

Reference Answer:
{answer}

The candidate solution steps will be sent next, one message per step.
"""

def load_json(file_path):
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        return load_jsonl(file_path)
    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path):
    data = []
    with Path(file_path).open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skip invalid JSONL line {file_path}:{line_no}: {e}")
    return data


def iter_input_files(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl")) + sorted(input_path.glob("*.json"))
    return [input_path]


def load_samples(input_path):
    data = []
    for file_path in iter_input_files(input_path):
        data.extend(load_json(file_path))
    return data


def sample_records(records, sample_size, rng):
    if sample_size is None or sample_size <= 0 or len(records) <= sample_size:
        return records
    return rng.sample(records, sample_size)


def allocate_even_quotas(counts, total):
    if total is None or total <= 0:
        return counts[:]

    total = min(total, sum(counts))
    quotas = [0] * len(counts)
    active = {idx for idx, count in enumerate(counts) if count > 0}
    remaining = total

    while active and remaining > 0:
        base, extra = divmod(remaining, len(active))
        if base == 0:
            for idx in sorted(active)[:extra]:
                quotas[idx] += 1
            break

        next_active = set()
        distributed = 0
        for idx in sorted(active):
            capacity = counts[idx] - quotas[idx]
            give = min(capacity, base)
            quotas[idx] += give
            distributed += give
            if quotas[idx] < counts[idx]:
                next_active.add(idx)

        remaining -= distributed
        active = next_active

    return quotas

def save_json(new_data, output_path):
    """追加数据到JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 尝试读取现有数据
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或为空，初始化新列表
        existing_data = []
    
    # 合并数据
    if isinstance(existing_data, list):
        existing_data.extend(new_data)
    else:
        existing_data = [existing_data] + new_data
    
    # 写入更新后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
def save_filed_sample_json(new_data: List[Dict], file_path: str):
    """追加数据到JSON文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 尝试读取现有数据
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或为空，初始化新列表
        existing_data = []
    
    # 合并数据
    if isinstance(existing_data, list):
        existing_data.extend(new_data)
    else:
        existing_data = [existing_data] + new_data
    
    # 写入更新后的数据
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


def resolve_visualprm_image_path(image_path, image_root):
    """Map VisualPRM400K-v1.1-Raw annotation paths to the local images directory."""
    if not image_path:
        return None

    image_root = Path(image_root)
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return str(path)

    raw_path = str(image_path).replace("\\", "/")
    marker = "VisualPRM400K-v1.1-Raw/"
    if marker in raw_path:
        rel_path = raw_path.split(marker, 1)[1]
        if rel_path.startswith("images/"):
            rel_path = rel_path[len("images/"):]
        return str(image_root / rel_path)

    if raw_path.startswith("images/"):
        rel_path = raw_path[len("images/"):]
        candidate = image_root / rel_path
        if candidate.exists():
            return str(candidate)
        nlvr2_candidate = image_root / "nlvr2" / "images" / rel_path
        if nlvr2_candidate.exists():
            return str(nlvr2_candidate)
        return str(candidate)

    candidate = DATA_ROOT / raw_path
    if candidate.exists():
        return str(candidate)

    return str(image_root / raw_path)


def get_image_paths(sample, image_root):
    images = sample.get("images", sample.get("image", []))
    if isinstance(images, str):
        images = [images]
    return [
        resolved
        for image in images
        if (resolved := resolve_visualprm_image_path(image, image_root)) is not None
    ]


def normalize_question(question):
    question = str(question or "").strip()
    if question.startswith("[Question]"):
        question = question.replace("[Question]", "", 1).strip()
    return question


def normalize_steps(steps_with_score):
    steps = []
    scores = []
    for item in steps_with_score or []:
        if isinstance(item, dict):
            step = str(item.get("step", "")).strip()
            score = item.get("score", 0)
        else:
            step = str(item).strip()
            score = 0
        if not step:
            continue
        steps.append(step)
        scores.append(float(score))
    return steps, scores


def build_user_content(question, steps, answer):
    numbered_steps = [f"Step {idx}: {step}" for idx, step in enumerate(steps)]
    return (
        f"[Question]\n{question}\n"
        f"[Solution]\n{'<step split>'.join(numbered_steps)}\n"
        f"[Answer]\n{answer}\n"
    )


def normalize_visualprm400k_raw_sample(sample, image_root, source_annotation=None):
    """Convert a VisualPRM400K-v1.1-Raw row into the rollout format used here."""
    question = normalize_question(sample.get("question", sample.get("question_orig", "")))
    answer = str(sample.get("answer", "")).strip()
    steps, scores = normalize_steps(sample.get("steps_with_score", []))
    images = get_image_paths(sample, image_root)

    if not question or not answer or not steps:
        return None

    user = {"role": "user", "content": build_user_content(question, steps, answer)}
    assistant = {"role": "assistant", "content": json.dumps({"Score": scores}, ensure_ascii=False)}
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            user,
            assistant,
        ],
        "images": images,
        "source_image": sample.get("image"),
        "source_annotation": source_annotation,
        "num_mc_sequences": sample.get("num_mc_sequences"),
    }


def normalize_sample(sample, image_root, source_annotation=None):
    if "messages" in sample and "images" in sample:
        item = sample.copy()
        item.setdefault("source_annotation", source_annotation)
        return item
    return normalize_visualprm400k_raw_sample(sample, image_root, source_annotation=source_annotation)


def normalize_samples(samples, image_root, source_annotation=None):
    normalized = []
    skipped = 0
    for sample in samples:
        item = normalize_sample(sample, image_root, source_annotation=source_annotation)
        if item is None:
            skipped += 1
            continue
        normalized.append(item)
    if skipped:
        source = f" in {source_annotation}" if source_annotation else ""
        print(f"Skipped {skipped} samples{source} with missing question/answer/steps.")
    return normalized


def load_and_normalize_samples(
    input_path,
    image_root,
    sample_total=5000,
    samples_per_annotation=None,
    seed=42,
):
    data = []
    groups = []
    for file_path in iter_input_files(input_path):
        raw_samples = load_json(file_path)
        normalized = normalize_samples(
            raw_samples,
            image_root,
            source_annotation=str(file_path),
        )
        groups.append((file_path, raw_samples, normalized))

    if samples_per_annotation is not None and samples_per_annotation > 0:
        quotas = [min(len(normalized), samples_per_annotation) for _, _, normalized in groups]
    else:
        quotas = allocate_even_quotas([len(normalized) for _, _, normalized in groups], sample_total)

    for (file_path, raw_samples, normalized), quota in zip(groups, quotas):
        rng = random.Random(f"{seed}:{file_path}")
        selected = sample_records(normalized, quota, rng)
        print(
            f"Loaded {file_path}: raw={len(raw_samples)}, "
            f"normalized={len(normalized)}, quota={quota}, selected={len(selected)}"
        )
        data.extend(selected)
    random.Random(seed).shuffle(data)
    print(f"Total selected samples: {len(data)}")
    return data
        
def extract_score_from_response(response):
    try:
        score_match = re.search(r'\{\s*"Score"\s*:\s*\[([0-9\.\s,\-]+)\]\s*\}', response)
        if not score_match:
            raise ValueError("No valid JSON block with 'score' found.")
        
        else:
            score_list_str = score_match.group(1)
            score_list = [float(x.strip()) for x in score_list_str.split(',') if x.strip()]
            return score_list
    except Exception as e:
        return None

def extract_judge_from_response(response):
    try:
        judge_match = re.search(r'\{\s*"Judge"\s*:\s*([01])\s*\}', response)
        if not judge_match:
            raise ValueError("No valid JSON block with 'Judge' found.")
        return int(judge_match.group(1))
    except Exception as e:
        return None

def count_think_blocks(response):
    return len(re.findall(r"<think>[\s\S]*?</think>", response or ""))


def extract_think_text(response):
    match = re.search(r"<think>([\s\S]*?)</think>", response or "")
    if not match:
        return ""
    return match.group(1).strip()


def extract_reasoning_from_message(message):
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if value:
            return str(value).strip()

    model_extra = getattr(message, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if value:
                return str(value).strip()
    return ""


def strip_thinking_prefix(response):
    response = (response or "").strip()
    response = re.sub(r"\[Preliminary thinking\]\s*", "", response).strip()
    response = re.sub(r"<think>[\s\S]*?</think>\s*", "", response).strip()
    return response


def trim_after_judge(response):
    match = re.search(r'\{\s*"Judge"\s*:\s*[01]\s*\}', response or "")
    if not match:
        return response
    return response[:match.end()].strip()


def truncate_reasoning_text(text, max_chars=4000):
    text = (text or "").strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    for separator in ("\n\n", "\n", ". ", "; "):
        idx = cut.rfind(separator)
        if idx >= int(max_chars * 0.6):
            return cut[:idx].rstrip()
    return cut.rstrip()


def format_response_for_sft(response, reasoning=None, assistant_format="preliminary", max_reasoning_chars=4000):
    """Attach visible/API reasoning to the target SFT format."""
    response = normalize_response_format(response)
    fallback_reasoning = extract_think_text(response)
    response = strip_thinking_prefix(response)
    if "[Step Analysis]" in response:
        response = response[response.find("[Step Analysis]"):].strip()
    response = trim_after_judge(response)

    reasoning = (fallback_reasoning or reasoning or "").strip()
    reasoning = re.sub(r"</?think>", "", reasoning).strip()
    if not reasoning:
        reasoning = "The visual evidence, problem constraints, reference answer, and candidate steps are checked before assigning step scores."
    reasoning = truncate_reasoning_text(reasoning, max_chars=max_reasoning_chars)

    if assistant_format == "thinking":
        return f"<think>{reasoning}</think>\n{response}".strip()
    if assistant_format == "preliminary":
        return f"[Preliminary thinking]\n<think>{reasoning}</think>\n{response}".strip()
    raise ValueError(f"Unknown assistant format: {assistant_format}")


def judge_is_think(response):
    try:
        return count_think_blocks(response) == 1
    except Exception as e:
        print(f"Failed to validate think process from response: {e}")
        return False


def normalize_response_format(response):
    """Repair common format drift without changing model scores or judgments."""
    if not response:
        return response

    response = response.strip()
    if response.startswith("[Step Analysis]") and "[Final Scores]" in response:
        cutoff_match = re.search(r'\{\s*"Judge"\s*:\s*[01]\s*\}', response)
        if cutoff_match:
            response = response[:cutoff_match.end()]
        return (
            "[Preliminary thinking]\n"
            "<think>The visual evidence, problem goal, and candidate steps are assessed below.</think>\n"
            f"{response}"
        ).strip()

    if "[Preliminary thinking]" in response and not response.startswith("[Preliminary thinking]"):
        response = response[response.rfind("[Preliminary thinking]"):].strip()

    if "[Preliminary thinking]" not in response or "[Step Analysis]" not in response:
        return response

    before_steps, after_steps = response.split("[Step Analysis]", 1)
    preliminary = before_steps.replace("[Preliminary thinking]", "", 1).strip()
    preliminary = re.sub(r"</?think>", "", preliminary).strip()
    if not preliminary:
        return response

    return (
        "[Preliminary thinking]\n"
        f"<think>{preliminary}</think>\n"
        "[Step Analysis]"
        f"{after_steps}"
    ).strip()


def validate_response(response, ground_truth):
    pred_score_list = extract_score_from_response(response=response)
    if not pred_score_list:
        return False, "missing_or_empty_score"

    if not judge_is_think(response=response):
        return False, f"bad_think_count:{count_think_blocks(response)}"

    if extract_judge_from_response(response=response) is None:
        return False, "missing_judge"
    
    # If lengths don't match, return False directly
    if len(pred_score_list) != len(ground_truth):
        return False, f"score_length_mismatch:{len(pred_score_list)}!={len(ground_truth)}"
    
    # Compare elements of both lists
    for idx, (pred_score, true_score) in enumerate(zip(pred_score_list, ground_truth)):
        # If one is greater than 0 and the other is less than or equal to 0, return False
        if (pred_score > 0) != (true_score > 0):
            return False, f"score_sign_mismatch:step_{idx}"
    
    # If all elements meet the conditions, return True
    return True, "accepted"


def judge_response(response, ground_truth):
    judge, _ = validate_response(response, ground_truth)
    return judge
    
@dataclass
class RolloutConfig:
    model_name: str = "kimi-k26-w4a8"
    max_tokens: int = 4096
    temperature: float = 0.2
    base_url: str = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    log_path: str = str(DEFAULT_OUTPUT_DIR / "processing_log.txt")
    max_retries: int = 2
    retry_sleep: float = 2.0
    assistant_format: str = "preliminary"
    api_extra_body_json: str = '{"chat_template_kwargs":{"thinking":false}}'
    

def parse_api_extra_body(extra_body_json):
    extra_body_json = (extra_body_json or "").strip()
    if not extra_body_json or extra_body_json.lower() in {"none", "null", "{}"}:
        return None
    extra_body = json.loads(extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--api-extra-body-json must decode to a JSON object.")
    return extra_body


class Rolloutor:
    def __init__(self, config: RolloutConfig):
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install the openai package before running rollout generation.") from e
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )

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

    def append_log(self, image_path, ground_truth, prediction_text=None, error=None, request_info=None):
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("="*50 + '\n')
            log_file.write(f"Image: {image_path}\n")
            if request_info is not None:
                log_file.write(f"Request Info:\n{json.dumps(request_info, ensure_ascii=False, indent=2)}\n")
            log_file.write(f"Ground Truth:\n{json.dumps(ground_truth)}\n")
            if error is not None:
                log_file.write(f"Error:\n{json.dumps(error, ensure_ascii=False, indent=2, default=str)}\n\n")
            else:
                log_file.write(f"Prediction:\n{prediction_text}\n\n")
        
    def encode_image(self, img_path):
        try:
            with open(img_path, "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode("utf-8")
                return encoded
        except Exception as e:
            print(f"Failed to encode image {img_path}: {e}")
            return None

    def make_image_content(self, image_path):
        encoded = self.encode_image(image_path)
        if encoded is None:
            return None
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
        }
        
    def process_sample(self, sample: Dict) -> Tuple[str, List[float]]:
        """Process a single sample and return prediction and ground truth"""
        
        _, user, assistant = sample["messages"]
        question, step_answer = user["content"].split("\n[Solution]\n", 1)
        question = question.replace("[Question]\n", "")
        step_join, answer_str = step_answer.split("\n[Answer]\n", 1)
        answer = answer_str.strip()
        steps = [step.strip() for step in step_join.split("<step split>") if step.strip()]
        gt_str = assistant["content"].replace("'", '"')
        ground_truth = json.loads(gt_str)['Score']
        
        # Build messages
        messages = []
        for step_i in range(len(steps)+1):
            if step_i == 0:
                first_content = build_first_content(question=question, answer=answer)
                images = sample["images"]
                user_content = [{"type": "text", "text": first_content}]
                for image_path in images:
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue
                    image_content = self.make_image_content(image_path)
                    if image_content is not None:
                        user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
            else:
                step = steps[step_i - 1]
                if re.match(r"^Step\s+\d+\s*:", step):
                    step_content = step
                else:
                    step_content = f"Step {step_i-1}: {step}"
                messages.append({"role": "user", "content": step_content})
        
        image_path = sample.get('images', [''])[0] if sample.get('images') else ''
        last_error = None
        request_info = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "num_messages": len(messages),
            "num_images": len(sample.get("images", [])),
            "num_steps": len(steps),
            "api_extra_body": parse_api_extra_body(self.config.api_extra_body_json),
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.get_response_message(message=messages)
                choice = response.choices[0]
                message = choice.message
                prediction_text = (message.content or "").strip()
                reasoning_text = extract_reasoning_from_message(message)
                request_attempt_info = {
                    **request_info,
                    "attempt": attempt + 1,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_chars": len(prediction_text),
                    "reasoning_chars": len(reasoning_text),
                }
                if not prediction_text and reasoning_text:
                    self.append_log(
                        image_path,
                        ground_truth,
                        error={
                            "attempt": attempt + 1,
                            "error_type": "EmptyVisibleContent",
                            "message": "API returned reasoning but empty visible content.",
                            "reasoning_chars": len(reasoning_text),
                        },
                        request_info=request_attempt_info,
                    )
                    if attempt < self.config.max_retries:
                        time.sleep(self.config.retry_sleep * (attempt + 1))
                        continue
                    return prediction_text, reasoning_text, ground_truth

                self.append_log(
                    image_path,
                    ground_truth,
                    prediction_text=prediction_text,
                    request_info=request_attempt_info,
                )
                return prediction_text, reasoning_text, ground_truth
            except Exception as e:
                last_error = e
                error_info = self.format_api_error(e, attempt=attempt + 1)
                self.append_log(
                    image_path,
                    ground_truth,
                    error=error_info,
                    request_info={**request_info, "attempt": attempt + 1},
                )
                print(f"Failed to generate prediction text on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep * (attempt + 1))

        return '{"Score": []}', "", ground_truth
        
    def get_response_message(self, message):
        kwargs = {
            "model": self.config.model_name,
            "messages": message,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        extra_body = parse_api_extra_body(self.config.api_extra_body_json)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self.client.chat.completions.create(**kwargs)


def process_sample_wrapper(rollout: Rolloutor, sample: Dict) -> Tuple[Dict, int]:
    """Wrapper function for processing a single sample with error handling"""
    try:
        prediction_text, reasoning_text, ground_truth = rollout.process_sample(sample)
        normalized_prediction_text = format_response_for_sft(
            prediction_text,
            reasoning=reasoning_text,
            assistant_format=rollout.config.assistant_format,
        )
        if not prediction_text.strip():
            failed_sample = sample.copy()
            failed_sample["response"] = prediction_text
            failed_sample["normalized_response"] = normalized_prediction_text
            failed_sample["failure_reason"] = (
                "empty_content_with_reasoning" if reasoning_text.strip() else "empty_content"
            )
            rollout.append_log(
                sample.get("images", [""])[0] if sample.get("images") else "",
                ground_truth,
                error={
                    "error_type": "ValidationFailed",
                    "failure_reason": failed_sample["failure_reason"],
                    "response_chars": len(prediction_text),
                    "normalized_response_chars": len(normalized_prediction_text),
                    "reasoning_chars": len(reasoning_text),
                },
                request_info={"stage": "validate_response"},
            )
            return failed_sample, 0

        judge, failure_reason = validate_response(
            response=normalized_prediction_text,
            ground_truth=ground_truth,
        )
        if judge:
            new_sample_info = {}
            _, user, _ = sample["messages"]
            images = sample["images"]
            messages = []
            messages.append({"role": "system", "content": SFT_SYSTEM_PROMPT})
            messages.append(user)
            messages.append({"role": "assistant", "content": normalized_prediction_text})
            new_sample_info["messages"] = messages
            new_sample_info["images"] = images
            return new_sample_info, 1
        else:
            new_sample = sample.copy()
            new_sample["response"] = prediction_text
            if normalized_prediction_text != prediction_text:
                new_sample["normalized_response"] = normalized_prediction_text
            new_sample["failure_reason"] = failure_reason
            rollout.append_log(
                sample.get("images", [""])[0] if sample.get("images") else "",
                ground_truth,
                error={
                    "error_type": "ValidationFailed",
                    "failure_reason": failure_reason,
                    "response_chars": len(prediction_text),
                    "normalized_response_chars": len(normalized_prediction_text),
                    "reasoning_chars": len(reasoning_text),
                },
                request_info={"stage": "validate_response"},
            )
            return new_sample, 0
    except Exception as e:
        print(f"Error processing sample: {e}")
    failed_sample = sample.copy()
    failed_sample["response"] = ""
    failed_sample["failure_reason"] = "pipeline_exception"
    return failed_sample, 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out SFT supervision for VisualPRM400K-v1.1-Raw process reward data."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_ANNOTATION_PATH),
        help="A VisualPRM400K-v1.1-Raw annotation JSON/JSONL file or an annotations directory.",
    )
    parser.add_argument(
        "--image-root",
        default=str(DEFAULT_IMAGE_ROOT),
        help="Local VisualPRM400K-v1.1-Raw images directory.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_DIR / "visualprm400k_rollout_sft_success.json"),
        help="Path for accepted SFT samples.",
    )
    parser.add_argument(
        "--failed-output-path",
        default=str(DEFAULT_OUTPUT_DIR / "visualprm400k_rollout_sft_failed.json"),
        help="Path for failed or rejected samples.",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_OUTPUT_DIR / "visualprm400k_rollout_processing_log.txt"),
        help="Path for per-sample rollout logs.",
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--api-extra-body-json",
        default='{"chat_template_kwargs":{"enable_thinking":false}}',
        help=(
            "Extra JSON object sent to chat.completions.create as extra_body. "
            "Default tries to disable hidden thinking for local OpenAI-compatible servers. "
            "Use '' or '{}' to disable this extra body."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=min(32, (os.cpu_count() or 1) * 4))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--assistant-format",
        choices=("preliminary", "thinking"),
        default="preliminary",
        help="Saved assistant format. Use thinking for Qwen3-VL thinking models.",
    )
    parser.add_argument(
        "--samples-per-annotation",
        type=int,
        default=None,
        help="Optional override: uniformly sample up to this many rows per annotation file.",
    )
    parser.add_argument(
        "--sample-total",
        type=int,
        default=5000,
        help="Total uniformly sampled rows across annotation files. Use 0 to disable total sampling.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to process.")
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_and_normalize_samples(
        args.input_path,
        args.image_root,
        sample_total=args.sample_total,
        samples_per_annotation=args.samples_per_annotation,
        seed=args.seed,
    )
    if args.limit is not None:
        data = data[:args.limit]
    if not data:
        print("No samples to process.")
        save_json([], args.output_path)
        save_filed_sample_json([], args.failed_output_path)
        return

    config = RolloutConfig(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=args.api_key,
        log_path=args.log_path,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        assistant_format=args.assistant_format,
        api_extra_body_json=args.api_extra_body_json,
    )
    rollout = Rolloutor(config=config)
    new_train_data = []
    failed_samples = []
    num_workers = max(1, args.num_workers)
    
    start_time = time.time()
    
    with tqdm(total=len(data), desc="Processing samples") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create a list of futures
            futures = [executor.submit(process_sample_wrapper, rollout, sample) for sample in data]
            
            # Process completed futures as they come in
            for future in concurrent.futures.as_completed(futures):
                result, judge = future.result()
                if judge:
                    new_train_data.append(result)
                else:
                    failed_samples.append(result)
                pbar.update(1)
                pbar.set_description(f"Processed: {len(new_train_data)} success, {len(failed_samples)} failed")
    
    end_time = time.time()
    print(f"\nProcessing completed!")
    print(f"Total samples processed: {len(data)}")
    print(f"Valid samples obtained: {len(new_train_data)}")
    if data:
        print(f"Success rate: {len(new_train_data)/len(data)*100:.2f}%")
    print(f"Total run time: {end_time-start_time:.2f} seconds")
    save_json(new_train_data, args.output_path)
    save_filed_sample_json(failed_samples, args.failed_output_path)
        

if __name__ == "__main__":
    main()
