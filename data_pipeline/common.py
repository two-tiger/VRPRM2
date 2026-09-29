"""Shared utilities for the VRPRM SFT data pipelines.

Extracted from the original rollout_sft_data_pipeline.py (v1). Only the helpers
used by the paper pipeline (teacher rollout + multiturn dataset builder) are
kept; the v1 full-response rollout lives on archive/snapshot-202609.
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, List


# data_pipeline/ sits one level below the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RAW_DATA_DIR = DATA_ROOT / "VisualPRM400K-v1.1-Raw"
DEFAULT_ANNOTATION_PATH = DEFAULT_RAW_DATA_DIR / "annotations"
DEFAULT_IMAGE_ROOT = DEFAULT_RAW_DATA_DIR / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rollout_outputs"


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
def parse_api_extra_body(extra_body_json):
    extra_body_json = (extra_body_json or "").strip()
    if not extra_body_json or extra_body_json.lower() in {"none", "null", "{}"}:
        return None
    extra_body = json.loads(extra_body_json)
    if not isinstance(extra_body, dict):
        raise ValueError("--api-extra-body-json must decode to a JSON object.")
    return extra_body
