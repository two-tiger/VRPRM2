import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BENCH_DIR = Path(__file__).resolve().parent
VRPRM_ROOT = BENCH_DIR.parents[1]
REPO_ROOT = BENCH_DIR.parents[2]
VLMEVAL_ROOT = REPO_ROOT / "third_party" / "VLMEvalKit"
LOCAL_LMU_DATA = BENCH_DIR / "datasets"

if str(VLMEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMEVAL_ROOT))

os.environ.setdefault("LMUData", str(LOCAL_LMU_DATA))


DATASET_ALIASES = {
    "MathVista": "MathVista_MINI",
    "MathVista_MINI": "MathVista_MINI",
    "MathVision": "MathVision",
    "MathVerse": "MathVerse_MINI_Vision_Only",
    "MathVerse-VO": "MathVerse_MINI_Vision_Only",
    "MathVerse_MINI_Vision_Only": "MathVerse_MINI_Vision_Only",
    "WeMath": "WeMath",
    "LogicVista": "LogicVista",
    "DynaMath": "DynaMath",
    "MMMU": "MMMU_DEV_VAL",
    "MMMU_DEV_VAL": "MMMU_DEV_VAL",
}

TABLE2_DATASETS = [
    "MMMU",
    "MathVista",
    "MathVision",
    "MathVerse-VO",
    "DynaMath",
    "WeMath",
    "LogicVista",
]


SFT_PRM_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a step-by-step candidate solution, generate SFT-quality supervision for a visual reasoning process reward model.

For each solution step, output 1 only when the step is fully supported by the image/problem context and remains logically and mathematically correct. Output 0 when the step is unsupported, contains a visual misunderstanding, uses invalid logic, has a calculation error, contradicts earlier valid reasoning, or depends on a previous incorrect step.

Output requirements:
- Start with [Preliminary thinking] and include exactly one <think>...</think> block.
- Then provide [Step Analysis] with one line per received step. Each line must end with \\boxed{0} or \\boxed{1}.
- Then provide [Final Scores] as strict JSON: {"Score": [comma-separated integer scores]}.
- Then provide [Final Judgment] as strict JSON: {"Judge": 0 or 1}.
- The Score array length must exactly equal the number of received steps.
- Do not add any text after the final JSON object."""


SCORE_RE = re.compile(r'\{\s*"Score"\s*:\s*\[([^\]]*)\]\s*\}')
EXCEL_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def parse_dataset_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return TABLE2_DATASETS.copy()
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_excel_value(value: Any) -> Any:
    if isinstance(value, str):
        return EXCEL_ILLEGAL_CHAR_RE.sub("", value)
    return value


def sanitize_dataframe_for_excel(data: Any) -> Any:
    sanitized = data.copy()
    for col in sanitized.columns:
        dtype = str(sanitized[col].dtype)
        if dtype == "object" or dtype.startswith("string"):
            sanitized[col] = sanitized[col].map(sanitize_excel_value)
    return sanitized


def encode_image(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def prompt_items_to_openai_content(items: Iterable[Dict[str, Any]]) -> Tuple[str, List[str]]:
    text_parts: List[str] = []
    image_paths: List[str] = []
    for item in items:
        item_type = item.get("type")
        value = item.get("value")
        if item_type == "text":
            text_parts.append(str(value))
        elif item_type == "image":
            image_paths.append(str(value))
    return "\n".join(text_parts).strip(), image_paths


def make_policy_prompt(text: str, dataset_type: str) -> str:
    text = re.sub(r"\n?Answer with the option's letter from the given choices directly\.?", "", text).strip()
    text = re.sub(r"\n?Answer the question directly\.?", "", text).strip()
    if dataset_type in {"MCQ", "MCQ_MMMU_Pro"}:
        suffix = (
            "Solve the problem step by step. The last line of your response must be "
            "'Answer: \\boxed{$LETTER}', where LETTER is one of the option letters."
        )
    else:
        suffix = (
            "Solve the problem step by step. The last line of your response must be "
            "'Answer: \\boxed{$FINAL_ANSWER}', where FINAL_ANSWER is your final answer."
        )
    return f"{text}\n\n{suffix}"


def openai_content(text: str, image_paths: Iterable[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_image(path)}})
    return content


def split_steps(response: str) -> List[str]:
    text = (response or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    if "<step split>" in text:
        parts = text.split("<step split>")
    else:
        parts = re.split(r"\n\s*\n+", text)
        if len(parts) <= 1:
            parts = re.split(r"\n(?=(?:Step\s*\d+|步骤\s*\d+|[-*]\s+))", text)
        if len(parts) <= 1:
            parts = [line for line in text.split("\n") if line.strip()]
    return [part.strip() for part in parts if part.strip()]


def parse_score_array(text: str, expected_len: int) -> Optional[List[int]]:
    match = SCORE_RE.search(text or "")
    if not match:
        return None
    values: List[int] = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(1 if float(raw) > 0 else 0)
        except ValueError:
            return None
    if expected_len and len(values) != expected_len:
        return None
    return values


def response_score(scores: Optional[List[int]]) -> float:
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def default_judge_for(dataset_name: str, dataset_type: str) -> str:
    if dataset_type in {"MCQ", "Y/N", "MCQ_MMMU_Pro"}:
        return "gpt-4o-mini" if "WeMath" in dataset_name else "chatgpt-0125"
    if any(key in dataset_name for key in ["MathVista", "MathVerse", "MathVision", "LogicVista"]):
        return "gpt-4o-mini"
    return "gpt-4o-mini"


def env_default(name: str, default: Any) -> Any:
    return os.environ.get(name, default)


def normalize_openai_api_base(api_base: Optional[str]) -> Optional[str]:
    if api_base is None:
        return None
    api_base = str(api_base).strip()
    if not api_base:
        return api_base
    stripped = api_base.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return api_base


def apply_eval_api_env() -> None:
    eval_api_key = os.environ.get("EVAL_API_KEY")
    if eval_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = eval_api_key

    api_base = (
        os.environ.get("EVAL_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )
    normalized = normalize_openai_api_base(api_base)
    if normalized:
        os.environ["OPENAI_API_BASE"] = normalized


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    value = str(value)
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "***" + value[-4:]


def response_preview(resp: Any, limit: int = 1000) -> str:
    if resp is None:
        return ""
    text = getattr(resp, "text", None)
    if text is None:
        text = str(resp)
    text = str(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def probe_eval_judge_api(judge_kwargs: Dict[str, Any], context: str = "") -> None:
    """Print a minimal VLMEvalKit judge API probe for debugging failed evaluations."""
    apply_eval_api_env()
    probe_kwargs = dict(judge_kwargs)
    probe_kwargs.pop("nproc", None)
    probe_kwargs["verbose"] = True
    probe_kwargs["max_tokens"] = min(int(probe_kwargs.get("max_tokens", 16)), 16)

    print("==== VLMEvalKit judge API debug ====", flush=True)
    if context:
        print(f"context: {context}", flush=True)
    print(f"requested judge: {judge_kwargs.get('model')}", flush=True)
    print(f"OPENAI_API_BASE: {os.environ.get('OPENAI_API_BASE', '')}", flush=True)
    print(f"OPENAI_API_KEY: {redact_secret(os.environ.get('OPENAI_API_KEY', ''))}", flush=True)
    try:
        from vlmeval.dataset.utils.judge_util import build_judge

        model = build_judge(**probe_kwargs)
        print(f"resolved api_base: {getattr(model, 'api_base', '')}", flush=True)
        print(f"resolved model: {getattr(model, 'model', '')}", flush=True)
        code, answer, resp = model.generate_inner([dict(type="text", value="hello")])
        print(f"probe ret_code: {code}", flush=True)
        print(f"probe answer: {answer}", flush=True)
        print(f"probe response status: {getattr(resp, 'status_code', '')}", flush=True)
        print(f"probe response text: {response_preview(resp)}", flush=True)
    except Exception as err:
        print(f"probe exception: {type(err).__name__}: {err}", flush=True)
    print("==== end VLMEvalKit judge API debug ====", flush=True)
