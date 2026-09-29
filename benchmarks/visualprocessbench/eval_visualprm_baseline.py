import argparse
import json
import time
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from metrics import compute_metrics, write_json


DEFAULT_BENCH_DIR = Path(__file__).resolve().parent / "VisualProcessBench"


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
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


def patch_transformers_tied_weights_compat():
    from transformers.modeling_utils import PreTrainedModel

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def get_all_tied_weights_keys(self):
        stored_keys = getattr(self, "_all_tied_weights_keys", None)
        if stored_keys is not None:
            return stored_keys
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, (list, tuple, set)):
            return {key: None for key in keys}
        return {}

    def set_all_tied_weights_keys(self, value):
        self.__dict__["_all_tied_weights_keys"] = value

    PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)


def load_visualprm_model(model_path, dtype):
    from transformers import AutoModel, AutoTokenizer

    patch_transformers_tied_weights_compat()
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    ).eval().cuda()
    if not hasattr(model, "generate_steps_with_soft_score"):
        raise AttributeError(
            "Loaded model does not expose generate_steps_with_soft_score(). "
            "Please use OpenGVLab/VisualPRM-8B or a compatible local snapshot."
        )
    return tokenizer, model


def load_image_for_visualprm(image_path):
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
    return torch.stack([transform(img) for img in images])


def parse_soft_score_item(item):
    if isinstance(item, dict):
        for key in ("soft_score", "score", "Score", "reward"):
            if key in item:
                return float(item[key])
    if isinstance(item, (list, tuple)):
        for value in reversed(item):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(item)


def normalize_step_text(step):
    return "\n".join(line.strip() for line in str(step).splitlines() if line.strip())


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def score_visualprm_steps(tokenizer, model, question, steps, pixel_values):
    response = "\n\n".join(normalize_step_text(step) for step in steps)
    with torch.inference_mode():
        raw_scores = model.generate_steps_with_soft_score(
            tokenizer=tokenizer,
            question=question,
            response=response,
            pixel_values=pixel_values,
        )
    scores = [parse_soft_score_item(item) for item in raw_scores]
    return scores, raw_scores


def count_text_tokens(tokenizer, text):
    if not text:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        return len(encoded.get("input_ids", []))
    except Exception:
        return 0


def visualprm_text_token_counts(tokenizer, row):
    steps = row["response"]["steps"]
    question_tokens = count_text_tokens(tokenizer, row.get("question", ""))
    response_tokens = count_text_tokens(tokenizer, "\n\n".join(normalize_step_text(step) for step in steps))
    return {
        "question_tokens": question_tokens,
        "response_tokens": response_tokens,
        "text_input_tokens": question_tokens + response_tokens,
    }


def evaluate(args):
    rows = load_jsonl(Path(args.benchmark_dir) / "test.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    tokenizer, model = load_visualprm_model(args.model_path, args.dtype)

    progress = tqdm(total=len(rows), desc="VisualPRM", dynamic_ncols=True) if tqdm is not None else None
    errors = 0
    for idx, row in enumerate(rows):
        start = time.perf_counter()
        image_path = Path(args.benchmark_dir) / row["image"][0]
        steps = row["response"]["steps"]
        error = None
        scores = []
        raw_scores = []
        pred = [-2] * len(steps)
        image_preprocess_latency = 0.0
        score_latency = 0.0
        gpu_max_memory_allocated_mb = None
        gpu_max_memory_reserved_mb = None
        token_counts = visualprm_text_token_counts(tokenizer, row)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            image_start = time.perf_counter()
            pixel_values = load_image_for_visualprm(image_path).to(getattr(torch, args.dtype)).cuda()
            image_preprocess_latency = time.perf_counter() - image_start
            score_start = time.perf_counter()
            scores, raw_scores = score_visualprm_steps(tokenizer, model, row["question"], steps, pixel_values)
            score_latency = time.perf_counter() - score_start
            raw_scores = to_jsonable(raw_scores)
            if len(scores) == len(steps):
                pred = [1 if score > args.threshold else 0 for score in scores]
            else:
                error = f"score length mismatch: got {len(scores)}, expected {len(steps)}"
            if torch.cuda.is_available():
                gpu_max_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                gpu_max_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            error = repr(exc)
            errors += 1

        latency = time.perf_counter() - start
        result = {
            "backend": "visualprm",
            "mode": "paper_soft_score",
            "id": idx,
            "data_source": row.get("data_source"),
            "policy_model": row.get("policy_model"),
            "image": row.get("image", []),
            "image_paths": [str(image_path)],
            "num_images": len(row.get("image", [])),
            "num_steps": len(steps),
            "ground_truth": row["response"]["process_correctness"],
            "pred": pred,
            "scores": scores,
            "raw_response": raw_scores,
            "latency": latency,
            "latency_sec": latency,
            "image_preprocess_latency_sec": image_preprocess_latency,
            "score_latency_sec": score_latency,
            "num_requests": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_available_requests": 0,
            },
            "generated_tokens": 0,
            "generated_tokens_source": "not_applicable_visualprm_soft_score",
            "num_step_scores": len(scores),
            **token_counts,
            "gpu_max_memory_allocated_mb": gpu_max_memory_allocated_mb,
            "gpu_max_memory_reserved_mb": gpu_max_memory_reserved_mb,
            "error": error,
            "parse_error": any(value == -2 for value in pred),
        }
        append_jsonl(result, output_path)
        if progress is not None:
            progress.update(1)
            progress.set_postfix(errors=errors)
        elif (idx + 1) % args.log_every == 0 or idx + 1 == len(rows):
            print(f"completed {idx + 1}/{len(rows)}, errors={errors}")

    if progress is not None:
        progress.close()

    pred_rows = load_jsonl(output_path)
    metrics = compute_metrics(pred_rows)
    metrics_path = output_path.with_suffix(".metrics.json")
    write_json(metrics, metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"predictions: {output_path}")
    print(f"metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenGVLab/VisualPRM-8B on VisualProcessBench using generate_steps_with_soft_score()."
    )
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--model-path", required=True, help="Local or HF path for OpenGVLab/VisualPRM-8B.")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "visualprm_predictions.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.85, help="Predict correct when VisualPRM step score > threshold.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
