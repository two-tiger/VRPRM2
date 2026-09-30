#!/usr/bin/env python3
"""J1-style entropy data selection for the VRPRM RL split.

Serves the frozen SFT checkpoint behind an OpenAI-compatible endpoint (vLLM),
samples K rollouts per RL training item with the SAME prompt template as RL,
and measures how self-consistent the model's step judgments are:

- p_j = fraction of the K samples that judge confident step j as correct;
- uncertainty = mean binary entropy of p_j over confident steps
  (normalized to [0, 1]; 1 = judgments coin-flip across samples).

The output dataset is REORDERED by uncertainty (descending) so that, with
DATA_SHUFFLE=false, GRPO trains on the samples the SFT model is least sure
about first — the samples most likely to yield nonzero group advantages
(saturated / hopeless items are pushed to the end, and degenerate groups are
additionally skipped online by train_vrprm_rl_v2.sh).

Usage (after serving the merged SFT checkpoint):
    python easyr1/scripts/select_rl_data_by_entropy.py \
        --input  easyr1/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k/train.jsonl \
        --output-dir easyr1/data/visualprm400k_source_macro_rl_clean_pos0875_balanced_40k_entropy \
        --num-samples 4 --temperature 0.7

Resumable: per-item scores are appended to <output-dir>/item_scores.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
VRPRM_DIR = SCRIPT_DIR.parent / "examples" / "vrprm"
sys.path.insert(0, str(VRPRM_DIR))

from reward_source_macro import (  # noqa: E402
    _labels_from_target,
    _load_ground_truth,
    _parse_blocks,
    _parse_step_scores,
)

TEMPLATE_PLACEHOLDER = re.compile(r"\{\{\s*content\s*\|\s*trim\s*\}\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="RL train.jsonl (prompt/images/ground_truth).")
    parser.add_argument("--output-dir", required=True, help="Output dir for the entropy-ordered split.")
    parser.add_argument("--template", default=str(VRPRM_DIR / "vrprm_source_macro.jinja"),
                        help="Rollout prompt template (same file RL uses).")
    parser.add_argument("--image-dir", default=str(SCRIPT_DIR.parents[2] / "data"),
                        help="Root that the row image paths are relative to.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                        help="Comma-separated OpenAI-compatible endpoints of the SFT model.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--num-samples", type=int, default=4, help="K rollouts per item.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--top-k", type=int, default=0,
                        help="Keep only the top-K most uncertain items (0 = keep all, reordered).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for tie-break shuffling.")
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_prompt(template_text: str, content: str) -> str:
    return TEMPLATE_PLACEHOLDER.sub(content.strip(), template_text)


def encode_image(path: Path) -> dict[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def item_metrics(samples: list[str], labels: list[int]) -> dict[str, float]:
    confident = [idx for idx, label in enumerate(labels) if label in (0, 1)]
    parsed = []
    for text in samples:
        _, answer, _ = _parse_blocks(text)
        predictions = _parse_step_scores(answer)
        if predictions:
            parsed.append(predictions)

    parse_rate = len(parsed) / max(len(samples), 1)
    if not parsed or not confident:
        return {"uncertainty": 0.5, "accuracy": 0.0, "parse_rate": parse_rate}

    entropies = []
    correct = 0
    for idx in confident:
        p_positive = sum(1 for predictions in parsed if predictions.get(idx) == 1) / len(parsed)
        entropies.append(binary_entropy(p_positive))
        correct += sum(1 for predictions in parsed if predictions.get(idx) == labels[idx])
    accuracy = correct / (len(confident) * len(parsed))
    return {
        "uncertainty": sum(entropies) / len(entropies),
        "accuracy": accuracy,
        "parse_rate": parse_rate,
    }


async def sample_one(client: AsyncOpenAI, args: argparse.Namespace, model: str,
                     user_content: list[dict[str, Any]]) -> str:
    for attempt in range(args.max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
                timeout=args.request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception:
            if attempt == args.max_retries:
                return ""
            await asyncio.sleep(1.5 * (attempt + 1))
    return ""


async def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    template_text = Path(args.template).read_text(encoding="utf-8")
    image_root = Path(args.image_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "item_scores.jsonl"

    done: dict[str, dict[str, Any]] = {}
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done[record["id"]] = record
    print(f"Items: {len(rows)} total, {len(done)} already scored (resume).")

    base_urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [AsyncOpenAI(base_url=u, api_key=args.api_key) for u in base_urls]
    model = args.model
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def score_row(row: dict[str, Any], row_idx: int) -> None:
        target = _load_ground_truth(row.get("ground_truth", ""))
        item_id = str(target.get("source_id") or row_idx)
        if item_id in done:
            return
        async with semaphore:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": render_prompt(template_text, row["prompt"])}
            ]
            for rel in row.get("images", []):
                image_path = Path(rel)
                if not image_path.is_absolute():
                    image_path = image_root / rel
                if image_path.exists():
                    user_content.append(encode_image(image_path))
            client = clients[row_idx % len(clients)]
            if model == "auto":
                listed = await client.models.list()
                model_name = listed.data[0].id
            else:
                model_name = model
            samples = await asyncio.gather(*[
                sample_one(client, args, model_name, user_content) for _ in range(args.num_samples)
            ])
            labels = _labels_from_target(target)
            metrics = item_metrics(list(samples), labels)
            record = {"id": item_id, **metrics}
            async with lock:
                with scores_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[item_id] = record
                if len(done) % 50 == 0:
                    print(f"scored {len(done)} / {len(rows)}")

    await asyncio.gather(*[score_row(row, idx) for idx, row in enumerate(rows)])

    # Reorder: most uncertain first; ties broken by lower accuracy, then id.
    import random

    rng = random.Random(args.seed)
    def order_key(row: dict[str, Any], row_idx: int) -> tuple[float, float, float, str]:
        target = _load_ground_truth(row.get("ground_truth", ""))
        item_id = str(target.get("source_id") or row_idx)
        record = done.get(item_id, {"uncertainty": 0.5, "accuracy": 0.0, "parse_rate": 0.0})
        jitter = rng.random() * 1e-6
        return (-record["uncertainty"], record["accuracy"], jitter, item_id)

    indexed = sorted(enumerate(rows), key=lambda pair: order_key(pair[1], pair[0]))
    ordered_rows = [row for _, row in indexed]
    if args.top_k and args.top_k > 0:
        ordered_rows = ordered_rows[: args.top_k]

    train_out = out_dir / "train.jsonl"
    with train_out.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    uncertainties = [done[k]["uncertainty"] for k in done]
    stats = {
        "input": str(args.input),
        "output_train": str(train_out),
        "template": str(args.template),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "items_total": len(rows),
        "items_scored": len(done),
        "items_kept": len(ordered_rows),
        "top_k": args.top_k,
        "uncertainty_mean": sum(uncertainties) / len(uncertainties) if uncertainties else 0.0,
        "uncertainty_p50": sorted(uncertainties)[len(uncertainties) // 2] if uncertainties else 0.0,
        "note": "Reordered by judgment uncertainty (descending). Train with DATA_SHUFFLE=false.",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / ".entropy_selected").write_text(
        json.dumps({"seed": args.seed, "num_samples": args.num_samples}) + "\n", encoding="utf-8"
    )

    # Copy the validation split unchanged so the launcher finds it.
    val_src = Path(args.input).parent / "test.jsonl"
    if val_src.exists():
        (out_dir / "test.jsonl").write_text(val_src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
