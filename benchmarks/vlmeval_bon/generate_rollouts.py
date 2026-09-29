import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from common import (
    append_jsonl,
    env_default,
    load_jsonl,
    make_policy_prompt,
    openai_content,
    prompt_items_to_openai_content,
    resolve_dataset_name,
)


class AsyncOpenAIClientPool:
    def __init__(self, base_urls: str, api_key: str):
        urls = [url.strip() for url in str(base_urls).split(",") if url.strip()]
        if not urls:
            raise ValueError("--policy-base-url must contain at least one endpoint")
        self.items = [(url, AsyncOpenAI(base_url=url, api_key=api_key)) for url in urls]
        self.lock = asyncio.Lock()
        self.next_idx = 0

    async def next_client(self) -> Tuple[str, AsyncOpenAI]:
        async with self.lock:
            item = self.items[self.next_idx % len(self.items)]
            self.next_idx += 1
            return item

    def __len__(self) -> int:
        return len(self.items)


def candidate_cache_path(output: Path) -> Path:
    return output.with_suffix(".candidates.jsonl")


def latest_rows_by_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest = {}
    for row in rows:
        if "index" in row:
            latest[str(row["index"])] = row
    return latest


def load_successful_candidate_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    candidates = {}
    for row in load_jsonl(path):
        if "index" not in row or "candidate_idx" not in row:
            continue
        if row.get("error"):
            continue
        rollout = str(row.get("rollout") or "")
        if not rollout.strip():
            continue
        key = (str(row["index"]), int(row["candidate_idx"]))
        candidates[key] = row
    return candidates


def seed_candidates_from_aggregate(
    rows: List[Dict[str, Any]],
    candidates: Dict[Tuple[str, int], Dict[str, Any]],
    bon: int,
) -> int:
    added = 0
    for row in rows:
        if "index" not in row:
            continue
        index = str(row["index"])
        rollouts = row.get("rollouts") or []
        errors = row.get("errors") or []
        for candidate_idx, rollout in enumerate(rollouts[:bon]):
            key = (index, candidate_idx)
            if key in candidates:
                continue
            if candidate_idx < len(errors) and errors[candidate_idx]:
                continue
            if not str(rollout or "").strip():
                continue
            candidate = dict(row)
            candidate.update(
                {
                    "candidate_idx": candidate_idx,
                    "rollout": rollout,
                    "error": "",
                }
            )
            candidate.pop("rollouts", None)
            candidate.pop("errors", None)
            candidates[key] = candidate
            added += 1
    return added


def seed_candidate_rows(seed_output: Path, candidates: Dict[Tuple[str, int], Dict[str, Any]], bon: int) -> int:
    seed_candidates_path = candidate_cache_path(seed_output)
    if not seed_output.exists() and not seed_candidates_path.exists():
        return 0
    before = len(candidates)
    if seed_candidates_path.exists():
        seed_candidates = load_successful_candidate_rows(seed_candidates_path)
        for key, row in seed_candidates.items():
            if key[1] < bon and key not in candidates:
                candidates[key] = row
    if seed_output.exists():
        seed_candidates_from_aggregate(load_jsonl(seed_output), candidates, bon)
    return len(candidates) - before


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or resume cached BoN policy rollouts.")
    parser.add_argument("--dataset", required=True, help="Dataset alias/name, e.g. MathVerse-VO.")
    parser.add_argument("--output", required=True, help="Sample-level JSONL output, e.g. rollouts_n128.jsonl.")
    parser.add_argument("--seed-output", default=env_default("SEED_ROLLOUTS", ""),
                        help="Optional older rollout JSONL used to seed existing candidates, e.g. rollouts_n128.jsonl.")
    parser.add_argument("--policy-base-url", default=env_default("POLICY_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--policy-api-key", default=env_default("POLICY_API_KEY", "EMPTY"))
    parser.add_argument("--policy-model", default=env_default("POLICY_MODEL", "auto"))
    parser.add_argument("--bon", type=int, default=int(env_default("ROLLOUT_N", 128)))
    parser.add_argument("--concurrency", type=int, default=int(env_default("POLICY_CONCURRENCY", 64)))
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=int(env_default("POLICY_ROLLOUT_CONCURRENCY", 1)),
        help="Max candidate requests per sample.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(env_default("POLICY_MAX_TOKENS", 2048)))
    parser.add_argument("--temperature", type=float, default=float(env_default("POLICY_TEMPERATURE", 0.7)))
    parser.add_argument("--top-p", type=float, default=float(env_default("POLICY_TOP_P", 0.95)))
    parser.add_argument("--max-retries", type=int, default=int(env_default("POLICY_MAX_RETRIES", 3)))
    parser.add_argument("--request-timeout", type=float, default=float(env_default("POLICY_REQUEST_TIMEOUT", 600)))
    parser.add_argument("--limit", type=int, default=int(env_default("DATA_LIMIT", 0)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_records(args) -> List[Dict[str, Any]]:
    from vlmeval.dataset import DATASET_TYPE, build_dataset

    dataset_name = resolve_dataset_name(args.dataset)
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")
    dataset_type = DATASET_TYPE(dataset_name)

    records = []
    for row_pos in range(len(dataset.data)):
        line = dataset.data.iloc[row_pos]
        row = dict(line)
        prompt_items = dataset.build_prompt(line)
        question, image_paths = prompt_items_to_openai_content(prompt_items)
        prompt = make_policy_prompt(question, dataset_type)
        records.append(
            {
                "dataset": args.dataset,
                "vlmeval_dataset": dataset_name,
                "dataset_type": dataset_type,
                "index": str(row.get("index")),
                "row_pos": row_pos,
                "prompt": prompt,
                "question": question,
                "answer": "" if row.get("answer") is None else str(row.get("answer")),
                "images": image_paths,
                "policy_model": args.policy_model,
            }
        )
        if args.limit and len(records) >= args.limit:
            break
    return records


async def request_one(client_pool: AsyncOpenAIClientPool, args, record: Dict[str, Any], candidate_idx: int) -> Dict[str, Any]:
    error = ""
    rollout = ""
    policy_base_url = ""
    for attempt in range(args.max_retries):
        try:
            policy_base_url, client = await client_pool.next_client()
            request = client.chat.completions.create(
                model=args.policy_model,
                messages=[
                    {
                        "role": "user",
                        "content": openai_content(record["prompt"], record["images"]),
                    }
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            if args.request_timeout > 0:
                response = await asyncio.wait_for(request, timeout=args.request_timeout)
            else:
                response = await request
            rollout = response.choices[0].message.content or ""
            error = ""
            break
        except Exception as exc:
            error = repr(exc)
            await asyncio.sleep(1.5 * (attempt + 1))

    row = dict(record)
    row.update(
        {
            "candidate_idx": candidate_idx,
            "rollout": rollout,
            "error": error,
            "policy_base_url": policy_base_url,
        }
    )
    return row


def aggregate_row(record: Dict[str, Any], candidate_rows: Dict[Tuple[str, int], Dict[str, Any]], bon: int, output: Path):
    index = str(record["index"])
    rollouts = []
    errors = []
    for candidate_idx in range(bon):
        row = candidate_rows.get((index, candidate_idx))
        if row is None:
            return None
        rollouts.append(row.get("rollout", ""))
        errors.append(row.get("error", ""))

    out = dict(record)
    out.update(
        {
            "rollouts": rollouts,
            "errors": errors,
            "num_rollouts": len(rollouts),
            "candidate_cache": str(candidate_cache_path(output).resolve()),
        }
    )
    return out


async def main_async():
    args = parse_args()
    if args.bon < 1:
        raise ValueError("--bon must be >= 1")

    output = Path(args.output)
    candidates_path = candidate_cache_path(output)
    if args.overwrite:
        output.unlink(missing_ok=True)
        candidates_path.unlink(missing_ok=True)

    records = build_records(args)
    aggregate_rows = latest_rows_by_index(load_jsonl(output))
    candidate_rows = load_successful_candidate_rows(candidates_path)
    seeded_candidates = 0
    if args.seed_output:
        seeded_candidates = seed_candidate_rows(Path(args.seed_output), candidate_rows, args.bon)

    client_pool = AsyncOpenAIClientPool(args.policy_base_url, args.policy_api_key)
    request_semaphore = asyncio.Semaphore(max(1, args.concurrency))

    pending_records = []
    already_complete = 0
    for record in records:
        index = str(record["index"])
        existing = aggregate_rows.get(index)
        if existing and len(existing.get("rollouts") or []) >= args.bon:
            already_complete += 1
            continue
        full = aggregate_row(record, candidate_rows, args.bon, output)
        if full is not None:
            append_jsonl(output, full)
            aggregate_rows[index] = full
            already_complete += 1
            continue
        pending_records.append(record)

    iterator = pending_records
    progress = None
    if tqdm is not None and not args.no_progress:
        progress = tqdm(total=len(pending_records), desc=f"{args.dataset} Bo{args.bon}")

    async def generate_missing_for_record(record: Dict[str, Any]):
        index = str(record["index"])
        per_sample_semaphore = asyncio.Semaphore(max(1, args.rollout_concurrency))
        missing = [
            candidate_idx
            for candidate_idx in range(args.bon)
            if (index, candidate_idx) not in candidate_rows
        ]

        async def one(candidate_idx: int):
            async with per_sample_semaphore:
                async with request_semaphore:
                    row = await request_one(client_pool, args, record, candidate_idx)
                    append_jsonl(candidates_path, row)
                    if not row.get("error") and str(row.get("rollout") or "").strip():
                        candidate_rows[(index, candidate_idx)] = row

        await asyncio.gather(*(one(candidate_idx) for candidate_idx in missing))
        full = aggregate_row(record, candidate_rows, args.bon, output)
        if full is not None:
            append_jsonl(output, full)
            aggregate_rows[index] = full
        if progress is not None:
            progress.update(1)

    await asyncio.gather(*(generate_missing_for_record(record) for record in iterator))
    if progress is not None:
        progress.close()

    complete = sum(
        1
        for record in records
        if str(record["index"]) in aggregate_rows
        and len(aggregate_rows[str(record["index"])].get("rollouts") or []) >= args.bon
    )
    print(
        f"dataset={args.dataset} output={output} complete={complete}/{len(records)} "
        f"already_complete={already_complete} pending_at_start={len(pending_records)} "
        f"seeded_candidates={seeded_candidates} policy_endpoints={len(client_pool)}"
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
