import argparse
import json
import os
from pathlib import Path

from common import (
    apply_eval_api_env,
    default_judge_for,
    env_default,
    probe_eval_judge_api,
    resolve_dataset_name,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a BoN-selected prediction xlsx with VLMEvalKit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prediction", required=True, help="Prediction xlsx created by select_bon_from_scores.py.")
    parser.add_argument("--output", required=True, help="Evaluation result JSON/CSV path.")
    parser.add_argument("--judge", default=env_default("EVAL_JUDGE", None))
    parser.add_argument("--api-nproc", type=int, default=int(env_default("EVAL_API_NPROC", 4)))
    parser.add_argument("--retry", type=int, default=int(env_default("EVAL_RETRY", 3)))
    parser.add_argument("--judge-args", default=env_default("EVAL_JUDGE_ARGS", None), help="Extra judge kwargs as JSON.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    import pandas as pd
    from vlmeval.dataset import DATASET_TYPE, build_dataset

    args = parse_args()
    apply_eval_api_env()
    dataset_name = resolve_dataset_name(args.dataset)
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"unsupported VLMEvalKit dataset: {dataset_name}")

    judge = args.judge or default_judge_for(dataset_name, DATASET_TYPE(dataset_name))
    judge_kwargs = {
        "model": judge,
        "nproc": args.api_nproc,
        "retry": args.retry,
        "verbose": args.verbose,
    }
    if args.judge_args:
        judge_kwargs.update(json.loads(args.judge_args))

    print(
        f"VLMEvalKit evaluate dataset={dataset_name}, judge={judge}, "
        f"OPENAI_API_BASE={os.environ.get('OPENAI_API_BASE', '')}"
    )
    try:
        result = dataset.evaluate(args.prediction, **judge_kwargs)
    except AssertionError as err:
        probe_eval_judge_api(
            judge_kwargs,
            context=f"dataset={dataset_name}, prediction={args.prediction}",
        )
        raise AssertionError(
            f"{err}\n"
            f"Current VLMEvalKit judge={judge}. "
            f"OPENAI_API_BASE={os.environ.get('OPENAI_API_BASE', '')}. "
            "For OpenAI-compatible endpoints, OPENAI_API_BASE/EVAL_API_BASE must point to "
            "the chat completions endpoint, e.g. http://host:port/v1/chat/completions. "
            "MMMU may finish even when this check fails because MCQ evaluation can fall back "
            "to exact matching; MathVista/MathVerse/MathVision require a working judge API."
        ) from err
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, pd.DataFrame):
        if output.suffix.lower() == ".json":
            write_json(output, result.to_dict(orient="records"))
        else:
            result.to_csv(output, index=False)
    elif isinstance(result, dict):
        write_json(output, result)
    else:
        write_json(output, {"result": None})

    print(f"wrote evaluation result: {output}")


if __name__ == "__main__":
    main()
