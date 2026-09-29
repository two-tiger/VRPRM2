import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "rollout_outputs"
DEFAULT_NEGATIVE_PATH = DEFAULT_OUTPUT_DIR / "visualprm400k_negative_global_think_stepwise_sft_success.json"
DEFAULT_POSITIVE_PATH = DEFAULT_OUTPUT_DIR / "visualprm400k_positive_global_think_stepwise_sft_success.json"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "visualprm400k_global_think_stepwise_multiturn_sft_neg35.json"

GLOBAL_THINK_MULTITURN_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, first produce global thinking for later step-level scoring.

The global thinking should identify the key visual evidence, the question goal, the reference-answer constraint, dependencies between candidate steps, and the main correctness issues. It may include a compact step-level overview when useful.

After global thinking, answer follow-up step-scoring turns with exactly one token: 1 or 0."""

STEP_SCORE_TURN_PROMPT = """[Previous Step Judgments]
{previous_judgments}
[Current Step]
{current_step}
Is the current step correct in context? Output only one token: 1 or 0."""

NO_THINK_MULTITURN_SYSTEM_PROMPT = """You are a visual process reward model. Given image(s), a problem, a reference answer, and a complete candidate solution, judge each candidate solution step.

Answer every step-scoring turn with exactly one token: 1 or 0. Do not reason out loud and do not output <think>...</think>."""

NO_THINK_TAG = "_nothink"


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sample_key(row, polarity):
    return (
        polarity,
        row.get("source_annotation"),
        json.dumps(row.get("source_image"), ensure_ascii=False, sort_keys=True),
        row.get("source_sample_id"),
    )


def group_rows(rows, polarity):
    groups = defaultdict(lambda: {"global": None, "steps": []})
    for row in rows:
        key = sample_key(row, polarity)
        if row.get("task_type") == "global_thinking":
            groups[key]["global"] = row
        elif row.get("task_type") == "step_score":
            groups[key]["steps"].append(row)

    complete = []
    dropped = Counter()
    for key, group in groups.items():
        global_row = group["global"]
        steps = sorted(group["steps"], key=lambda x: x.get("step_index", -1))
        if global_row is None:
            dropped["missing_global"] += 1
            continue
        if not steps:
            dropped["missing_step_rows"] += 1
            continue
        complete.append((key, global_row, steps))
    return complete, dropped


def parse_step_lines(global_user_content):
    marker = "[Candidate Solution]\n"
    if marker not in global_user_content:
        return []
    solution = global_user_content.split(marker, 1)[1]
    lines = []
    current = []
    for line in solution.splitlines():
        if line.startswith("Step ") and current:
            lines.append("\n".join(current).strip())
            current = [line]
        elif line.strip() or current:
            current.append(line)
    if current:
        lines.append("\n".join(current).strip())
    return lines


def build_previous_text(previous_scores):
    if not previous_scores:
        return "None"
    return "\n".join(f"Step {idx}: \\boxed{{{score}}}" for idx, score in enumerate(previous_scores))


def build_step_user(current_step, previous_scores):
    return STEP_SCORE_TURN_PROMPT.format(
        previous_judgments=build_previous_text(previous_scores),
        current_step=current_step,
    )


def confident_label(score, negative_threshold, positive_threshold):
    score = float(score)
    if score <= negative_threshold:
        return 0
    if score >= positive_threshold:
        return 1
    return None


def build_multiturn_sample(
    key,
    global_row,
    step_rows,
    polarity,
    global_loss_scale,
    step_loss_scale,
    negative_threshold,
    positive_threshold,
    no_think=False,
):
    messages = []
    global_messages = global_row["messages"]
    global_user = global_messages[1]["content"]
    if no_think:
        messages.append({"role": "system", "content": NO_THINK_MULTITURN_SYSTEM_PROMPT})
    else:
        global_assistant = dict(global_messages[2])
        global_assistant["loss_scale"] = global_loss_scale
        messages.append({"role": "system", "content": GLOBAL_THINK_MULTITURN_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": global_user})
        messages.append(global_assistant)

    step_lines = parse_step_lines(global_user)
    source_scores = [float(x) for x in global_row.get("source_scores", [])]
    kimi_scores = [int(x) for x in global_row.get("kimi_scores", [])]

    step_by_index = {int(row["step_index"]): row for row in step_rows}
    max_steps = int(global_row.get("num_steps") or len(source_scores) or len(kimi_scores) or len(step_lines))
    if not step_lines:
        step_lines = [f"Step {idx}" for idx in range(max_steps)]

    previous_scores = []
    trainable_labels = []
    ignored_labels = []
    missing_confident_rows = []
    for idx in range(max_steps):
        current_step = step_lines[idx] if idx < len(step_lines) else f"Step {idx}"
        label = None
        if idx < len(source_scores):
            label = confident_label(source_scores[idx], negative_threshold, positive_threshold)
        row = step_by_index.get(idx)
        if label is not None and row is None:
            missing_confident_rows.append(idx)
            score = int(kimi_scores[idx]) if idx < len(kimi_scores) else int(label)
            loss = False
            ignored_labels.append(score)
        elif label is None:
            score = int(kimi_scores[idx]) if idx < len(kimi_scores) else 0
            loss = False
            ignored_labels.append(score)
        else:
            score = int(row.get("step_label", label))
            loss = True
            trainable_labels.append(score)

        if no_think:
            # Evaluation-aligned no-think stepwise interface: every step turn
            # repeats the full question/reference/solution context, then adds
            # previous judgments and the current step (mirrors the evaluator's
            # no_think_stepwise mode).
            step_user = (
                f"{global_user}\n{build_step_user(current_step, previous_scores)}"
            )
        else:
            step_user = build_step_user(current_step, previous_scores)
        messages.append({"role": "user", "content": step_user})
        assistant = {"role": "assistant", "content": str(score)}
        if loss:
            assistant["loss_scale"] = step_loss_scale
        else:
            assistant["loss"] = False
        messages.append(assistant)
        previous_scores.append(score)

    neg_steps = sum(1 for x in trainable_labels if x == 0)
    pos_steps = sum(1 for x in trainable_labels if x == 1)

    return {
        "messages": messages,
        "images": global_row.get("images", []),
        "source_image": global_row.get("source_image"),
        "source_annotation": global_row.get("source_annotation"),
        "source_sample_id": global_row.get("source_sample_id"),
        "source_key": list(key),
        "source_polarity": polarity,
        "task_type": "no_think_stepwise_multiturn" if no_think else "global_think_stepwise_multiturn",
        "num_steps": max_steps,
        "num_trainable_steps": len(trainable_labels),
        "num_ignored_steps": len(ignored_labels),
        "negative_trainable_steps": neg_steps,
        "positive_trainable_steps": pos_steps,
        "source_scores": source_scores,
        "kimi_scores": kimi_scores,
        "negative_score_threshold": negative_threshold,
        "positive_score_threshold": positive_threshold,
        "missing_confident_step_rows": missing_confident_rows,
    }


def trainable_counts(samples):
    neg = sum(sample.get("negative_trainable_steps", 0) for sample in samples)
    pos = sum(sample.get("positive_trainable_steps", 0) for sample in samples)
    return neg, pos


def select_positive_samples(negative_samples, positive_samples, target_negative_ratio, seed):
    neg_steps, pos_steps_from_negative = trainable_counts(negative_samples)
    if neg_steps <= 0:
        raise ValueError("No negative trainable steps found in negative samples.")

    target_positive_steps = round(neg_steps * (1 - target_negative_ratio) / target_negative_ratio)
    needed_positive_steps = max(0, target_positive_steps - pos_steps_from_negative)

    rng = random.Random(seed)
    candidates = positive_samples[:]
    rng.shuffle(candidates)

    selected = []
    selected_pos_steps = 0
    for sample in candidates:
        step_count = sample.get("positive_trainable_steps", 0)
        if step_count <= 0:
            continue
        if selected_pos_steps >= needed_positive_steps:
            break
        selected.append(sample)
        selected_pos_steps += step_count

    return selected, {
        "target_negative_ratio": target_negative_ratio,
        "negative_steps": neg_steps,
        "positive_steps_from_negative_file": pos_steps_from_negative,
        "target_positive_steps_total": target_positive_steps,
        "needed_positive_steps_from_positive_file": needed_positive_steps,
        "selected_positive_steps_from_positive_file": selected_pos_steps,
        "selected_positive_source_samples": len(selected),
        "available_positive_source_samples": len(positive_samples),
    }


def summarize(samples):
    neg, pos = trainable_counts(samples)
    ignored = sum(sample.get("num_ignored_steps", 0) for sample in samples)
    return {
        "samples": len(samples),
        "trainable_steps": neg + pos,
        "negative_trainable_steps": neg,
        "positive_trainable_steps": pos,
        "negative_ratio": neg / (neg + pos) if neg + pos else 0,
        "ignored_steps": ignored,
        "messages": sum(len(sample.get("messages", [])) for sample in samples),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build source-level multi-turn global-thinking + stepwise SFT data."
    )
    parser.add_argument("--negative-path", default=str(DEFAULT_NEGATIVE_PATH))
    parser.add_argument("--positive-path", default=str(DEFAULT_POSITIVE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--target-negative-ratio", type=float, default=0.35)
    parser.add_argument("--negative-score-threshold", type=float, default=0.125)
    parser.add_argument("--positive-score-threshold", type=float, default=0.75)
    parser.add_argument("--global-loss-scale", type=float, default=0.5)
    parser.add_argument("--step-loss-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-think",
        action="store_true",
        help=(
            "Build the no-thinking ablation dataset: drop the global-thinking "
            "turn and put the full context into every stepwise scoring turn "
            "(evaluation-aligned with the no_think_stepwise evaluator mode)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.target_negative_ratio < 1:
        raise ValueError("--target-negative-ratio must be between 0 and 1.")
    if args.negative_score_threshold >= args.positive_score_threshold:
        raise ValueError("--negative-score-threshold must be smaller than --positive-score-threshold.")

    if args.no_think and args.output_path == str(DEFAULT_OUTPUT_PATH):
        args.output_path = str(Path(args.output_path).with_suffix(".json")).replace(
            "_multiturn_sft_neg35", "_multiturn_sft_neg35" + NO_THINK_TAG
        )

    negative_rows = load_json(args.negative_path)
    positive_rows = load_json(args.positive_path)
    negative_groups, negative_dropped = group_rows(negative_rows, "negative")
    positive_groups, positive_dropped = group_rows(positive_rows, "positive")

    negative_samples = [
        build_multiturn_sample(
            key,
            global_row,
            step_rows,
            "negative",
            args.global_loss_scale,
            args.step_loss_scale,
            args.negative_score_threshold,
            args.positive_score_threshold,
            no_think=args.no_think,
        )
        for key, global_row, step_rows in negative_groups
    ]
    positive_samples = [
        build_multiturn_sample(
            key,
            global_row,
            step_rows,
            "positive",
            args.global_loss_scale,
            args.step_loss_scale,
            args.negative_score_threshold,
            args.positive_score_threshold,
            no_think=args.no_think,
        )
        for key, global_row, step_rows in positive_groups
    ]

    selected_positive, sampling_stats = select_positive_samples(
        negative_samples,
        positive_samples,
        target_negative_ratio=args.target_negative_ratio,
        seed=args.seed,
    )

    final_data = negative_samples + selected_positive
    random.Random(args.seed).shuffle(final_data)

    stats = {
        "negative_input": str(args.negative_path),
        "positive_input": str(args.positive_path),
        "output": str(args.output_path),
        "negative_dropped_groups": dict(negative_dropped),
        "positive_dropped_groups": dict(positive_dropped),
        "negative_samples": summarize(negative_samples),
        "positive_samples_available": summarize(positive_samples),
        "positive_samples_selected": summarize(selected_positive),
        "sampling": sampling_stats,
        "final": summarize(final_data),
        "no_think": args.no_think,
        "global_loss_scale": args.global_loss_scale,
        "step_loss_scale": args.step_loss_scale,
        "note": "Use --loss_scale default and --is_binary_loss_scale false in ms-swift.",
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    save_json(final_data, args.output_path)
    stats_path = Path(args.stats_path) if args.stats_path else Path(args.output_path).with_suffix(".stats.json")
    save_json(stats, stats_path)


if __name__ == "__main__":
    main()
