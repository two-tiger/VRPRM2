#!/usr/bin/env python3
"""Plot SFT and RL training curves for VRPRM-v2 paper figures.

The script parses the json/jsonl logs produced by ms-swift SFT and EasyR1 RL.
It intentionally avoids TensorBoard dependencies so that it can run in a
minimal analysis environment with only matplotlib installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_DIR = (
    REPO_ROOT
    / "sft/output/qwen3_vl_8b_thinking_global_stepwise_multiturn_sft/v0-20260628-015250"
)
DEFAULT_RL_DIR = (
    REPO_ROOT
    / "easyr1/checkpoints/visualprm_easy_r1/"
    "qwen3_vl_8b_source_macro_visualprm400k_gspo_lora_clean_balanced_40k_450"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures/training_curves"


Point = Tuple[float, float]
Series = Dict[str, List[Point]]
ChartLine = Tuple[Series, str, str, int]
ChartSpec = Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def parse_step(record: Dict[str, Any]) -> Optional[int]:
    step = record.get("step")
    if isinstance(step, (int, float)):
        return int(step)
    step_text = record.get("global_step/max_steps")
    if isinstance(step_text, str) and "/" in step_text:
        left = step_text.split("/", 1)[0].strip()
        if left.isdigit():
            return int(left)
    step = record.get("global_step")
    if isinstance(step, (int, float)):
        return int(step)
    return None


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def flatten(prefix: str, obj: Dict[str, Any], out: Dict[str, float]) -> None:
    for key, value in obj.items():
        new_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flatten(new_key, value, out)
        else:
            val = numeric(value)
            if val is not None:
                out[new_key] = val


def add_point(series: Series, name: str, step: Optional[int], value: Any) -> None:
    val = numeric(value)
    if step is not None and val is not None:
        series[name].append((float(step), val))


def dedupe_and_sort(points: Sequence[Point]) -> List[Point]:
    by_step: Dict[float, float] = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items(), key=lambda x: x[0])


def smooth_points(points: Sequence[Point], window: int) -> List[Point]:
    clean = dedupe_and_sort(points)
    if window <= 1 or len(clean) < 3:
        return clean
    window = min(window, len(clean))
    smoothed: List[Point] = []
    values = [v for _, v in clean]
    for idx, (step, _) in enumerate(clean):
        start = max(0, idx - window + 1)
        local = values[start : idx + 1]
        smoothed.append((step, sum(local) / len(local)))
    return smoothed


def load_sft_series(sft_dir: Path) -> Tuple[Series, Dict[str, Any]]:
    sft_dir = sft_dir.resolve()
    series: Series = defaultdict(list)
    meta: Dict[str, Any] = {"source": str(sft_dir)}

    log_path = sft_dir / "logging.jsonl"
    records: List[Dict[str, Any]] = []
    if log_path.exists():
        records = read_jsonl(log_path)
        meta["log_path"] = str(log_path)

    if not records:
        trainer_states = sorted(sft_dir.glob("checkpoint-*/trainer_state.json"))
        if trainer_states:
            state_path = trainer_states[-1]
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            records = state.get("log_history", [])
            meta["log_path"] = str(state_path)

    for record in records:
        # The final ms-swift jsonl line embeds the full log_history. Skipping it
        # prevents duplicate points when logging.jsonl already has each step.
        if "log_history" in record:
            meta["final_summary"] = {
                k: v for k, v in record.items() if k not in {"log_history"}
            }
            continue
        step = parse_step(record)
        add_point(series, "train/loss", step, record.get("loss"))
        add_point(series, "train/token_acc", step, record.get("token_acc"))
        add_point(series, "train/learning_rate", step, record.get("learning_rate"))
        add_point(series, "train/grad_norm", step, record.get("grad_norm"))
        add_point(series, "eval/loss", step, record.get("eval_loss"))
        add_point(series, "eval/token_acc", step, record.get("eval_token_acc"))

    for key in list(series):
        series[key] = dedupe_and_sort(series[key])

    if series.get("eval/loss"):
        best_step, best_loss = min(series["eval/loss"], key=lambda x: x[1])
        meta["best_eval_loss_step"] = int(best_step)
        meta["best_eval_loss"] = best_loss

    return series, meta


def maybe_resolve_full_rl_dir(rl_dir: Path) -> Path:
    """Return the full EasyR1 run dir when a selected checkpoint dir is passed."""
    rl_dir = rl_dir.resolve()
    current_log = rl_dir / "experiment_log.jsonl"
    current_lines = sum(1 for _ in current_log.open("r", encoding="utf-8")) if current_log.exists() else 0

    candidates: List[Path] = [rl_dir]
    tracker = rl_dir / "checkpoint_tracker.json"
    if tracker.exists():
        try:
            with tracker.open("r", encoding="utf-8") as f:
                data = json.load(f)
            actor_path = data.get("last_actor_path")
            if actor_path:
                actor_dir = Path(actor_path).resolve()
                if actor_dir.name == "actor" and len(actor_dir.parents) >= 2:
                    candidates.append(actor_dir.parents[1])
        except (OSError, json.JSONDecodeError):
            pass

    # If the selected checkpoint has only a tail copy of logs, prefer a sibling
    # full run whose name shares the selected checkpoint prefix.
    parent = rl_dir.parent
    prefix = rl_dir.name.rsplit("_", 1)[0]
    for sibling in parent.glob(f"{prefix}_*"):
        if sibling.is_dir():
            candidates.append(sibling.resolve())

    best_dir = rl_dir
    best_lines = current_lines
    for candidate in candidates:
        log_path = candidate / "experiment_log.jsonl"
        if not log_path.exists():
            continue
        try:
            lines = sum(1 for _ in log_path.open("r", encoding="utf-8"))
        except OSError:
            continue
        if lines > best_lines:
            best_dir = candidate
            best_lines = lines

    return best_dir


def infer_selected_rl_step(rl_dir: Path) -> Optional[int]:
    for child in sorted(rl_dir.glob("global_step_*")):
        suffix = child.name.replace("global_step_", "", 1)
        if suffix.isdigit():
            return int(suffix)
    tail = rl_dir.name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def load_rl_series(rl_dir: Path, selected_step: Optional[int] = None) -> Tuple[Series, Dict[str, Any]]:
    resolved_dir = maybe_resolve_full_rl_dir(rl_dir)
    log_path = resolved_dir / "experiment_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find RL log: {log_path}")

    series: Series = defaultdict(list)
    meta: Dict[str, Any] = {
        "source": str(rl_dir.resolve()),
        "resolved_source": str(resolved_dir),
        "log_path": str(log_path),
    }
    if selected_step is None:
        selected_step = infer_selected_rl_step(rl_dir.resolve())
    if selected_step is not None:
        meta["selected_rl_step"] = int(selected_step)

    tracker = resolved_dir / "checkpoint_tracker.json"
    if tracker.exists():
        try:
            with tracker.open("r", encoding="utf-8") as f:
                meta["checkpoint_tracker"] = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    for record in read_jsonl(log_path):
        step = parse_step(record)
        if step is None:
            continue
        flat: Dict[str, float] = {}
        flatten("", record, flat)
        for key, value in flat.items():
            if key == "step":
                continue
            series[key].append((float(step), value))

    for key in list(series):
        series[key] = dedupe_and_sort(series[key])

    if series.get("val/reward_score"):
        best_step, best_reward = max(series["val/reward_score"], key=lambda x: x[1])
        meta["best_val_reward_step"] = int(best_step)
        meta["best_val_reward"] = best_reward
    return series, meta


def import_matplotlib(optional: bool = False):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        if optional:
            return None
        raise SystemExit(
            "matplotlib is required. Install it in the plotting environment first."
        ) from exc
    return plt


def configure_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
        }
    )


def plot_series(
    ax: Any,
    series: Series,
    key: str,
    label: str,
    smooth_window: int,
    *,
    marker: Optional[str] = None,
    linestyle: str = "-",
) -> bool:
    points = series.get(key, [])
    if not points:
        return False
    smooth_window = 1 if len(points) <= 8 else smooth_window
    points = smooth_points(points, smooth_window)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    ax.plot(xs, ys, label=label, marker=marker, markersize=3, linestyle=linestyle)
    return True


def finalize_axis(ax: Any, title: str, ylabel: str, xlabel: str = "Training step") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_sft(plt: Any, sft: Series, meta: Dict[str, Any], output_dir: Path, smooth: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    axes = axes.reshape(-1)

    plot_series(axes[0], sft, "train/loss", "train loss", smooth)
    plot_series(axes[0], sft, "eval/loss", "eval loss", 1, marker="o")
    if "best_eval_loss_step" in meta:
        axes[0].axvline(meta["best_eval_loss_step"], color="0.35", linestyle="--", linewidth=1)
    finalize_axis(axes[0], "SFT Loss", "Loss")

    plot_series(axes[1], sft, "train/token_acc", "train token acc", smooth)
    plot_series(axes[1], sft, "eval/token_acc", "eval token acc", 1, marker="o")
    finalize_axis(axes[1], "SFT Token Accuracy", "Accuracy")

    plot_series(axes[2], sft, "train/learning_rate", "learning rate", smooth)
    finalize_axis(axes[2], "SFT Learning Rate", "LR")

    plot_series(axes[3], sft, "train/grad_norm", "grad norm", smooth)
    finalize_axis(axes[3], "SFT Grad Norm", "Norm")

    save_figure(fig, output_dir, "sft_training_curves")
    plt.close(fig)


def plot_rl(plt: Any, rl: Series, meta: Dict[str, Any], output_dir: Path, smooth: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0))
    axes = axes.reshape(-1)

    plot_series(axes[0], rl, "reward/overall", "train overall", smooth)
    plot_series(axes[0], rl, "reward/step", "train step", smooth)
    plot_series(axes[0], rl, "val/reward_score", "val reward", 1, marker="o")
    if "selected_rl_step" in meta:
        axes[0].axvline(meta["selected_rl_step"], color="0.35", linestyle="--", linewidth=1)
    finalize_axis(axes[0], "RL Reward", "Reward")

    plot_series(axes[1], rl, "reward/macro_f1", "train macro F1", smooth)
    plot_series(axes[1], rl, "reward/negative_f1", "train negative F1", smooth)
    plot_series(axes[1], rl, "reward/positive_f1", "train positive F1", smooth)
    plot_series(axes[1], rl, "val/macro_f1_reward", "val macro F1", 1, marker="o")
    plot_series(axes[1], rl, "val/negative_f1_reward", "val negative F1", 1, marker="o")
    plot_series(axes[1], rl, "val/positive_f1_reward", "val positive F1", 1, marker="o")
    finalize_axis(axes[1], "RL Source-Level F1 Rewards", "Reward")

    plot_series(axes[2], rl, "reward/format", "train format", smooth)
    plot_series(axes[2], rl, "reward/think", "train think", smooth)
    plot_series(axes[2], rl, "reward/count", "train count", smooth)
    plot_series(axes[2], rl, "val/format_reward", "val format", 1, marker="o")
    plot_series(axes[2], rl, "val/think_reward", "val think", 1, marker="o")
    plot_series(axes[2], rl, "val/count_reward", "val count", 1, marker="o")
    finalize_axis(axes[2], "RL Auxiliary Rewards", "Reward")

    plot_series(axes[3], rl, "actor/kl_loss", "KL loss", smooth)
    plot_series(axes[3], rl, "actor/ppo_kl", "PPO KL", smooth)
    plot_series(axes[3], rl, "actor/kl_coef", "KL coef", smooth, linestyle="--")
    finalize_axis(axes[3], "RL KL Control", "Value")

    plot_series(axes[4], rl, "actor/pg_loss", "policy loss", smooth)
    plot_series(axes[4], rl, "actor/entropy_loss", "entropy loss", smooth)
    plot_series(axes[4], rl, "actor/grad_norm", "grad norm", smooth)
    finalize_axis(axes[4], "RL Optimization Signals", "Value")

    plot_series(axes[5], rl, "response_length/mean", "response length mean", smooth)
    plot_series(axes[5], rl, "response_length/clip_ratio", "response clip ratio", smooth)
    plot_series(axes[5], rl, "val_response_length/mean", "val response length mean", 1, marker="o")
    plot_series(axes[5], rl, "val_response_length/clip_ratio", "val response clip ratio", 1, marker="o")
    finalize_axis(axes[5], "RL Response Length", "Tokens / Ratio")

    save_figure(fig, output_dir, "rl_training_curves")
    plt.close(fig)


def plot_summary(
    plt: Any,
    sft: Series,
    sft_meta: Dict[str, Any],
    rl: Series,
    rl_meta: Dict[str, Any],
    output_dir: Path,
    smooth: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    plot_series(axes[0], sft, "train/loss", "train loss", smooth)
    plot_series(axes[0], sft, "eval/loss", "eval loss", 1, marker="o")
    if "best_eval_loss_step" in sft_meta:
        axes[0].axvline(sft_meta["best_eval_loss_step"], color="0.35", linestyle="--", linewidth=1)
    finalize_axis(axes[0], "SFT Objective", "Loss")

    plot_series(axes[1], rl, "reward/overall", "train reward", smooth)
    plot_series(axes[1], rl, "val/reward_score", "val reward", 1, marker="o")
    plot_series(axes[1], rl, "val/macro_f1_reward", "val macro F1 reward", 1, marker="o")
    best_step = rl_meta.get("selected_rl_step")
    if best_step is None:
        best_step = rl_meta.get("best_val_reward_step")
    if best_step is not None:
        axes[1].axvline(float(best_step), color="0.35", linestyle="--", linewidth=1)
    finalize_axis(axes[1], "RL Objective", "Reward")

    save_figure(fig, output_dir, "sft_rl_training_summary")
    plt.close(fig)


def write_metadata(output_dir: Path, sft_meta: Dict[str, Any], rl_meta: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_curve_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"sft": sft_meta, "rl": rl_meta}, f, indent=2, ensure_ascii=False)


def write_series_csv(output_dir: Path, name: str, series: Series) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "step", "value"])
        for metric in sorted(series):
            for step, value in dedupe_and_sort(series[metric]):
                writer.writerow([metric, int(step) if float(step).is_integer() else step, value])


def svg_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if abs(vmax - vmin) < 1e-12:
        delta = max(abs(vmax) * 0.1, 1.0)
        vmin -= delta
        vmax += delta
    raw = (vmax - vmin) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(abs(raw))) if raw else 1.0
    candidates = [1, 2, 2.5, 5, 10]
    step = min(candidates, key=lambda c: abs(raw - c * power)) * power
    start = math.floor(vmin / step) * step
    ticks: List[float] = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(0.0 if abs(value) < 1e-12 else value)
        value += step
    return ticks[: max(count + 2, 2)]


def format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value == 0:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def line_points(line: ChartLine) -> List[Point]:
    series, key, _label, smooth = line
    return smooth_points(series.get(key, []), smooth)


def render_svg_chart(
    spec: ChartSpec,
    x0: int,
    y0: int,
    width: int,
    height: int,
    colors: Sequence[str],
) -> List[str]:
    title = spec["title"]
    ylabel = spec.get("ylabel", "")
    xlabel = spec.get("xlabel", "Training step")
    lines: List[ChartLine] = spec["lines"]
    vlines: List[Tuple[float, str]] = spec.get("vlines", [])

    margin_left, margin_right, margin_top, margin_bottom = 60, 18, 36, 48
    plot_x0 = x0 + margin_left
    plot_y0 = y0 + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    plot_x1 = plot_x0 + plot_w
    plot_y1 = plot_y0 + plot_h

    all_points: List[Point] = []
    prepared: List[Tuple[List[Point], str, str]] = []
    for idx, line in enumerate(lines):
        points = line_points(line)
        if not points:
            continue
        prepared.append((points, line[2], colors[idx % len(colors)]))
        all_points.extend(points)

    out: List[str] = []
    out.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18}" text-anchor="middle" class="title">{svg_escape(title)}</text>')
    if not all_points:
        out.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + height / 2:.1f}" text-anchor="middle" class="muted">No data</text>')
        return out

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(xmax - xmin) < 1e-12:
        xmin -= 1
        xmax += 1
    if abs(ymax - ymin) < 1e-12:
        pad = max(abs(ymax) * 0.1, 1.0)
        ymin -= pad
        ymax += pad
    else:
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

    xticks = nice_ticks(xmin, xmax, 5)
    yticks = nice_ticks(ymin, ymax, 5)

    def sx(x: float) -> float:
        return plot_x0 + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return plot_y1 - (y - ymin) / (ymax - ymin) * plot_h

    out.append(f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#d0d7de" stroke-width="1"/>')
    for tick in yticks:
        y = sy(tick)
        out.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{plot_x0 - 8}" y="{y + 3:.1f}" text-anchor="end" class="tick">{format_tick(tick)}</text>')
    for tick in xticks:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" y1="{plot_y0}" x2="{x:.1f}" y2="{plot_y1}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{plot_y1 + 18}" text-anchor="middle" class="tick">{format_tick(tick)}</text>')

    for value, label in vlines:
        if xmin <= value <= xmax:
            x = sx(value)
            out.append(f'<line x1="{x:.1f}" y1="{plot_y0}" x2="{x:.1f}" y2="{plot_y1}" class="vline"/>')
            if label:
                out.append(f'<text x="{x + 4:.1f}" y="{plot_y0 + 12}" class="tick">{svg_escape(label)}</text>')

    for points, label, color in prepared:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.0" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            if len(points) <= 12:
                out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.2" fill="{color}"/>')

    out.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + height - 10}" text-anchor="middle" class="label">{svg_escape(xlabel)}</text>')
    out.append(
        f'<text x="{x0 + 14}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 + 14} {y0 + height / 2:.1f})" text-anchor="middle" class="label">{svg_escape(ylabel)}</text>'
    )

    legend_x = plot_x0 + 8
    legend_y = plot_y0 + 14
    for idx, (_points, label, color) in enumerate(prepared):
        y = legend_y + idx * 15
        out.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 16}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{legend_x + 21}" y="{y + 3}" class="legend">{svg_escape(label)}</text>')

    return out


def write_svg_grid(output_dir: Path, stem: str, charts: List[ChartSpec], cols: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_w, chart_h = 520, 330
    gap_x, gap_y = 26, 24
    rows = math.ceil(len(charts) / cols)
    width = cols * chart_w + (cols - 1) * gap_x
    height = rows * chart_h + (rows - 1) * gap_y
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    body: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#24292f}.title{font-size:15px;font-weight:600}.label{font-size:11px}.tick{font-size:10px;fill:#57606a}.legend{font-size:10px}.muted{font-size:12px;fill:#57606a}.grid{stroke:#d8dee4;stroke-width:0.8;opacity:0.65}.vline{stroke:#57606a;stroke-width:1.2;stroke-dasharray:4 4}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, chart in enumerate(charts):
        row = idx // cols
        col = idx % cols
        x = col * (chart_w + gap_x)
        y = row * (chart_h + gap_y)
        body.extend(render_svg_chart(chart, x, y, chart_w, chart_h, colors))
    body.append("</svg>")
    (output_dir / f"{stem}.svg").write_text("\n".join(body), encoding="utf-8")


def write_svg_figures(
    output_dir: Path,
    sft: Series,
    sft_meta: Dict[str, Any],
    rl: Series,
    rl_meta: Dict[str, Any],
    smooth: int,
) -> None:
    sft_best = sft_meta.get("best_eval_loss_step")
    rl_best = rl_meta.get("selected_rl_step")
    if rl_best is None:
        rl_best = rl_meta.get("best_val_reward_step")

    write_svg_grid(
        output_dir,
        "sft_training_curves",
        [
            {
                "title": "SFT Loss",
                "ylabel": "Loss",
                "lines": [(sft, "train/loss", "train loss", smooth), (sft, "eval/loss", "eval loss", 1)],
                "vlines": [(float(sft_best), "best eval")] if sft_best is not None else [],
            },
            {
                "title": "SFT Token Accuracy",
                "ylabel": "Accuracy",
                "lines": [(sft, "train/token_acc", "train token acc", smooth), (sft, "eval/token_acc", "eval token acc", 1)],
            },
            {
                "title": "SFT Learning Rate",
                "ylabel": "LR",
                "lines": [(sft, "train/learning_rate", "learning rate", smooth)],
            },
            {
                "title": "SFT Grad Norm",
                "ylabel": "Norm",
                "lines": [(sft, "train/grad_norm", "grad norm", smooth)],
            },
        ],
        cols=2,
    )

    write_svg_grid(
        output_dir,
        "rl_training_curves",
        [
            {
                "title": "RL Reward",
                "ylabel": "Reward",
                "lines": [
                    (rl, "reward/overall", "train overall", smooth),
                    (rl, "reward/step", "train step", smooth),
                    (rl, "val/reward_score", "val reward", 1),
                ],
                "vlines": [(float(rl_best), "best")] if rl_best is not None else [],
            },
            {
                "title": "RL Source-Level F1 Rewards",
                "ylabel": "Reward",
                "lines": [
                    (rl, "reward/macro_f1", "train macro F1", smooth),
                    (rl, "reward/negative_f1", "train negative F1", smooth),
                    (rl, "reward/positive_f1", "train positive F1", smooth),
                    (rl, "val/macro_f1_reward", "val macro F1", 1),
                    (rl, "val/negative_f1_reward", "val negative F1", 1),
                    (rl, "val/positive_f1_reward", "val positive F1", 1),
                ],
            },
            {
                "title": "RL Auxiliary Rewards",
                "ylabel": "Reward",
                "lines": [
                    (rl, "reward/format", "train format", smooth),
                    (rl, "reward/think", "train think", smooth),
                    (rl, "reward/count", "train count", smooth),
                    (rl, "val/format_reward", "val format", 1),
                    (rl, "val/think_reward", "val think", 1),
                    (rl, "val/count_reward", "val count", 1),
                ],
            },
            {
                "title": "RL KL Control",
                "ylabel": "Value",
                "lines": [
                    (rl, "actor/kl_loss", "KL loss", smooth),
                    (rl, "actor/ppo_kl", "PPO KL", smooth),
                    (rl, "actor/kl_coef", "KL coef", smooth),
                ],
            },
            {
                "title": "RL Optimization Signals",
                "ylabel": "Value",
                "lines": [
                    (rl, "actor/pg_loss", "policy loss", smooth),
                    (rl, "actor/entropy_loss", "entropy loss", smooth),
                    (rl, "actor/grad_norm", "grad norm", smooth),
                ],
            },
            {
                "title": "RL Response Length",
                "ylabel": "Tokens / Ratio",
                "lines": [
                    (rl, "response_length/mean", "response length mean", smooth),
                    (rl, "response_length/clip_ratio", "clip ratio", smooth),
                    (rl, "val_response_length/mean", "val response length mean", 1),
                    (rl, "val_response_length/clip_ratio", "val clip ratio", 1),
                ],
            },
        ],
        cols=2,
    )

    write_svg_grid(
        output_dir,
        "sft_rl_training_summary",
        [
            {
                "title": "SFT Objective",
                "ylabel": "Loss",
                "lines": [(sft, "train/loss", "train loss", smooth), (sft, "eval/loss", "eval loss", 1)],
                "vlines": [(float(sft_best), "best eval")] if sft_best is not None else [],
            },
            {
                "title": "RL Objective",
                "ylabel": "Reward",
                "lines": [
                    (rl, "reward/overall", "train reward", smooth),
                    (rl, "val/reward_score", "val reward", 1),
                    (rl, "val/macro_f1_reward", "val macro F1 reward", 1),
                ],
                "vlines": [(float(rl_best), "best")] if rl_best is not None else [],
            },
        ],
        cols=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--rl-dir", type=Path, default=DEFAULT_RL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--selected-rl-step",
        type=int,
        default=None,
        help="Step to mark as the selected RL checkpoint. Defaults to inferring it from --rl-dir.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only export parsed CSV/metadata and skip plotting.",
    )
    args = parser.parse_args()

    sft_series, sft_meta = load_sft_series(args.sft_dir)
    rl_series, rl_meta = load_rl_series(args.rl_dir, selected_step=args.selected_rl_step)

    output_dir = args.output_dir.resolve()
    write_metadata(output_dir, sft_meta, rl_meta)
    write_series_csv(output_dir, "sft", sft_series)
    write_series_csv(output_dir, "rl", rl_series)

    if not args.csv_only:
        write_svg_figures(output_dir, sft_series, sft_meta, rl_series, rl_meta, args.smooth_window)
        plt = import_matplotlib(optional=True)
        if plt is not None:
            configure_style(plt)
            plot_sft(plt, sft_series, sft_meta, output_dir, args.smooth_window)
            plot_rl(plt, rl_series, rl_meta, output_dir, args.smooth_window)
            plot_summary(plt, sft_series, sft_meta, rl_series, rl_meta, output_dir, args.smooth_window)
        else:
            print("matplotlib is not installed; wrote SVG figures and CSV files only.")

    print(f"Wrote figures to: {output_dir}")
    print(f"SFT log: {sft_meta.get('log_path')}")
    print(f"RL log: {rl_meta.get('log_path')}")


if __name__ == "__main__":
    main()
