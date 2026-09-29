import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_JSONL = (
    DEFAULT_REPO_ROOT
    / "github_repo"
    / "VRPRM"
    / "vrprm"
    / "evaluation"
    / "VisualProcessBench_PRM"
    / "VisualProcessBench"
    / "test.jsonl"
)
DEFAULT_SOURCE_IMAGES = Path(
    os.environ.get("VRPRM_VPB_ASSETS", str(DEFAULT_REPO_ROOT / "assets" / "VisualProcessBench-assets" ))
) / "images"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "VisualProcessBench"


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_images_link(source_images, output_images):
    source_images = Path(source_images).resolve()
    output_images = Path(output_images)
    output_images.parent.mkdir(parents=True, exist_ok=True)

    if output_images.exists() or output_images.is_symlink():
        if output_images.is_symlink() and output_images.resolve() == source_images:
            return
        raise FileExistsError(f"Refusing to replace existing images path: {output_images}")

    os.symlink(source_images, output_images, target_is_directory=True)


def summarize(rows, output_dir):
    source_counts = Counter(row.get("data_source", "unknown") for row in rows)
    policy_counts = Counter(row.get("policy_model", "unknown") for row in rows)
    label_counts = Counter()
    steps_by_source = defaultdict(int)
    missing_images = []

    for idx, row in enumerate(rows):
        labels = row.get("response", {}).get("process_correctness", [])
        label_counts.update(labels)
        steps_by_source[row.get("data_source", "unknown")] += len(labels)
        for rel_image in row.get("image", []):
            image_path = Path(output_dir) / rel_image
            if not image_path.exists():
                missing_images.append({"index": idx, "image": rel_image})

    return {
        "total_samples": len(rows),
        "total_steps": sum(label_counts.values()),
        "label_counts": {
            "correct_1": label_counts.get(1, 0),
            "neutral_0": label_counts.get(0, 0),
            "incorrect_-1": label_counts.get(-1, 0),
        },
        "samples_by_source": dict(sorted(source_counts.items())),
        "steps_by_source": dict(sorted(steps_by_source.items())),
        "samples_by_policy_model": dict(sorted(policy_counts.items())),
        "missing_images_count": len(missing_images),
        "missing_images_preview": missing_images[:20],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local VisualProcessBench files.")
    parser.add_argument("--source-jsonl", default=str(DEFAULT_SOURCE_JSONL))
    parser.add_argument("--source-images", default=str(DEFAULT_SOURCE_IMAGES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    source_jsonl = Path(args.source_jsonl)
    source_images = Path(args.source_images)
    output_dir = Path(args.output_dir)

    if not source_jsonl.exists():
        raise FileNotFoundError(source_jsonl)
    if not source_images.exists():
        raise FileNotFoundError(source_images)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_jsonl, output_dir / "test.jsonl")
    make_images_link(source_images, output_dir / "images")

    rows = load_jsonl(output_dir / "test.jsonl")
    stats = summarize(rows, output_dir)
    write_json(stats, output_dir / "stats.json")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
