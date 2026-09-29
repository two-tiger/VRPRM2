import argparse
import json
import shutil
from pathlib import Path


TOKENIZER_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
]


def copy_if_exists(src_dir, dst_dir, name):
    src = src_dir / name
    if not src.exists():
        return False
    shutil.copy2(src, dst_dir / name)
    return True


def sanitize_tokenizer_config(path):
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    extra = data.get("extra_special_tokens")
    if isinstance(extra, list):
        data.setdefault("additional_special_tokens", extra)
        data.pop("extra_special_tokens", None)
    elif extra is not None and not isinstance(extra, dict):
        data.pop("extra_special_tokens", None)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Fix merged Qwen3-VL tokenizer files for vLLM serving.")
    parser.add_argument("--merged-model-dir", required=True)
    parser.add_argument("--base-model", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    merged_dir = Path(args.merged_model_dir).expanduser().resolve()
    base_dir = Path(args.base_model).expanduser().resolve()

    if not merged_dir.is_dir():
        raise FileNotFoundError(f"merged model directory not found: {merged_dir}")
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model directory not found: {base_dir}")

    copied = [name for name in TOKENIZER_FILES if copy_if_exists(base_dir, merged_dir, name)]
    sanitize_tokenizer_config(merged_dir / "tokenizer_config.json")

    print(f"fixed tokenizer files in: {merged_dir}")
    print(f"copied from base model: {', '.join(copied) if copied else 'none'}")


if __name__ == "__main__":
    main()
