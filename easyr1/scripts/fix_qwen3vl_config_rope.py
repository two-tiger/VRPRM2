#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix Qwen3-VL merged config RoPE fields for Transformers.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--reference-config", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("model_type") != "qwen3_vl":
        print(f"Skip non-Qwen3-VL config: {config_path}")
        return

    text_cfg = cfg.setdefault("text_config", {})
    changed = False

    rope_parameters = text_cfg.get("rope_parameters") or {}
    if text_cfg.get("rope_scaling") is None:
        if rope_parameters:
            text_cfg["rope_scaling"] = {
                key: value for key, value in rope_parameters.items() if key in {"mrope_interleaved", "mrope_section", "rope_type"}
            }
            changed = True
        elif args.reference_config and args.reference_config.is_file():
            ref = json.loads(args.reference_config.read_text(encoding="utf-8"))
            ref_text = ref.get("text_config") or {}
            if ref_text.get("rope_scaling") is not None:
                text_cfg["rope_scaling"] = ref_text["rope_scaling"]
                changed = True

    if text_cfg.get("rope_theta") is None:
        if "rope_theta" in rope_parameters:
            text_cfg["rope_theta"] = rope_parameters["rope_theta"]
            changed = True
        elif args.reference_config and args.reference_config.is_file():
            ref = json.loads(args.reference_config.read_text(encoding="utf-8"))
            ref_text = ref.get("text_config") or {}
            if ref_text.get("rope_theta") is not None:
                text_cfg["rope_theta"] = ref_text["rope_theta"]
                changed = True

    if changed:
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Fixed Qwen3-VL RoPE fields in: {config_path}")
    else:
        print(f"Qwen3-VL RoPE fields already OK: {config_path}")


if __name__ == "__main__":
    main()
