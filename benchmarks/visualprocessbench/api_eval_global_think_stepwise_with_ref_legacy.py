#!/usr/bin/env python
"""Legacy/oracle entrypoint for global-thinking stepwise VPB evaluation.

This preserves the previous behavior of api_eval_global_think_stepwise.py:
the VPB reference answer is included in the model prompts by default.
Use the no-ref main script for the official evaluation protocol.
"""

import sys
from pathlib import Path

from api_eval_global_think_stepwise import main


def has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if not has_arg("--use-reference-answer"):
    sys.argv.extend(["--use-reference-answer", "1"])

if not has_arg("--output"):
    sys.argv.extend(
        [
            "--output",
            str(Path(__file__).resolve().parent / "outputs" / "global_think_stepwise_predictions.jsonl"),
        ]
    )

main()
