#!/usr/bin/env python
"""Legacy/oracle entrypoint for no-thinking VPB evaluation.

This preserves the previous no-thinking behavior where the VPB reference
answer is included in the model prompt by default.
Use api_eval_no_think_stepwise.py with --use-reference-answer 0 for the
official no-leakage ablation protocol.
"""

import sys
from pathlib import Path

from api_eval_no_think_stepwise import main


def has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if not has_arg("--use-reference-answer"):
    sys.argv.extend(["--use-reference-answer", "1"])

if not has_arg("--output"):
    sys.argv.extend(
        [
            "--output",
            str(Path(__file__).resolve().parent / "outputs" / "no_think_single_pass_predictions.jsonl"),
        ]
    )

main()
