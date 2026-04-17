from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .infer import main as infer_main

DEFAULT_PRODUCTION_CHECKPOINT = Path(r"ai\checkpoints\transformer_large_current.best.pt")
PRESET_ALIASES = {
    "fast": "production_fast",
    "pure": "production_pure",
}


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the current production Sudoku inference path without typing the checkpoint path."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_PRODUCTION_CHECKPOINT,
        help="Override the default production checkpoint path.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_ALIASES.keys()),
        default="fast",
        help="Choose the product-facing wrapper preset.",
    )
    args, infer_args = parser.parse_known_args(argv)
    if not args.checkpoint.exists():
        parser.error(f"--checkpoint not found: {args.checkpoint}")
    return parser, args, infer_args


def build_infer_argv(args: argparse.Namespace, infer_args: Sequence[str]) -> list[str]:
    forwarded = ["--checkpoint", str(args.checkpoint)]
    if not any(option.startswith("--decode-preset") for option in infer_args):
        forwarded.extend(["--decode-preset", PRESET_ALIASES[args.preset]])
    forwarded.extend(infer_args)
    return forwarded


def main(argv: Sequence[str] | None = None) -> None:
    _parser, args, infer_args = parse_args(argv)
    infer_main(build_infer_argv(args, infer_args))


if __name__ == "__main__":
    main()
