from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from .checkpoint import load_model_from_checkpoint
from .dataset import build_sample, flat_to_board, parse_board_text
from .decode import compose_completed_boards, decode_completed_boards, ITERATIVE_CONFIDENCE_THRESHOLD
from .eval import summarize_board_violations
from .presets import DECODE_PRESETS, apply_decode_preset, get_decode_preset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-puzzle Sudoku inference from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--puzzle", help="Puzzle text using digits and 0 or . for blanks.")
    source_group.add_argument("--file", type=Path, help="Path to a text file containing the puzzle.")
    source_group.add_argument("--stdin", action="store_true", help="Read a pasted puzzle from standard input.")
    parser.add_argument("--decode-preset", choices=sorted(DECODE_PRESETS.keys()), default="production_fast")
    parser.add_argument("--decode-mode", choices=["argmax", "iterative", "solver_guided"], default="iterative")
    parser.add_argument("--iterative-threshold", type=float, default=ITERATIVE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iterative-max-fills-per-round", type=int)
    parser.add_argument("--show-raw-prediction", action="store_true")
    args = parser.parse_args(argv)
    args.decode_mode, args.iterative_threshold, args.iterative_max_fills_per_round = apply_decode_preset(
        args.decode_preset,
        args.decode_mode,
        args.iterative_threshold,
        args.iterative_max_fills_per_round,
    )
    _validate_decode_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    puzzle = load_puzzle(args)
    sample = build_sample(puzzle, puzzle)
    digits = sample["digits"].unsqueeze(0)
    givens = sample["givens"].unsqueeze(0)

    with torch.no_grad():
        logits = model(digits.to(device), givens.to(device))
        raw_predictions = logits.argmax(dim=-1)
        raw_completed = compose_completed_boards(digits, raw_predictions)[0]
        decoded_boards, postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
            model,
            digits,
            givens,
            device,
            mode=args.decode_mode,
            initial_logits=logits,
            iterative_confidence_threshold=args.iterative_threshold,
            iterative_max_fills_per_round=args.iterative_max_fills_per_round,
        )

    decoded_board = flat_to_board(decoded_boards[0].tolist())
    raw_board = flat_to_board(raw_completed.tolist())
    summary = summarize_board_violations(decoded_boards[0].tolist())
    blank_count = sum(value == 0 for row in puzzle for value in row)
    model_config = payload.get("model_config", {})
    checkpoint_config = payload.get("config", {})
    preset = get_decode_preset(args.decode_preset)
    preset_profile = preset.profile if preset is not None else "custom"

    print(
        "checkpoint={checkpoint} model_type={model_type} decode_preset={decode_preset} preset_profile={preset_profile} decode_mode={decode_mode} blanks={blanks} "
        "iterative_threshold={iterative_threshold:.2f} iterative_max_fills_per_round={iterative_max_fills_per_round} "
        "postprocess_change_count={postprocess_change_count} decode_iteration_count={decode_iteration_count}".format(
            checkpoint=args.checkpoint,
            model_type=payload.get("model_type", "unknown"),
            decode_preset=args.decode_preset,
            preset_profile=preset_profile,
            decode_mode=args.decode_mode,
            blanks=blank_count,
            iterative_threshold=args.iterative_threshold,
            iterative_max_fills_per_round=args.iterative_max_fills_per_round,
            postprocess_change_count=postprocess_change_counts[0],
            decode_iteration_count=decode_iteration_counts[0],
        )
    )
    if preset is not None:
        print(f"preset_summary={preset.summary}")
    print("puzzle:")
    print(format_board(puzzle))
    if args.show_raw_prediction or args.decode_mode != "argmax":
        print("raw_prediction:")
        print(format_board(raw_board))
    print("prediction:")
    print(format_board(decoded_board))
    print(
        "is_valid={is_valid} total_conflicts={total_conflicts} row_conflicts={row_conflicts} "
        "col_conflicts={col_conflicts} box_conflicts={box_conflicts}".format(**summary)
    )
    print(f"model_config={model_config}")
    print(f"training_config={checkpoint_config}")


def load_puzzle(args: argparse.Namespace) -> list[list[int]]:
    puzzle_text = args.puzzle
    if args.file is not None:
        puzzle_text = args.file.read_text(encoding="utf-8")
    if args.stdin:
        puzzle_text = sys.stdin.read()

    if puzzle_text is None:
        raise ValueError("One of --puzzle, --file, or --stdin must be provided.")

    return parse_board_text(puzzle_text)


def format_board(board: list[list[int]]) -> str:
    lines = []
    for row_index, row in enumerate(board):
        chunks = [
            " ".join("." if value == 0 else str(value) for value in row[start : start + 3])
            for start in range(0, 9, 3)
        ]
        lines.append(" | ".join(chunks))
        if row_index in {2, 5}:
            lines.append("-" * 21)
    return "\n".join(lines)


def _validate_decode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.iterative_threshold <= 1.0:
        parser.error("--iterative-threshold must be between 0.0 and 1.0.")
    if args.iterative_max_fills_per_round is not None and args.iterative_max_fills_per_round < 1:
        parser.error("--iterative-max-fills-per-round must be at least 1.")


if __name__ == "__main__":
    main()
