from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import (
    board_to_string,
    count_solutions,
    export_puzzle_dataset,
    export_puzzle_dataset_splits,
    generate_puzzle,
    solve_board,
)

IGNORED_PUZZLE_CHARACTERS = {" ", "\t", "\r", "\n", "|", "-", "+"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sudoku command-line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a Sudoku puzzle.")
    generate_parser.add_argument("--blanks", type=int, default=40)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--output", type=Path, help="Write the generated puzzle to a file.")
    generate_parser.add_argument(
        "--solution-output",
        type=Path,
        help="Write the generated solution to a file.",
    )
    generate_parser.add_argument(
        "--show-solution",
        action="store_true",
        help="Print the solved board after the generated puzzle.",
    )
    generate_parser.add_argument(
        "--skip-unique-check",
        action="store_true",
        help="Skip the unique-solution check during generation.",
    )

    export_parser = subparsers.add_parser(
        "export-dataset",
        help="Generate many Sudoku puzzle files for training data.",
    )
    export_parser.add_argument("--size", type=int, help="Number of records to export in single-manifest mode.")
    export_parser.add_argument("--train-size", type=int, default=0)
    export_parser.add_argument("--val-size", type=int, default=0)
    export_parser.add_argument("--test-size", type=int, default=0)
    export_parser.add_argument("--blanks", type=int, default=40)
    export_parser.add_argument("--seed", type=int, default=7)
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/puzzles"),
        help="Directory that will receive puzzle and solution text files.",
    )
    export_parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSONL manifest with generated file paths and seeds.",
    )
    export_parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory for train/val/test JSONL manifests in split-export mode.",
    )
    export_parser.add_argument(
        "--skip-unique-check",
        action="store_true",
        help="Skip the unique-solution check during generation.",
    )

    solve_parser = subparsers.add_parser("solve", help="Solve a Sudoku puzzle.")
    source_group = solve_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--puzzle",
        help="Puzzle text using digits and 0 or . for blanks. Pretty-printed 9-line boards are supported.",
    )
    source_group.add_argument("--file", type=Path, help="Path to a text file containing the puzzle.")
    source_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read a pasted puzzle from standard input.",
    )
    solve_parser.add_argument(
        "--check-unique",
        action="store_true",
        help="Also report whether the puzzle has a unique solution.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _handle_generate(args)
    if args.command == "export-dataset":
        return _handle_export_dataset(args)
    if args.command == "solve":
        return _handle_solve(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


def parse_puzzle_text(text: str) -> list[list[int]]:
    values: list[int] = []
    for character in text:
        if character in IGNORED_PUZZLE_CHARACTERS:
            continue
        if character == ".":
            values.append(0)
            continue
        if character not in "0123456789":
            raise ValueError("Puzzle input may only contain digits, dots, whitespace, or board separators like | and -.")
        values.append(int(character))

    if len(values) != 81:
        raise ValueError("Puzzle input must contain exactly 81 cells.")

    return [values[index : index + 9] for index in range(0, 81, 9)]


def _handle_generate(args: argparse.Namespace) -> int:
    puzzle, solution = generate_puzzle(
        blanks=args.blanks,
        ensure_unique=not args.skip_unique_check,
        seed=args.seed,
    )
    puzzle_text = board_to_string(puzzle)
    solution_text = board_to_string(solution)

    if args.output is not None:
        _write_text(args.output, puzzle_text)
        print(f"Saved puzzle to {args.output}")
    else:
        print("Puzzle:")
        print(puzzle_text)

    if args.solution_output is not None:
        _write_text(args.solution_output, solution_text)
        print(f"Saved solution to {args.solution_output}")

    if args.show_solution:
        print("Solution:")
        print(solution_text)

    return 0


def _handle_export_dataset(args: argparse.Namespace) -> int:
    split_mode = any(size > 0 for size in (args.train_size, args.val_size, args.test_size))
    if split_mode:
        if args.manifest is not None:
            raise ValueError("Use --manifest-dir instead of --manifest when exporting train/val/test splits.")

        split_records = export_puzzle_dataset_splits(
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            blanks=args.blanks,
            ensure_unique=not args.skip_unique_check,
            seed=args.seed,
        )
        manifest_root = args.output_dir if args.manifest_dir is None else args.manifest_dir
        for split_name, records in split_records.items():
            print(f"Exported {len(records)} puzzle files to {args.output_dir / split_name}")
            print(f"Saved manifest to {Path(manifest_root) / (split_name + '.jsonl')}")
        return 0

    size = 128 if args.size is None else args.size
    records = export_puzzle_dataset(
        output_dir=args.output_dir,
        size=size,
        blanks=args.blanks,
        ensure_unique=not args.skip_unique_check,
        seed=args.seed,
    )

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record))
                handle.write("\n")
        print(f"Saved manifest to {args.manifest}")

    print(f"Exported {len(records)} puzzle files to {args.output_dir}")
    return 0


def _handle_solve(args: argparse.Namespace) -> int:
    puzzle_text = args.puzzle
    if args.file is not None:
        puzzle_text = args.file.read_text(encoding="utf-8")
    if args.stdin:
        puzzle_text = sys.stdin.read()

    if puzzle_text is None:
        raise ValueError("One of --puzzle, --file, or --stdin must be provided.")

    puzzle = parse_puzzle_text(puzzle_text)
    solution = solve_board(puzzle)

    print("Puzzle:")
    print(board_to_string(puzzle))
    print("Solution:")
    print(board_to_string(solution))

    if args.check_unique:
        solution_count = count_solutions(puzzle, limit=2)
        unique_label = "yes" if solution_count == 1 else "no"
        print(f"Unique solution: {unique_label}")

    return 0


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{content}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
