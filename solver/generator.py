from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .solver import _solve_in_place, count_solutions
from .validator import board_to_string, validate_board


Record = dict[str, object]


def generate_solved_board(seed: Optional[int] = None) -> list[list[int]]:
    """Generate a complete valid Sudoku solution."""
    rng = random.Random(seed)
    board = [[0 for _ in range(9)] for _ in range(9)]

    if not _solve_in_place(board, rng=rng):
        raise RuntimeError("Failed to generate a solved Sudoku board.")

    return board


def generate_puzzle(
    blanks: int = 40,
    ensure_unique: bool = True,
    seed: Optional[int] = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate a puzzle and its solution.

    blanks controls how many cells are removed from the solved board.
    """
    if blanks < 0 or blanks > 81:
        raise ValueError("blanks must be between 0 and 81.")

    rng = random.Random(seed)
    solution = generate_solved_board(seed=seed)
    puzzle = [row[:] for row in solution]

    positions = [(row, col) for row in range(9) for col in range(9)]
    rng.shuffle(positions)

    removed = 0
    for row, col in positions:
        if removed == blanks:
            break

        previous = puzzle[row][col]
        puzzle[row][col] = 0

        if ensure_unique and count_solutions(puzzle, limit=2) != 1:
            puzzle[row][col] = previous
            continue

        removed += 1

    if removed != blanks:
        raise ValueError(
            f"Could not generate a puzzle with {blanks} blanks under the current constraints."
        )

    validate_board(puzzle)
    return puzzle, solution


def export_puzzle_dataset(
    output_dir: Path | str,
    size: int,
    blanks: int = 40,
    ensure_unique: bool = True,
    seed: Optional[int] = None,
) -> list[Record]:
    """Generate a reproducible set of puzzle and solution text files."""
    if size < 1:
        raise ValueError("size must be at least 1.")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    base_seed = 0 if seed is None else seed
    records: list[Record] = []

    for index in range(size):
        sample_seed = base_seed + index
        puzzle, solution = generate_puzzle(
            blanks=blanks,
            ensure_unique=ensure_unique,
            seed=sample_seed,
        )

        stem = f"puzzle_{index + 1:05d}"
        puzzle_path = directory / f"{stem}.txt"
        solution_path = directory / f"{stem}_solution.txt"

        puzzle_path.write_text(f"{board_to_string(puzzle)}\n", encoding="utf-8")
        solution_path.write_text(f"{board_to_string(solution)}\n", encoding="utf-8")

        records.append(
            {
                "index": index,
                "seed": sample_seed,
                "blank_count": blanks,
                "puzzle": [row[:] for row in puzzle],
                "solution": [row[:] for row in solution],
                "puzzle_path": str(puzzle_path),
                "solution_path": str(solution_path),
            }
        )

    return records


def export_puzzle_dataset_splits(
    output_dir: Path | str,
    manifest_dir: Path | str | None = None,
    train_size: int = 0,
    val_size: int = 0,
    test_size: int = 0,
    blanks: int = 40,
    ensure_unique: bool = True,
    seed: Optional[int] = None,
) -> dict[str, list[Record]]:
    """Generate fixed train/val/test dataset splits and JSONL manifests."""
    split_sizes = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    active_splits = {name: size for name, size in split_sizes.items() if size > 0}
    if not active_splits:
        raise ValueError("At least one of train_size, val_size, or test_size must be greater than 0.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_root = output_root if manifest_dir is None else Path(manifest_dir)
    manifest_root.mkdir(parents=True, exist_ok=True)

    base_seed = 0 if seed is None else seed
    seed_offset = 0
    split_records: dict[str, list[Record]] = {}

    for split_name, split_size in active_splits.items():
        records = export_puzzle_dataset(
            output_dir=output_root / split_name,
            size=split_size,
            blanks=blanks,
            ensure_unique=ensure_unique,
            seed=base_seed + seed_offset,
        )
        seed_offset += split_size

        records_with_split: list[Record] = []
        for record in records:
            split_record = dict(record)
            split_record["split"] = split_name
            records_with_split.append(split_record)

        manifest_path = manifest_root / f"{split_name}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records_with_split:
                handle.write(json.dumps(record))
                handle.write("\n")

        split_records[split_name] = records_with_split

    return split_records
