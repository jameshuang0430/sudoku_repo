from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from solver import generate_puzzle

Sample = dict[str, torch.Tensor]
IGNORED_BOARD_CHARACTERS = {" ", "\t", "\r", "\n", "|", "-", "+"}


class SudokuDataset(Dataset[Sample]):
    def __init__(self, size: int, blanks: int = 40, seed: Optional[int] = None) -> None:
        if size < 1:
            raise ValueError("size must be at least 1.")

        self._samples: list[Sample] = []
        base_seed = 0 if seed is None else seed

        for index in range(size):
            puzzle, solution = generate_puzzle(blanks=blanks, seed=base_seed + index)
            self._samples.append(build_sample(puzzle, solution))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]


class SudokuFileDataset(Dataset[Sample]):
    def __init__(self, path: Path | str) -> None:
        dataset_path = Path(path)
        self._samples: list[Sample] = []

        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                record = json.loads(stripped)
                puzzle, solution = load_record_boards(record, dataset_path.parent)
                self._samples.append(build_sample(puzzle, solution))

        if not self._samples:
            raise ValueError("Dataset file must contain at least one record.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]


def build_sample(puzzle: list[list[int]], solution: list[list[int]]) -> Sample:
    digits = torch.tensor(flatten_board(puzzle), dtype=torch.long)
    givens = (digits != 0).to(dtype=torch.float32)
    targets = torch.tensor([value - 1 for value in flatten_board(solution)], dtype=torch.long)
    return {
        "digits": digits,
        "givens": givens,
        "targets": targets,
    }


def flatten_board(board: list[list[int]]) -> list[int]:
    return [value for row in board for value in row]


def flat_to_board(values: list[int]) -> list[list[int]]:
    if len(values) != 81:
        raise ValueError("A flattened Sudoku board must contain exactly 81 values.")
    return [values[index : index + 9] for index in range(0, 81, 9)]


def sample_to_record(sample: Sample) -> dict[str, object]:
    puzzle = flat_to_board(sample["digits"].tolist())
    solution = flat_to_board((sample["targets"] + 1).tolist())
    return {
        "puzzle": puzzle,
        "solution": solution,
        "blank_count": sum(value == 0 for row in puzzle for value in row),
    }


def load_record_boards(
    record: dict[str, object],
    base_dir: Path,
) -> tuple[list[list[int]], list[list[int]]]:
    if "puzzle" in record and "solution" in record:
        return normalize_board(record["puzzle"], field_name="puzzle"), normalize_board(
            record["solution"],
            field_name="solution",
        )

    if "puzzle_path" in record and "solution_path" in record:
        puzzle_path = resolve_dataset_path(record["puzzle_path"], base_dir)
        solution_path = resolve_dataset_path(record["solution_path"], base_dir)
        return (
            parse_board_text(puzzle_path.read_text(encoding="utf-8")),
            parse_board_text(solution_path.read_text(encoding="utf-8")),
        )

    raise ValueError("Dataset record must contain puzzle/solution boards or puzzle_path/solution_path fields.")


def normalize_board(raw_board: object, field_name: str) -> list[list[int]]:
    if not isinstance(raw_board, list) or len(raw_board) != 9:
        raise ValueError(f"{field_name} must be a 9x9 board.")

    board: list[list[int]] = []
    for row in raw_board:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError(f"{field_name} must be a 9x9 board.")
        normalized_row: list[int] = []
        for value in row:
            if not isinstance(value, int):
                raise ValueError(f"{field_name} must only contain integers.")
            normalized_row.append(value)
        board.append(normalized_row)

    return board


def resolve_dataset_path(raw_path: object, base_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Dataset path fields must be non-empty strings.")

    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def parse_board_text(text: str) -> list[list[int]]:
    values: list[int] = []
    for character in text:
        if character in IGNORED_BOARD_CHARACTERS:
            continue
        if character == ".":
            values.append(0)
            continue
        if character not in "0123456789":
            raise ValueError("Board text may only contain digits, dots, whitespace, or board separators like | and -.")
        values.append(int(character))

    if len(values) != 81:
        raise ValueError("Board text must contain exactly 81 cells.")

    return flat_to_board(values)
