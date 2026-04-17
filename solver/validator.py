from __future__ import annotations

from typing import Sequence

Board = Sequence[Sequence[int]]


def validate_board(board: Board) -> None:
    if len(board) != 9:
        raise ValueError("A Sudoku board must have exactly 9 rows.")

    for row_index, row in enumerate(board):
        if len(row) != 9:
            raise ValueError(f"Row {row_index} must contain exactly 9 values.")
        for value in row:
            if not isinstance(value, int):
                raise ValueError("Board values must be integers.")
            if value < 0 or value > 9:
                raise ValueError("Board values must be between 0 and 9.")

    for row_index in range(9):
        _validate_unit(board[row_index], f"row {row_index}")

    for column_index in range(9):
        column = [board[row_index][column_index] for row_index in range(9)]
        _validate_unit(column, f"column {column_index}")

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = [
                board[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            ]
            _validate_unit(box, f"box ({box_row // 3}, {box_col // 3})")


def board_to_string(board: Board) -> str:
    validate_board(board)
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


def _validate_unit(values: Sequence[int], label: str) -> None:
    seen = set()
    for value in values:
        if value == 0:
            continue
        if value in seen:
            raise ValueError(f"Duplicate value {value} found in {label}.")
        seen.add(value)
