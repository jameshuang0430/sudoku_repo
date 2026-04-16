from __future__ import annotations

from copy import deepcopy
import random
from typing import Callable, Optional

from .validator import Board, validate_board

CandidateRanker = Callable[[int, int, list[int]], list[int]]


def solve_board(board: Board) -> list[list[int]]:
    """Return a solved copy of the board or raise if the puzzle is invalid."""
    working_board = [list(row) for row in board]
    validate_board(working_board)

    if not _solve_in_place(working_board):
        raise ValueError("The puzzle has no valid solution.")

    return working_board


def solve_board_with_scores(
    board: Board,
    candidate_scores: list[list[float]],
) -> list[list[int]]:
    """Solve a puzzle while ordering blank-cell candidates by model scores."""
    if len(candidate_scores) != 81 or any(len(scores) != 9 for scores in candidate_scores):
        raise ValueError("candidate_scores must contain 81 score vectors with 9 entries each.")

    working_board = [list(row) for row in board]
    validate_board(working_board)

    def rank_candidates(row: int, col: int, candidates: list[int]) -> list[int]:
        scores = candidate_scores[row * 9 + col]
        return sorted(candidates, key=lambda value: (-scores[value - 1], value))

    if not _solve_in_place(working_board, candidate_ranker=rank_candidates):
        raise ValueError("The puzzle has no valid solution.")

    return working_board


def count_solutions(board: Board, limit: int = 2) -> int:
    """Count solutions up to a limit. Use limit=2 to test uniqueness."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    working_board = deepcopy([list(row) for row in board])
    validate_board(working_board)
    return _count_in_place(working_board, limit)


def _solve_in_place(
    board: list[list[int]],
    rng: Optional[random.Random] = None,
    candidate_ranker: Optional[CandidateRanker] = None,
) -> bool:
    next_cell = _find_most_constrained_empty(board)
    if next_cell is None:
        return True

    row, col, candidates = next_cell
    if not candidates:
        return False

    if candidate_ranker is not None:
        candidates = _apply_candidate_ranker(candidate_ranker, row, col, candidates)
    elif rng is not None:
        rng.shuffle(candidates)

    for candidate in candidates:
        board[row][col] = candidate
        if _solve_in_place(board, rng=rng, candidate_ranker=candidate_ranker):
            return True
        board[row][col] = 0

    return False


def _count_in_place(board: list[list[int]], limit: int) -> int:
    next_cell = _find_most_constrained_empty(board)
    if next_cell is None:
        return 1

    row, col, candidates = next_cell
    if not candidates:
        return 0

    total = 0
    for candidate in candidates:
        board[row][col] = candidate
        total += _count_in_place(board, limit)
        board[row][col] = 0
        if total >= limit:
            return total

    return total


def _find_most_constrained_empty(
    board: list[list[int]],
) -> Optional[tuple[int, int, list[int]]]:
    best: Optional[tuple[int, int, list[int]]] = None
    best_count = 10

    for row in range(9):
        for col in range(9):
            if board[row][col] != 0:
                continue
            candidates = _get_candidates(board, row, col)
            if len(candidates) < best_count:
                best = (row, col, candidates)
                best_count = len(candidates)
                if best_count == 1:
                    return best

    return best


def _get_candidates(board: list[list[int]], row: int, col: int) -> list[int]:
    used = set(board[row])
    used.update(board[row_index][col] for row_index in range(9))

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for current_row in range(box_row, box_row + 3):
        for current_col in range(box_col, box_col + 3):
            used.add(board[current_row][current_col])

    return [value for value in range(1, 10) if value not in used]


def _apply_candidate_ranker(
    candidate_ranker: CandidateRanker,
    row: int,
    col: int,
    candidates: list[int],
) -> list[int]:
    ranked = candidate_ranker(row, col, list(candidates))
    ordered: list[int] = []
    seen: set[int] = set()

    for candidate in ranked:
        if candidate in candidates and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)

    return ordered
