from __future__ import annotations

from typing import Literal

import torch

from solver import solve_board_with_scores

from .dataset import flat_to_board, flatten_board

DecodeMode = Literal["argmax", "solver_guided"]


def compose_completed_boards(digits: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.where(digits.cpu() == 0, predictions.cpu() + 1, digits.cpu())


def decode_completed_boards(
    digits: torch.Tensor,
    logits: torch.Tensor,
    mode: DecodeMode = "argmax",
) -> tuple[torch.Tensor, list[int]]:
    predictions = logits.argmax(dim=-1).cpu()
    if mode == "argmax":
        return compose_completed_boards(digits, predictions), [0 for _ in range(predictions.size(0))]
    if mode != "solver_guided":
        raise ValueError(f"Unsupported decode mode: {mode}")

    digits_cpu = digits.cpu()
    logits_cpu = logits.detach().cpu()
    completed_boards: list[torch.Tensor] = []
    postprocess_change_counts: list[int] = []

    for original_digits, raw_predictions, sample_logits in zip(digits_cpu, predictions, logits_cpu):
        solved_board = solve_board_with_scores(
            flat_to_board(original_digits.tolist()),
            sample_logits.tolist(),
        )
        solved_flat = torch.tensor(flatten_board(solved_board), dtype=torch.long)
        blank_mask = original_digits == 0
        change_count = int(((solved_flat != (raw_predictions + 1)) & blank_mask).sum().item())
        completed_boards.append(solved_flat)
        postprocess_change_counts.append(change_count)

    return torch.stack(completed_boards), postprocess_change_counts
