from __future__ import annotations

from typing import Literal

import torch

from solver import solve_board_with_scores

from .dataset import flat_to_board, flatten_board

DecodeMode = Literal["argmax", "iterative", "solver_guided"]
ITERATIVE_CONFIDENCE_THRESHOLD = 0.5
ITERATIVE_MAX_FILLS_PER_ROUND: int | None = None


def compose_completed_boards(digits: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.where(digits.cpu() == 0, predictions.cpu() + 1, digits.cpu())


def decode_completed_boards(
    model: torch.nn.Module,
    digits: torch.Tensor,
    givens: torch.Tensor,
    device: torch.device,
    mode: DecodeMode = "argmax",
    initial_logits: torch.Tensor | None = None,
    iterative_confidence_threshold: float = ITERATIVE_CONFIDENCE_THRESHOLD,
    iterative_max_fills_per_round: int | None = ITERATIVE_MAX_FILLS_PER_ROUND,
) -> tuple[torch.Tensor, list[int], list[int]]:
    if initial_logits is None:
        initial_logits = model(digits.to(device), givens.to(device))

    predictions = initial_logits.argmax(dim=-1).cpu()
    raw_completed_boards = compose_completed_boards(digits, predictions)

    if mode == "argmax":
        return raw_completed_boards, [0 for _ in range(predictions.size(0))], [0 for _ in range(predictions.size(0))]

    if mode == "iterative":
        completed_boards: list[torch.Tensor] = []
        postprocess_change_counts: list[int] = []
        decode_iteration_counts: list[int] = []

        for original_digits, original_givens, raw_predictions, sample_logits in zip(
            digits.cpu(),
            givens.cpu(),
            predictions,
            initial_logits.detach().cpu(),
        ):
            completed_board, iteration_count = _decode_board_iteratively(
                model=model,
                digits=original_digits,
                givens=original_givens,
                device=device,
                initial_logits=sample_logits,
                confidence_threshold=iterative_confidence_threshold,
                max_fills_per_round=iterative_max_fills_per_round,
            )
            blank_mask = original_digits == 0
            change_count = int(((completed_board != (raw_predictions + 1)) & blank_mask).sum().item())
            completed_boards.append(completed_board)
            postprocess_change_counts.append(change_count)
            decode_iteration_counts.append(iteration_count)

        return torch.stack(completed_boards), postprocess_change_counts, decode_iteration_counts

    if mode != "solver_guided":
        raise ValueError(f"Unsupported decode mode: {mode}")

    digits_cpu = digits.cpu()
    logits_cpu = initial_logits.detach().cpu()
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

    return torch.stack(completed_boards), postprocess_change_counts, [0 for _ in range(predictions.size(0))]


def _decode_board_iteratively(
    model: torch.nn.Module,
    digits: torch.Tensor,
    givens: torch.Tensor,
    device: torch.device,
    initial_logits: torch.Tensor,
    confidence_threshold: float,
    max_fills_per_round: int | None,
) -> tuple[torch.Tensor, int]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
    if max_fills_per_round is not None and max_fills_per_round < 1:
        raise ValueError("max_fills_per_round must be at least 1 when provided.")

    current_values = digits.tolist()
    current_givens = givens.tolist()
    next_logits = initial_logits.detach().cpu()
    iteration_count = 0

    while 0 in current_values:
        if next_logits is None:
            current_digits_tensor = torch.tensor([current_values], dtype=torch.long, device=device)
            current_givens_tensor = torch.tensor([current_givens], dtype=torch.float32, device=device)
            step_logits = model(current_digits_tensor, current_givens_tensor)[0].detach().cpu()
        else:
            step_logits = next_logits
            next_logits = None

        probabilities = torch.softmax(step_logits, dim=-1)
        ranked_predictions = _rank_blank_predictions(current_values, probabilities)
        candidate_values = current_values.copy()
        confident_predictions: list[tuple[float, int, int]] = []
        for confidence, index, predicted_value in ranked_predictions:
            if confidence < confidence_threshold:
                continue
            if not _is_value_consistent(candidate_values, index, predicted_value):
                continue
            confident_predictions.append((confidence, index, predicted_value))
            candidate_values[index] = predicted_value
            if max_fills_per_round is not None and len(confident_predictions) >= max_fills_per_round:
                break

        placed_any = bool(confident_predictions)
        for _confidence, index, predicted_value in confident_predictions:
            current_values[index] = predicted_value
            current_givens[index] = 1.0

        if not placed_any:
            fallback = next(
                (
                    (confidence, index, predicted_value)
                    for confidence, index, predicted_value in ranked_predictions
                    if _is_value_consistent(current_values, index, predicted_value)
                ),
                None,
            )
            if fallback is None:
                fallback = ranked_predictions[0]
            _, index, predicted_value = fallback
            current_values[index] = predicted_value
            current_givens[index] = 1.0

        iteration_count += 1

    return torch.tensor(current_values, dtype=torch.long), iteration_count


def _rank_blank_predictions(
    current_values: list[int],
    probabilities: torch.Tensor,
) -> list[tuple[float, int, int]]:
    ranked: list[tuple[float, int, int]] = []
    for index, current_value in enumerate(current_values):
        if current_value != 0:
            continue
        predicted_class = int(probabilities[index].argmax().item())
        confidence = float(probabilities[index, predicted_class].item())
        ranked.append((confidence, index, predicted_class + 1))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _is_value_consistent(current_values: list[int], index: int, candidate: int) -> bool:
    row = index // 9
    col = index % 9

    for current_col in range(9):
        current_index = row * 9 + current_col
        if current_index != index and current_values[current_index] == candidate:
            return False

    for current_row in range(9):
        current_index = current_row * 9 + col
        if current_index != index and current_values[current_index] == candidate:
            return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for current_row in range(box_row, box_row + 3):
        for current_col in range(box_col, box_col + 3):
            current_index = current_row * 9 + current_col
            if current_index != index and current_values[current_index] == candidate:
                return False

    return True

