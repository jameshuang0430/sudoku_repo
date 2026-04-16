from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from solver import validate_board

from .checkpoint import load_model_from_checkpoint
from .dataset import SudokuDataset, SudokuFileDataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Sudoku model checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset file to evaluate.")
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    dataset = load_evaluation_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_model(model, dataloader, device)
    training_config = payload.get("config", {})
    evaluation_source = str(args.dataset) if args.dataset is not None else "generated"
    print(
        "checkpoint={checkpoint} eval_source={eval_source} train_seed={train_seed} "
        "blank_cell_acc={blank_cell_accuracy:.4f} board_solved_rate={board_solved_rate:.4f} "
        "valid_board_rate={valid_board_rate:.4f} mean_mismatch_count={mean_mismatch_count:.2f} "
        "mean_total_conflicts={mean_total_conflicts:.2f}".format(
            checkpoint=args.checkpoint,
            eval_source=evaluation_source,
            train_seed=training_config.get("seed", "unknown"),
            **metrics,
        )
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": metrics,
                    "training_config": training_config,
                    "evaluation_config": {
                        "checkpoint": str(args.checkpoint),
                        "dataset": str(args.dataset) if args.dataset is not None else None,
                        "dataset_size": args.dataset_size,
                        "blanks": args.blanks,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                    },
                },
                handle,
                indent=2,
            )
        print(f"saved_report={args.report}")


def load_evaluation_dataset(args: argparse.Namespace) -> Dataset[Any]:
    if args.dataset is not None:
        return SudokuFileDataset(args.dataset)
    return SudokuDataset(size=args.dataset_size, blanks=args.blanks, seed=args.seed)


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader[Any], device: torch.device) -> dict[str, float]:
    model.eval()
    total_blank_cells = 0
    correct_blank_cells = 0
    solved_boards = 0
    valid_boards = 0
    total_boards = 0
    total_mismatch_count = 0
    total_row_conflicts = 0
    total_col_conflicts = 0
    total_box_conflicts = 0
    total_constraint_conflicts = 0

    with torch.no_grad():
        for batch in dataloader:
            digits = batch["digits"].to(device)
            givens = batch["givens"].to(device)
            targets = batch["targets"].to(device)

            logits = model(digits, givens)
            predictions = logits.argmax(dim=-1)
            blank_mask = digits == 0

            correct_blank_cells += ((predictions == targets) & blank_mask).sum().item()
            total_blank_cells += blank_mask.sum().item()

            completed_boards = compose_completed_boards(digits, predictions)
            target_boards = targets.cpu() + 1

            for completed_board, target_board in zip(completed_boards, target_boards):
                total_boards += 1
                predicted_values = completed_board.tolist()
                target_values = target_board.tolist()
                mismatch_count = sum(
                    1
                    for predicted_value, target_value in zip(predicted_values, target_values)
                    if predicted_value != target_value
                )
                total_mismatch_count += mismatch_count

                board_summary = summarize_board_violations(predicted_values)
                total_row_conflicts += board_summary["row_conflicts"]
                total_col_conflicts += board_summary["col_conflicts"]
                total_box_conflicts += board_summary["box_conflicts"]
                total_constraint_conflicts += board_summary["total_conflicts"]

                if torch.equal(completed_board, target_board):
                    solved_boards += 1
                if board_summary["is_valid"]:
                    valid_boards += 1

    blank_cell_accuracy = 0.0
    if total_blank_cells:
        blank_cell_accuracy = correct_blank_cells / total_blank_cells

    board_solved_rate = solved_boards / total_boards if total_boards else 0.0
    valid_board_rate = valid_boards / total_boards if total_boards else 0.0

    return {
        "blank_cell_accuracy": blank_cell_accuracy,
        "board_solved_rate": board_solved_rate,
        "valid_board_rate": valid_board_rate,
        "mean_mismatch_count": total_mismatch_count / total_boards if total_boards else 0.0,
        "mean_row_conflicts": total_row_conflicts / total_boards if total_boards else 0.0,
        "mean_col_conflicts": total_col_conflicts / total_boards if total_boards else 0.0,
        "mean_box_conflicts": total_box_conflicts / total_boards if total_boards else 0.0,
        "mean_total_conflicts": total_constraint_conflicts / total_boards if total_boards else 0.0,
    }


def compose_completed_boards(digits: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.where(digits.cpu() == 0, predictions.cpu() + 1, digits.cpu())


def summarize_board_violations(flat_board: list[int]) -> dict[str, int | bool]:
    board = [flat_board[index : index + 9] for index in range(0, 81, 9)]
    row_conflicts = sum(count_unit_conflicts(row) for row in board)
    col_conflicts = sum(count_unit_conflicts([board[row][col] for row in range(9)]) for col in range(9))
    box_conflicts = 0
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            values = [board[row][col] for row in range(box_row, box_row + 3) for col in range(box_col, box_col + 3)]
            box_conflicts += count_unit_conflicts(values)

    total_conflicts = row_conflicts + col_conflicts + box_conflicts
    is_valid = _is_valid_completed_board(flat_board)
    return {
        "is_valid": is_valid,
        "row_conflicts": row_conflicts,
        "col_conflicts": col_conflicts,
        "box_conflicts": box_conflicts,
        "total_conflicts": total_conflicts,
    }


def count_unit_conflicts(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        if value == 0:
            continue
        counts[value] = counts.get(value, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _is_valid_completed_board(flat_board: list[int]) -> bool:
    board = [flat_board[index : index + 9] for index in range(0, 81, 9)]
    try:
        validate_board(board)
    except ValueError:
        return False
    return all(value != 0 for row in board for value in row)


if __name__ == "__main__":
    main()
