from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_model_from_checkpoint
from .dataset import SudokuDataset, SudokuFileDataset, flat_to_board
from .decode import compose_completed_boards, decode_completed_boards
from .eval import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Sudoku model prediction errors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset file to analyze.")
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decode-mode", choices=["argmax", "iterative", "solver_guided"], default="argmax")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    dataset = load_analysis_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_model(model, dataloader, device, decode_mode=args.decode_mode)
    error_cases = collect_error_cases(model, dataloader, device, args.limit, decode_mode=args.decode_mode)

    training_config = payload.get("config", {})
    evaluation_source = str(args.dataset) if args.dataset is not None else "generated"
    print(
        "checkpoint={checkpoint} eval_source={eval_source} decode_mode={decode_mode} train_seed={train_seed} "
        "blank_cell_acc={blank_cell_accuracy:.4f} board_solved_rate={board_solved_rate:.4f} "
        "valid_board_rate={valid_board_rate:.4f} mean_decode_iteration_count={mean_decode_iteration_count:.2f}".format(
            checkpoint=args.checkpoint,
            eval_source=evaluation_source,
            decode_mode=args.decode_mode,
            train_seed=training_config.get("seed", "unknown"),
            **metrics,
        )
    )

    if not error_cases:
        print("No error cases found in the evaluated sample.")
    else:
        for case in error_cases:
            print(
                f"case_index={case['index']} mismatch_count={case['mismatch_count']} "
                f"postprocess_change_count={case['postprocess_change_count']} "
                f"decode_iteration_count={case['decode_iteration_count']}"
            )
            print("puzzle:")
            print(format_board(case["puzzle"]))
            if case["postprocess_change_count"] > 0:
                print("raw argmax prediction:")
                print(format_board(case["raw_prediction"]))
            print("decoded prediction:")
            print(format_board(case["prediction"]))
            print("solution:")
            print(format_board(case["solution"]))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": metrics,
                    "training_config": training_config,
                    "analysis_config": {
                        "checkpoint": str(args.checkpoint),
                        "dataset": str(args.dataset) if args.dataset is not None else None,
                        "dataset_size": args.dataset_size,
                        "blanks": args.blanks,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                        "decode_mode": args.decode_mode,
                        "limit": args.limit,
                    },
                    "error_cases": error_cases,
                },
                handle,
                indent=2,
            )
        print(f"saved_report={args.report}")


def load_analysis_dataset(args: argparse.Namespace) -> Dataset[Any]:
    if args.dataset is not None:
        return SudokuFileDataset(args.dataset)
    return SudokuDataset(size=args.dataset_size, blanks=args.blanks, seed=args.seed)


def collect_error_cases(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    limit: int,
    decode_mode: str = "argmax",
) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    cases: list[dict[str, object]] = []
    sample_index = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            digits = batch["digits"].to(device)
            givens = batch["givens"].to(device)
            targets = batch["targets"].to(device)
            logits = model(digits, givens)
            raw_predictions = logits.argmax(dim=-1)
            raw_completed_boards = compose_completed_boards(digits, raw_predictions)
            completed_boards, postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
                model,
                digits,
                givens,
                device,
                mode=decode_mode,
                initial_logits=logits,
            )
            target_boards = targets.cpu() + 1

            for original_digits, raw_board, predicted_board, target_board, postprocess_change_count, decode_iteration_count in zip(
                batch["digits"],
                raw_completed_boards,
                completed_boards,
                target_boards,
                postprocess_change_counts,
                decode_iteration_counts,
            ):
                if torch.equal(predicted_board, target_board):
                    sample_index += 1
                    continue

                mismatch_positions = [
                    index
                    for index, (predicted_value, target_value) in enumerate(
                        zip(predicted_board.tolist(), target_board.tolist())
                    )
                    if predicted_value != target_value
                ]
                cases.append(
                    {
                        "index": sample_index,
                        "mismatch_count": len(mismatch_positions),
                        "mismatch_positions": mismatch_positions,
                        "postprocess_change_count": postprocess_change_count,
                        "decode_iteration_count": decode_iteration_count,
                        "puzzle": flat_to_board(original_digits.tolist()),
                        "raw_prediction": flat_to_board(raw_board.tolist()),
                        "prediction": flat_to_board(predicted_board.tolist()),
                        "solution": flat_to_board(target_board.tolist()),
                    }
                )
                sample_index += 1
                if len(cases) >= limit:
                    return cases

            if len(cases) >= limit:
                break

    return cases


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


if __name__ == "__main__":
    main()
