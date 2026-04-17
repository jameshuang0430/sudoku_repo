from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import SudokuDataset, SudokuFileDataset
from .eval import evaluate_model
from .model import create_model
from .run_metadata import build_run_metadata


UNIT_INDICES = torch.tensor(
    [
        [row * 9 + col for col in range(9)]
        for row in range(9)
    ]
    + [
        [row * 9 + col for row in range(9)]
        for col in range(9)
    ]
    + [
        [
            row * 9 + col
            for row in range(box_row, box_row + 3)
            for col in range(box_col, box_col + 3)
        ]
        for box_row in range(0, 9, 3)
        for box_col in range(0, 9, 3)
    ],
    dtype=torch.long,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline Sudoku model.")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--dataset", type=Path, help="Optional training JSONL dataset file exported ahead of training.")
    parser.add_argument("--val-dataset", type=Path, help="Optional validation JSONL dataset file for fixed evaluation splits.")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-embed-dim", type=int, default=16)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--transformer-embed-dim", type=int, default=128)
    parser.add_argument("--transformer-num-heads", type=int, default=8)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--constraint-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ai/checkpoints/sudoku_mlp.pt"),
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        help="Optional path for the best validation checkpoint. Defaults to the checkpoint path with a .best.pt suffix.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file for per-epoch metrics. Defaults to the checkpoint path with a .metrics.json suffix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_type, model_config = build_model_config(args)
    model = create_model(model_type, **model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_history: list[dict[str, float]] = []

    checkpoint_config = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "resolved_train_size": len(train_dataset),
        "resolved_val_size": len(val_dataset),
        "blanks": args.blanks,
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "val_dataset": str(args.val_dataset) if args.val_dataset is not None else None,
        "model_type": model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "constraint_loss_weight": args.constraint_loss_weight,
    }

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = resolve_best_checkpoint_path(args)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_epoch_record: dict[str, float] | None = None
    best_epoch_index: int | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            constraint_loss_weight=args.constraint_loss_weight,
        )
        metrics = evaluate_model(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            **train_stats,
            **metrics,
        }
        epoch_history.append(epoch_record)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} train_ce_loss={train_ce_loss:.4f} "
            "train_constraint_loss={train_constraint_loss:.4f} blank_cell_acc={blank_cell_accuracy:.4f} "
            "board_solved_rate={board_solved_rate:.4f} valid_board_rate={valid_board_rate:.4f}".format(
                **epoch_record,
            )
        )

        if is_better_epoch(epoch_record, best_epoch_record):
            best_epoch_record = dict(epoch_record)
            best_epoch_index = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                model_type=model_type,
                checkpoint_config={
                    **checkpoint_config,
                    "best_epoch": best_epoch_index,
                    "stopped_early": False,
                },
            )
            print(f"saved_best_checkpoint={best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True
                print(
                    f"early_stopping_triggered=1 epoch={epoch} best_epoch={best_epoch_index} "
                    f"patience={args.early_stopping_patience}"
                )
                break

    final_epoch = epoch_history[-1]["epoch"] if epoch_history else 0
    checkpoint_config = {
        **checkpoint_config,
        "best_epoch": best_epoch_index,
        "stopped_early": stopped_early,
        "final_epoch": final_epoch,
        "best_checkpoint": str(best_checkpoint_path),
    }

    save_checkpoint(
        path=args.checkpoint,
        model=model,
        model_type=model_type,
        checkpoint_config=checkpoint_config,
    )
    print(f"saved_checkpoint={args.checkpoint}")

    metrics_output = resolve_metrics_output_path(args)
    run_metadata = build_run_metadata(
        command_name="ai.train",
        argv=argv,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        model_type=model_type,
        extra={
            "val_dataset_path": args.val_dataset,
            "best_checkpoint_path": best_checkpoint_path,
            "metrics_output_path": metrics_output,
        },
    )
    write_metrics_report(
        output_path=metrics_output,
        config=checkpoint_config,
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        epoch_history=epoch_history,
        best_epoch_record=best_epoch_record,
        run_metadata=run_metadata,
    )
    print(f"saved_metrics={metrics_output}")


def build_model_config(args: argparse.Namespace) -> tuple[str, dict[str, int | float]]:
    if args.model == "mlp":
        return (
            "mlp",
            {
                "embed_dim": args.mlp_embed_dim,
                "hidden_dim": args.mlp_hidden_dim,
                "depth": args.mlp_depth,
                "dropout": args.dropout,
            },
        )

    return (
        "transformer",
        {
            "embed_dim": args.transformer_embed_dim,
            "num_heads": args.transformer_num_heads,
            "depth": args.transformer_depth,
            "ff_dim": args.transformer_ff_dim,
            "dropout": args.dropout,
        },
    )


def resolve_best_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.best_checkpoint is not None:
        return args.best_checkpoint
    checkpoint_name = args.checkpoint.name
    checkpoint_stem = checkpoint_name[:-3] if checkpoint_name.endswith(".pt") else args.checkpoint.stem
    return args.checkpoint.with_name(f"{checkpoint_stem}.best.pt")


def resolve_metrics_output_path(args: argparse.Namespace) -> Path:
    if args.metrics_output is not None:
        return args.metrics_output
    checkpoint_name = args.checkpoint.name
    checkpoint_stem = checkpoint_name[:-3] if checkpoint_name.endswith(".pt") else args.checkpoint.stem
    return args.checkpoint.with_name(f"{checkpoint_stem}.metrics.json")


def write_metrics_report(
    output_path: Path,
    config: dict[str, object],
    train_dataset_size: int,
    val_dataset_size: int,
    epoch_history: list[dict[str, float]],
    best_epoch_record: dict[str, float] | None,
    run_metadata: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_metrics = epoch_history[-1] if epoch_history else None
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_metadata": run_metadata,
                "config": config,
                "train_dataset_size": train_dataset_size,
                "val_dataset_size": val_dataset_size,
                "epochs": epoch_history,
                "final_metrics": final_metrics,
                "best_epoch_metrics": best_epoch_record,
            },
            handle,
            indent=2,
        )


def build_datasets(args: argparse.Namespace) -> tuple[Dataset[Any], Dataset[Any]]:
    if args.val_dataset is not None and args.dataset is None:
        raise ValueError("--val-dataset requires --dataset.")

    if args.dataset is None:
        return (
            SudokuDataset(size=args.train_size, blanks=args.blanks, seed=args.seed),
            SudokuDataset(size=args.val_size, blanks=args.blanks, seed=args.seed + 10_000),
        )

    if args.val_dataset is not None:
        return SudokuFileDataset(args.dataset), SudokuFileDataset(args.val_dataset)

    dataset = SudokuFileDataset(args.dataset)
    return split_loaded_dataset(dataset, val_size=args.val_size, seed=args.seed)


def split_loaded_dataset(
    dataset: Dataset[Any],
    val_size: int,
    seed: int,
) -> tuple[Subset[Any], Subset[Any]]:
    if len(dataset) < 2:
        raise ValueError("Dataset files must contain at least 2 records for train/validation splitting.")
    if val_size < 1 or val_size >= len(dataset):
        raise ValueError("When using --dataset, val-size must be between 1 and len(dataset) - 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    constraint_loss_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        digits = batch["digits"].to(device)
        givens = batch["givens"].to(device)
        targets = batch["targets"].to(device)
        blank_mask = digits == 0

        optimizer.zero_grad()
        logits = model(digits, givens)
        ce_loss = masked_cross_entropy(logits, targets, blank_mask)
        constraint_loss = logits.new_tensor(0.0)
        if constraint_loss_weight > 0.0:
            constraint_loss = constraint_consistency_penalty(logits, digits)
        loss = ce_loss + constraint_loss_weight * constraint_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_constraint_loss += constraint_loss.item()
        total_batches += 1

    return {
        "train_loss": total_loss / total_batches,
        "train_ce_loss": total_ce_loss / total_batches,
        "train_constraint_loss": total_constraint_loss / total_batches,
    }


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, 9),
        targets.reshape(-1),
        reduction="none",
    )
    active_losses = losses[blank_mask.reshape(-1)]
    if active_losses.numel() == 0:
        raise ValueError("blank_mask must include at least one trainable cell.")
    return active_losses.mean()


def constraint_consistency_penalty(logits: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    given_mask = digits != 0
    if given_mask.any():
        given_classes = digits.clamp(min=1) - 1
        given_one_hot = F.one_hot(given_classes, num_classes=9).to(dtype=probabilities.dtype)
        full_probabilities = torch.where(given_mask.unsqueeze(-1), given_one_hot, probabilities)
    else:
        full_probabilities = probabilities

    unit_probabilities = full_probabilities[:, UNIT_INDICES.to(logits.device), :]
    digit_totals = unit_probabilities.sum(dim=2)
    return (digit_totals - 1.0).pow(2).mean()


def is_better_epoch(candidate: dict[str, float], current_best: dict[str, float] | None) -> bool:
    if current_best is None:
        return True
    candidate_score = (
        candidate["board_solved_rate"],
        -candidate["mean_total_conflicts"],
        candidate["blank_cell_accuracy"],
    )
    current_score = (
        current_best["board_solved_rate"],
        -current_best["mean_total_conflicts"],
        current_best["blank_cell_accuracy"],
    )
    return candidate_score > current_score


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_type: str,
    checkpoint_config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "model_config": model.get_config(),
            "config": checkpoint_config,
        },
        path,
    )


if __name__ == "__main__":
    main()




