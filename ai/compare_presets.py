from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .benchmark import benchmark_model
from .checkpoint import load_model_from_checkpoint
from .dataset import SudokuDataset, SudokuFileDataset
from .eval import evaluate_model
from .presets import DECODE_PRESETS, DecodePreset
from .run_metadata import build_run_metadata


DEFAULT_COMPARE_PRESETS = ["production_fast", "production_pure", "research_raw"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple decode presets for one Sudoku checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset file to compare on.")
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=sorted(DECODE_PRESETS.keys()),
        default=DEFAULT_COMPARE_PRESETS,
    )
    parser.add_argument("--benchmark-max-samples", type=int, default=128)
    parser.add_argument("--benchmark-warmup-batches", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    dataset = load_compare_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    presets = [DECODE_PRESETS[name] for name in args.presets]

    comparisons = compare_presets(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
        presets=presets,
        batch_size=args.batch_size,
        benchmark_warmup_batches=args.benchmark_warmup_batches,
        benchmark_repeats=args.benchmark_repeats,
    )

    training_config = payload.get("config", {})
    compare_config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "dataset_size": len(dataset),
        "requested_dataset_size": args.dataset_size,
        "blanks": args.blanks,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "presets": args.presets,
        "benchmark_max_samples": min(args.benchmark_max_samples, len(dataset)),
        "benchmark_warmup_batches": args.benchmark_warmup_batches,
        "benchmark_repeats": args.benchmark_repeats,
    }
    run_metadata = build_run_metadata(
        command_name="ai.compare_presets",
        argv=argv,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        model_type=payload.get("model_type"),
        extra={
            "report_path": args.report,
            "device": device.type,
            "presets": args.presets,
            "batch_size": args.batch_size,
        },
    )

    for comparison in comparisons:
        metrics = comparison["metrics"]
        latency = comparison["latency"]
        print(
            "preset={preset} profile={profile} decode_mode={decode_mode} blank_cell_acc={blank_cell_accuracy:.4f} "
            "board_solved_rate={board_solved_rate:.4f} valid_board_rate={valid_board_rate:.4f} "
            "mean_total_conflicts={mean_total_conflicts:.2f} mean_board_ms={mean_board_duration_ms:.2f} "
            "throughput={throughput_boards_per_second:.2f} boards_per_sec".format(
                preset=comparison["preset"],
                profile=comparison["preset_profile"],
                decode_mode=comparison["decode_mode"],
                blank_cell_accuracy=metrics["blank_cell_accuracy"],
                board_solved_rate=metrics["board_solved_rate"],
                valid_board_rate=metrics["valid_board_rate"],
                mean_total_conflicts=metrics["mean_total_conflicts"],
                mean_board_duration_ms=latency["mean_board_duration_ms"],
                throughput_boards_per_second=latency["throughput_boards_per_second"],
            )
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_metadata": run_metadata,
                    "compare_config": compare_config,
                    "training_config": training_config,
                    "comparisons": comparisons,
                },
                handle,
                indent=2,
            )
        print(f"saved_report={args.report}")


def load_compare_dataset(args: argparse.Namespace) -> Dataset[Any]:
    if args.dataset is not None:
        dataset: Dataset[Any] = SudokuFileDataset(args.dataset)
    else:
        dataset = SudokuDataset(size=args.dataset_size, blanks=args.blanks, seed=args.seed)

    sample_count = min(args.benchmark_max_samples, len(dataset))
    if sample_count < len(dataset):
        return Subset(dataset, range(sample_count))
    return dataset


def compare_presets(
    *,
    model: torch.nn.Module,
    dataset: Dataset[Any],
    dataloader: DataLoader[Any],
    device: torch.device,
    presets: Sequence[DecodePreset],
    batch_size: int,
    benchmark_warmup_batches: int,
    benchmark_repeats: int,
) -> list[dict[str, Any]]:
    benchmark_results = benchmark_model(
        model=model,
        dataset=dataset,
        device=device,
        batch_sizes=[batch_size],
        decode_presets=presets,
        warmup_batches=benchmark_warmup_batches,
        repeats=benchmark_repeats,
    )
    latency_by_preset = {result["preset"]: result for result in benchmark_results}

    comparisons: list[dict[str, Any]] = []
    for preset in presets:
        metrics = evaluate_model(
            model,
            dataloader,
            device,
            decode_mode=preset.decode_mode,
            iterative_threshold=preset.iterative_threshold,
            iterative_max_fills_per_round=preset.iterative_max_fills_per_round,
        )
        comparisons.append(
            {
                "preset": preset.name,
                "preset_profile": preset.profile,
                "preset_summary": preset.summary,
                "decode_mode": preset.decode_mode,
                "iterative_threshold": preset.iterative_threshold,
                "iterative_max_fills_per_round": preset.iterative_max_fills_per_round,
                "metrics": metrics,
                "latency": latency_by_preset[preset.name],
            }
        )

    return comparisons


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.benchmark_max_samples < 1:
        parser.error("--benchmark-max-samples must be at least 1.")
    if args.benchmark_warmup_batches < 0:
        parser.error("--benchmark-warmup-batches cannot be negative.")
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be at least 1.")


if __name__ == "__main__":
    main()
