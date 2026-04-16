from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import load_model_from_checkpoint
from .dataset import SudokuDataset, SudokuFileDataset
from .decode import (
    DecodeMode,
    ITERATIVE_CONFIDENCE_THRESHOLD,
    decode_completed_boards,
)


@dataclass(frozen=True)
class DecodePreset:
    name: str
    decode_mode: DecodeMode
    iterative_threshold: float = ITERATIVE_CONFIDENCE_THRESHOLD
    iterative_max_fills_per_round: int | None = None


DECODE_PRESETS: dict[str, DecodePreset] = {
    "argmax": DecodePreset(name="argmax", decode_mode="argmax"),
    "iterative": DecodePreset(name="iterative", decode_mode="iterative"),
    "iterative_strict": DecodePreset(
        name="iterative_strict",
        decode_mode="iterative",
        iterative_threshold=0.75,
        iterative_max_fills_per_round=2,
    ),
    "solver_guided": DecodePreset(name="solver_guided", decode_mode="solver_guided"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sudoku checkpoint inference latency.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset file to benchmark.")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument(
        "--decode-presets",
        nargs="+",
        choices=sorted(DECODE_PRESETS.keys()),
        default=["argmax", "iterative", "iterative_strict", "solver_guided"],
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    dataset = load_benchmark_dataset(args)
    results = benchmark_model(
        model=model,
        dataset=dataset,
        device=device,
        batch_sizes=args.batch_sizes,
        decode_presets=[DECODE_PRESETS[name] for name in args.decode_presets],
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
    )

    training_config = payload.get("config", {})
    benchmark_config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "dataset_size": args.dataset_size,
        "blanks": args.blanks,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "batch_sizes": args.batch_sizes,
        "decode_presets": args.decode_presets,
    }

    for result in results:
        print(
            "preset={preset} decode_mode={decode_mode} batch_size={batch_size} samples={sample_count} "
            "mean_total_ms={mean_total_duration_ms:.2f} mean_board_ms={mean_board_duration_ms:.2f} "
            "throughput={throughput_boards_per_second:.2f} boards_per_sec".format(**result)
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "benchmark_config": benchmark_config,
                    "training_config": training_config,
                    "results": results,
                },
                handle,
                indent=2,
            )
        print(f"saved_report={args.report}")


def load_benchmark_dataset(args: argparse.Namespace) -> Dataset[Any]:
    if args.dataset is not None:
        dataset: Dataset[Any] = SudokuFileDataset(args.dataset)
    else:
        dataset = SudokuDataset(size=args.dataset_size, blanks=args.blanks, seed=args.seed)

    sample_count = min(args.max_samples, len(dataset))
    return Subset(dataset, range(sample_count))


def benchmark_model(
    model: torch.nn.Module,
    dataset: Dataset[Any],
    device: torch.device,
    batch_sizes: Sequence[int],
    decode_presets: Sequence[DecodePreset],
    warmup_batches: int,
    repeats: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model.eval()

    with torch.inference_mode():
        for preset in decode_presets:
            for batch_size in batch_sizes:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                _run_warmup(model, dataloader, device, preset, warmup_batches)
                repeat_summaries = [
                    _run_single_benchmark_pass(model, dataloader, device, preset)
                    for _ in range(repeats)
                ]
                sample_count = repeat_summaries[0]["sample_count"] if repeat_summaries else 0
                batch_count = repeat_summaries[0]["batch_count"] if repeat_summaries else 0
                total_duration_ms_values = [summary["total_duration_ms"] for summary in repeat_summaries]
                board_duration_ms_values = [summary["board_duration_ms"] for summary in repeat_summaries]
                batch_duration_ms_values = [summary["batch_duration_ms"] for summary in repeat_summaries]
                throughput_values = [summary["throughput_boards_per_second"] for summary in repeat_summaries]
                results.append(
                    {
                        "preset": preset.name,
                        "decode_mode": preset.decode_mode,
                        "iterative_threshold": preset.iterative_threshold,
                        "iterative_max_fills_per_round": preset.iterative_max_fills_per_round,
                        "batch_size": batch_size,
                        "sample_count": sample_count,
                        "batch_count": batch_count,
                        "repeat_count": repeats,
                        "mean_total_duration_ms": mean(total_duration_ms_values),
                        "min_total_duration_ms": min(total_duration_ms_values),
                        "max_total_duration_ms": max(total_duration_ms_values),
                        "mean_board_duration_ms": mean(board_duration_ms_values),
                        "mean_batch_duration_ms": mean(batch_duration_ms_values),
                        "throughput_boards_per_second": mean(throughput_values),
                    }
                )

    return results


def _run_warmup(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    preset: DecodePreset,
    warmup_batches: int,
) -> None:
    if warmup_batches < 1:
        return

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= warmup_batches:
            break
        _run_decode_batch(model, batch, device, preset)


def _run_single_benchmark_pass(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    preset: DecodePreset,
) -> dict[str, float | int]:
    total_samples = 0
    total_batches = 0
    total_duration_ms = 0.0

    for batch in dataloader:
        _synchronize(device)
        batch_start = time.perf_counter()
        batch_size = _run_decode_batch(model, batch, device, preset)
        _synchronize(device)
        total_duration_ms += (time.perf_counter() - batch_start) * 1000.0
        total_samples += batch_size
        total_batches += 1

    board_duration_ms = total_duration_ms / total_samples if total_samples else 0.0
    batch_duration_ms = total_duration_ms / total_batches if total_batches else 0.0
    throughput = (total_samples * 1000.0 / total_duration_ms) if total_duration_ms else 0.0
    return {
        "sample_count": total_samples,
        "batch_count": total_batches,
        "total_duration_ms": total_duration_ms,
        "board_duration_ms": board_duration_ms,
        "batch_duration_ms": batch_duration_ms,
        "throughput_boards_per_second": throughput,
    }


def _run_decode_batch(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    preset: DecodePreset,
) -> int:
    digits = batch["digits"].to(device)
    givens = batch["givens"].to(device)
    logits = model(digits, givens)
    decode_completed_boards(
        model,
        digits,
        givens,
        device,
        mode=preset.decode_mode,
        initial_logits=logits,
        iterative_confidence_threshold=preset.iterative_threshold,
        iterative_max_fills_per_round=preset.iterative_max_fills_per_round,
    )
    return int(digits.size(0))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1.")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes values must all be at least 1.")


if __name__ == "__main__":
    main()
