from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import load_model_from_checkpoint
from .compare_presets import DEFAULT_COMPARE_PRESETS, compare_presets
from .dataset import SudokuDataset, SudokuFileDataset
from .presets import DECODE_PRESETS
from .run_metadata import build_run_metadata


DEFAULT_TESTS_COMMAND = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard Sudoku release check workflow.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional JSONL dataset file to check.")
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
    parser.add_argument("--tests-command", nargs=argparse.REMAINDER)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--min-production-fast-solved-rate", type=float, default=0.95)
    parser.add_argument("--min-production-pure-solved-rate", type=float, default=0.90)
    parser.add_argument("--min-research-raw-blank-cell-accuracy", type=float, default=0.0)
    parser.add_argument("--max-production-fast-board-ms", type=float, default=5.0)
    parser.add_argument("--max-production-pure-board-ms", type=float, default=50.0)
    parser.add_argument("--max-production-fast-solved-rate-drop", type=float)
    parser.add_argument("--max-production-pure-solved-rate-drop", type=float)
    parser.add_argument("--max-production-fast-board-ms-increase", type=float)
    parser.add_argument("--max-production-pure-board-ms-increase", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.tests_command = list(args.tests_command) if args.tests_command else list(DEFAULT_TESTS_COMMAND)
    _validate_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(args.checkpoint, device)
    dataset = load_release_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    comparisons = compare_presets(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
        presets=[DECODE_PRESETS[name] for name in args.presets],
        batch_size=args.batch_size,
        benchmark_warmup_batches=args.benchmark_warmup_batches,
        benchmark_repeats=args.benchmark_repeats,
    )
    baseline_summary = load_baseline_summary(args.baseline_report, comparisons)
    tests_result = run_tests(args.tests_command, skip=args.skip_tests)
    gates = evaluate_release_gates(comparisons, args)
    gates.extend(evaluate_baseline_gates(baseline_summary, args))
    overall_passed = tests_result["passed"] and all(gate["passed"] for gate in gates)

    release_config = {
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
        "tests_command": args.tests_command,
        "skip_tests": args.skip_tests,
        "baseline_report": str(args.baseline_report) if args.baseline_report is not None else None,
        "thresholds": {
            "min_production_fast_solved_rate": args.min_production_fast_solved_rate,
            "min_production_pure_solved_rate": args.min_production_pure_solved_rate,
            "min_research_raw_blank_cell_accuracy": args.min_research_raw_blank_cell_accuracy,
            "max_production_fast_board_ms": args.max_production_fast_board_ms,
            "max_production_pure_board_ms": args.max_production_pure_board_ms,
            "max_production_fast_solved_rate_drop": args.max_production_fast_solved_rate_drop,
            "max_production_pure_solved_rate_drop": args.max_production_pure_solved_rate_drop,
            "max_production_fast_board_ms_increase": args.max_production_fast_board_ms_increase,
            "max_production_pure_board_ms_increase": args.max_production_pure_board_ms_increase,
        },
    }
    run_metadata = build_run_metadata(
        command_name="ai.release_check",
        argv=argv,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        model_type=payload.get("model_type"),
        extra={
            "report_path": args.report,
            "baseline_report_path": args.baseline_report,
            "device": device.type,
            "presets": args.presets,
            "batch_size": args.batch_size,
        },
    )

    print(f"tests_passed={tests_result['passed']} tests_returncode={tests_result['returncode']}")
    for gate in gates:
        print(
            "gate={name} passed={passed} comparator={comparator} observed={observed:.4f} threshold={threshold:.4f}".format(
                name=gate["name"],
                passed=gate["passed"],
                comparator=gate["comparator"],
                observed=gate["observed"],
                threshold=gate["threshold"],
            )
        )
    if baseline_summary is not None:
        for preset_summary in baseline_summary["comparisons"]:
            solved_rate_delta = preset_summary["metrics"]["board_solved_rate"]["delta"]
            conflict_delta = preset_summary["metrics"]["mean_total_conflicts"]["delta"]
            latency_delta = preset_summary["latency"]["mean_board_duration_ms"]["delta"]
            print(
                "baseline_delta preset={preset} solved_rate_delta={solved_rate_delta:.4f} "
                "mean_total_conflicts_delta={conflict_delta:.4f} mean_board_ms_delta={latency_delta:.4f}".format(
                    preset=preset_summary["preset"],
                    solved_rate_delta=solved_rate_delta,
                    conflict_delta=conflict_delta,
                    latency_delta=latency_delta,
                )
            )
    print(f"release_check_passed={overall_passed}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_metadata": run_metadata,
                    "release_config": release_config,
                    "training_config": payload.get("config", {}),
                    "tests": tests_result,
                    "comparisons": comparisons,
                    "baseline": baseline_summary,
                    "gates": gates,
                    "passed": overall_passed,
                },
                handle,
                indent=2,
            )
        print(f"saved_report={args.report}")

    if not overall_passed:
        raise SystemExit(1)


def load_release_dataset(args: argparse.Namespace) -> Dataset[Any]:
    if args.dataset is not None:
        dataset: Dataset[Any] = SudokuFileDataset(args.dataset)
    else:
        dataset = SudokuDataset(size=args.dataset_size, blanks=args.blanks, seed=args.seed)

    sample_count = min(args.benchmark_max_samples, len(dataset))
    if sample_count < len(dataset):
        return Subset(dataset, range(sample_count))
    return dataset


def run_tests(command: Sequence[str], *, skip: bool) -> dict[str, Any]:
    if skip:
        return {
            "command": list(command),
            "returncode": 0,
            "passed": True,
            "stdout": "",
            "stderr": "",
            "skipped": True,
        }

    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "skipped": False,
    }


def load_baseline_summary(
    baseline_report_path: Path | None,
    comparisons: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if baseline_report_path is None:
        return None

    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_comparisons = baseline_report.get("comparisons")
    if not isinstance(baseline_comparisons, list):
        raise ValueError("Baseline report must contain a top-level 'comparisons' list.")

    baseline_by_preset = {
        comparison.get("preset"): comparison
        for comparison in baseline_comparisons
        if isinstance(comparison, dict) and comparison.get("preset") is not None
    }
    missing_presets: list[str] = []
    matched_summaries: list[dict[str, Any]] = []
    for comparison in comparisons:
        preset_name = comparison["preset"]
        baseline_comparison = baseline_by_preset.get(preset_name)
        if baseline_comparison is None:
            missing_presets.append(preset_name)
            continue
        matched_summaries.append(build_baseline_delta_summary(comparison, baseline_comparison))

    baseline_run_metadata = baseline_report.get("run_metadata")
    return {
        "path": str(baseline_report_path),
        "run_metadata": baseline_run_metadata,
        "comparisons": matched_summaries,
        "missing_presets": missing_presets,
    }


def build_baseline_delta_summary(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    metric_names = [
        "blank_cell_accuracy",
        "board_solved_rate",
        "valid_board_rate",
        "mean_total_conflicts",
        "mean_postprocess_change_count",
        "mean_decode_iteration_count",
    ]
    latency_names = [
        "mean_board_duration_ms",
        "throughput_boards_per_second",
    ]
    return {
        "preset": current["preset"],
        "metrics": {
            name: make_delta_entry(current["metrics"].get(name), baseline.get("metrics", {}).get(name))
            for name in metric_names
        },
        "latency": {
            name: make_delta_entry(current["latency"].get(name), baseline.get("latency", {}).get(name))
            for name in latency_names
        },
    }


def make_delta_entry(current_value: Any, baseline_value: Any) -> dict[str, float | None]:
    current_float = float(current_value) if current_value is not None else None
    baseline_float = float(baseline_value) if baseline_value is not None else None
    delta = None
    if current_float is not None and baseline_float is not None:
        delta = current_float - baseline_float
    return {
        "current": current_float,
        "baseline": baseline_float,
        "delta": delta,
    }


def evaluate_release_gates(comparisons: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    comparison_by_preset = {comparison["preset"]: comparison for comparison in comparisons}
    gates: list[dict[str, Any]] = []

    _append_gate(
        gates,
        name="production_fast_min_solved_rate",
        observed=comparison_by_preset["production_fast"]["metrics"]["board_solved_rate"],
        threshold=args.min_production_fast_solved_rate,
        comparator=">=",
    )
    _append_gate(
        gates,
        name="production_pure_min_solved_rate",
        observed=comparison_by_preset["production_pure"]["metrics"]["board_solved_rate"],
        threshold=args.min_production_pure_solved_rate,
        comparator=">=",
    )
    _append_gate(
        gates,
        name="research_raw_min_blank_cell_accuracy",
        observed=comparison_by_preset["research_raw"]["metrics"]["blank_cell_accuracy"],
        threshold=args.min_research_raw_blank_cell_accuracy,
        comparator=">=",
    )
    _append_gate(
        gates,
        name="production_fast_max_board_ms",
        observed=comparison_by_preset["production_fast"]["latency"]["mean_board_duration_ms"],
        threshold=args.max_production_fast_board_ms,
        comparator="<=",
    )
    _append_gate(
        gates,
        name="production_pure_max_board_ms",
        observed=comparison_by_preset["production_pure"]["latency"]["mean_board_duration_ms"],
        threshold=args.max_production_pure_board_ms,
        comparator="<=",
    )
    return gates


def evaluate_baseline_gates(
    baseline_summary: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if baseline_summary is None:
        return []

    thresholds = {
        "production_fast": {
            "solved_rate_drop": args.max_production_fast_solved_rate_drop,
            "board_ms_increase": args.max_production_fast_board_ms_increase,
        },
        "production_pure": {
            "solved_rate_drop": args.max_production_pure_solved_rate_drop,
            "board_ms_increase": args.max_production_pure_board_ms_increase,
        },
    }
    summary_by_preset = {summary["preset"]: summary for summary in baseline_summary["comparisons"]}
    gates: list[dict[str, Any]] = []

    for preset_name, preset_thresholds in thresholds.items():
        preset_summary = summary_by_preset.get(preset_name)
        if preset_summary is None:
            continue
        solved_rate_drop_limit = preset_thresholds["solved_rate_drop"]
        if solved_rate_drop_limit is not None:
            solved_rate_delta = preset_summary["metrics"]["board_solved_rate"]["delta"]
            observed_drop = -solved_rate_delta if solved_rate_delta is not None else float("inf")
            _append_gate(
                gates,
                name=f"{preset_name}_max_solved_rate_drop",
                observed=observed_drop,
                threshold=solved_rate_drop_limit,
                comparator="<=",
            )
        board_ms_increase_limit = preset_thresholds["board_ms_increase"]
        if board_ms_increase_limit is not None:
            board_ms_delta = preset_summary["latency"]["mean_board_duration_ms"]["delta"]
            observed_increase = board_ms_delta if board_ms_delta is not None else float("inf")
            _append_gate(
                gates,
                name=f"{preset_name}_max_board_ms_increase",
                observed=observed_increase,
                threshold=board_ms_increase_limit,
                comparator="<=",
            )

    return gates


def _append_gate(
    gates: list[dict[str, Any]],
    *,
    name: str,
    observed: float,
    threshold: float,
    comparator: str,
) -> None:
    if comparator == ">=":
        passed = observed >= threshold
    elif comparator == "<=":
        passed = observed <= threshold
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")

    gates.append(
        {
            "name": name,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": passed,
        }
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.benchmark_max_samples < 1:
        parser.error("--benchmark-max-samples must be at least 1.")
    if args.benchmark_warmup_batches < 0:
        parser.error("--benchmark-warmup-batches cannot be negative.")
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be at least 1.")
    required_presets = {"production_fast", "production_pure", "research_raw"}
    missing_presets = sorted(required_presets.difference(args.presets))
    if missing_presets:
        parser.error("--presets must include production_fast, production_pure, and research_raw.")
    regression_thresholds = [
        args.max_production_fast_solved_rate_drop,
        args.max_production_pure_solved_rate_drop,
        args.max_production_fast_board_ms_increase,
        args.max_production_pure_board_ms_increase,
    ]
    if any(threshold is not None for threshold in regression_thresholds) and args.baseline_report is None:
        parser.error("Regression thresholds require --baseline-report.")


if __name__ == "__main__":
    main()
