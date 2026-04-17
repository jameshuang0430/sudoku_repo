from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .release_check import main as release_check_main


DEFAULT_PRODUCTION_CHECKPOINT = Path(r"ai\checkpoints\transformer_large_current.best.pt")
DEFAULT_RELEASE_DATASET = Path(r"data\manifests_generalization\test.jsonl")


@dataclass(frozen=True)
class ReleaseGateProfile:
    batch_size: int
    benchmark_max_samples: int
    benchmark_warmup_batches: int
    benchmark_repeats: int
    compare_report: Path
    baseline_report: Path
    min_production_fast_solved_rate: float
    min_production_pure_solved_rate: float
    min_research_raw_blank_cell_accuracy: float
    max_production_fast_board_ms: float
    max_production_pure_board_ms: float
    max_production_fast_solved_rate_drop: float
    max_production_pure_solved_rate_drop: float
    max_production_fast_board_ms_increase: float
    max_production_pure_board_ms_increase: float


RELEASE_GATE_PROFILES = {
    "smoke": ReleaseGateProfile(
        batch_size=1,
        benchmark_max_samples=128,
        benchmark_warmup_batches=1,
        benchmark_repeats=3,
        compare_report=Path(r"ai\reports\release_check_generalization_smoke.json"),
        baseline_report=Path(r"ai\reports\release_check_generalization_baseline.json"),
        min_production_fast_solved_rate=0.999,
        min_production_pure_solved_rate=0.99,
        min_research_raw_blank_cell_accuracy=0.84,
        max_production_fast_board_ms=3.0,
        max_production_pure_board_ms=30.0,
        max_production_fast_solved_rate_drop=0.001,
        max_production_pure_solved_rate_drop=0.01,
        max_production_fast_board_ms_increase=1.0,
        max_production_pure_board_ms_increase=6.0,
    ),
    "full": ReleaseGateProfile(
        batch_size=32,
        benchmark_max_samples=512,
        benchmark_warmup_batches=1,
        benchmark_repeats=3,
        compare_report=Path(r"ai\reports\release_check_generalization_full.json"),
        baseline_report=Path(r"ai\reports\release_check_generalization_full_baseline.json"),
        min_production_fast_solved_rate=0.999,
        min_production_pure_solved_rate=0.995,
        min_research_raw_blank_cell_accuracy=0.84,
        max_production_fast_board_ms=2.0,
        max_production_pure_board_ms=30.0,
        max_production_fast_solved_rate_drop=0.001,
        max_production_pure_solved_rate_drop=0.01,
        max_production_fast_board_ms_increase=1.0,
        max_production_pure_board_ms_increase=6.0,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the standard Sudoku release gate with fixed smoke or full-manifest defaults."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_PRODUCTION_CHECKPOINT,
        help="Override the default production checkpoint path.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_RELEASE_DATASET,
        help="Override the default release-gate dataset path.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RELEASE_GATE_PROFILES.keys()),
        default="smoke",
        help="Choose the release-gate profile.",
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "baseline"],
        default="compare",
        help="Use compare to check against the saved baseline or baseline to regenerate the profile baseline report.",
    )
    args, release_check_args = parser.parse_known_args(argv)
    if not args.checkpoint.exists():
        parser.error(f"--checkpoint not found: {args.checkpoint}")
    if not args.dataset.exists():
        parser.error(f"--dataset not found: {args.dataset}")
    if args.mode == "compare" and not _has_option(release_check_args, "--baseline-report"):
        baseline_path = RELEASE_GATE_PROFILES[args.profile].baseline_report
        if not baseline_path.exists():
            parser.error(f"default baseline report not found for profile '{args.profile}': {baseline_path}")
    return parser, args, release_check_args


def build_release_check_argv(args: argparse.Namespace, release_check_args: Sequence[str]) -> list[str]:
    profile = RELEASE_GATE_PROFILES[args.profile]
    forwarded = [
        "--checkpoint",
        str(args.checkpoint),
        "--dataset",
        str(args.dataset),
    ]

    if not _has_option(release_check_args, "--batch-size"):
        forwarded.extend(["--batch-size", str(profile.batch_size)])
    if not _has_option(release_check_args, "--benchmark-max-samples"):
        forwarded.extend(["--benchmark-max-samples", str(profile.benchmark_max_samples)])
    if not _has_option(release_check_args, "--benchmark-warmup-batches"):
        forwarded.extend(["--benchmark-warmup-batches", str(profile.benchmark_warmup_batches)])
    if not _has_option(release_check_args, "--benchmark-repeats"):
        forwarded.extend(["--benchmark-repeats", str(profile.benchmark_repeats)])

    default_report = profile.baseline_report if args.mode == "baseline" else profile.compare_report
    if not _has_option(release_check_args, "--report"):
        forwarded.extend(["--report", str(default_report)])

    if args.mode == "compare":
        if not _has_option(release_check_args, "--baseline-report"):
            forwarded.extend(["--baseline-report", str(profile.baseline_report)])
        _append_threshold(forwarded, release_check_args, "--min-production-fast-solved-rate", profile.min_production_fast_solved_rate)
        _append_threshold(forwarded, release_check_args, "--min-production-pure-solved-rate", profile.min_production_pure_solved_rate)
        _append_threshold(
            forwarded,
            release_check_args,
            "--min-research-raw-blank-cell-accuracy",
            profile.min_research_raw_blank_cell_accuracy,
        )
        _append_threshold(forwarded, release_check_args, "--max-production-fast-board-ms", profile.max_production_fast_board_ms)
        _append_threshold(forwarded, release_check_args, "--max-production-pure-board-ms", profile.max_production_pure_board_ms)
        _append_threshold(
            forwarded,
            release_check_args,
            "--max-production-fast-solved-rate-drop",
            profile.max_production_fast_solved_rate_drop,
        )
        _append_threshold(
            forwarded,
            release_check_args,
            "--max-production-pure-solved-rate-drop",
            profile.max_production_pure_solved_rate_drop,
        )
        _append_threshold(
            forwarded,
            release_check_args,
            "--max-production-fast-board-ms-increase",
            profile.max_production_fast_board_ms_increase,
        )
        _append_threshold(
            forwarded,
            release_check_args,
            "--max-production-pure-board-ms-increase",
            profile.max_production_pure_board_ms_increase,
        )

    forwarded.extend(release_check_args)
    return forwarded


def _append_threshold(
    forwarded: list[str],
    release_check_args: Sequence[str],
    option: str,
    value: float,
) -> None:
    if not _has_option(release_check_args, option):
        forwarded.extend([option, str(value)])


def _has_option(args: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in args)


def main(argv: Sequence[str] | None = None) -> None:
    _parser, args, release_check_args = parse_args(argv)
    print(f"release_gate_profile={args.profile} release_gate_mode={args.mode}")
    release_check_main(build_release_check_argv(args, release_check_args))


if __name__ == "__main__":
    main()
