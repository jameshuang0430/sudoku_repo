import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from ai.dataset import SudokuDataset, sample_to_record
from ai.model import SudokuMLP
from ai.release_check import main as release_check_main


class ReleaseCheckTests(unittest.TestCase):
    def test_release_check_main_writes_report_and_passes(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "release.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            report_path = Path(temp_dir) / "release_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)

            release_check_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_path),
                    "--batch-size",
                    "2",
                    "--benchmark-max-samples",
                    "4",
                    "--benchmark-warmup-batches",
                    "0",
                    "--benchmark-repeats",
                    "1",
                    "--min-production-fast-solved-rate",
                    "0.0",
                    "--min-production-pure-solved-rate",
                    "0.0",
                    "--min-research-raw-blank-cell-accuracy",
                    "0.0",
                    "--max-production-fast-board-ms",
                    "999.0",
                    "--max-production-pure-board-ms",
                    "999.0",
                    "--report",
                    str(report_path),
                    "--tests-command",
                    sys.executable,
                    "-c",
                    "print('release-ok')",
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["passed"])
        self.assertEqual(report["run_metadata"]["entrypoint"], "ai.release_check")
        self.assertEqual(report["run_metadata"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(report["run_metadata"]["dataset_path"], str(dataset_path))
        self.assertTrue(report["tests"]["passed"])
        self.assertFalse(report["tests"]["skipped"])
        self.assertEqual(len(report["comparisons"]), 3)
        self.assertEqual(len(report["gates"]), 5)
        self.assertTrue(all(gate["passed"] for gate in report["gates"]))

    def test_release_check_main_fails_when_gate_is_not_met(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "release.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            report_path = Path(temp_dir) / "release_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)

            with self.assertRaises(SystemExit) as context:
                release_check_main(
                    [
                        "--checkpoint",
                        str(checkpoint_path),
                        "--dataset",
                        str(dataset_path),
                        "--batch-size",
                        "2",
                        "--benchmark-max-samples",
                        "4",
                        "--benchmark-warmup-batches",
                        "0",
                        "--benchmark-repeats",
                        "1",
                        "--skip-tests",
                        "--min-production-fast-solved-rate",
                        "1.1",
                        "--min-production-pure-solved-rate",
                        "0.0",
                        "--min-research-raw-blank-cell-accuracy",
                        "0.0",
                        "--max-production-fast-board-ms",
                        "999.0",
                        "--max-production-pure-board-ms",
                        "999.0",
                        "--report",
                        str(report_path),
                    ]
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(context.exception.code, 1)
        self.assertFalse(report["passed"])
        self.assertTrue(report["tests"]["passed"])
        self.assertTrue(report["tests"]["skipped"])
        self.assertTrue(any(not gate["passed"] for gate in report["gates"]))

    def test_release_check_main_writes_baseline_comparison_and_regression_gates(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }
        baseline_report = {
            "run_metadata": {"entrypoint": "ai.release_check", "commit": "baseline-sha"},
            "comparisons": [
                {
                    "preset": "production_fast",
                    "metrics": {
                        "blank_cell_accuracy": 0.20,
                        "board_solved_rate": 0.0,
                        "valid_board_rate": 0.0,
                        "mean_total_conflicts": 12.0,
                        "mean_postprocess_change_count": 0.5,
                        "mean_decode_iteration_count": 0.0,
                    },
                    "latency": {
                        "mean_board_duration_ms": 999.0,
                        "throughput_boards_per_second": 1.0,
                    },
                },
                {
                    "preset": "production_pure",
                    "metrics": {
                        "blank_cell_accuracy": 0.20,
                        "board_solved_rate": 0.0,
                        "valid_board_rate": 0.0,
                        "mean_total_conflicts": 12.0,
                        "mean_postprocess_change_count": 0.5,
                        "mean_decode_iteration_count": 0.0,
                    },
                    "latency": {
                        "mean_board_duration_ms": 999.0,
                        "throughput_boards_per_second": 1.0,
                    },
                },
                {
                    "preset": "research_raw",
                    "metrics": {
                        "blank_cell_accuracy": 0.0,
                        "board_solved_rate": 0.0,
                        "valid_board_rate": 0.0,
                        "mean_total_conflicts": 20.0,
                        "mean_postprocess_change_count": 0.0,
                        "mean_decode_iteration_count": 0.0,
                    },
                    "latency": {
                        "mean_board_duration_ms": 999.0,
                        "throughput_boards_per_second": 1.0,
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "release.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            baseline_report_path = Path(temp_dir) / "baseline_report.json"
            report_path = Path(temp_dir) / "release_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)
            baseline_report_path.write_text(json.dumps(baseline_report), encoding="utf-8")

            release_check_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_path),
                    "--batch-size",
                    "2",
                    "--benchmark-max-samples",
                    "4",
                    "--benchmark-warmup-batches",
                    "0",
                    "--benchmark-repeats",
                    "1",
                    "--skip-tests",
                    "--baseline-report",
                    str(baseline_report_path),
                    "--min-production-fast-solved-rate",
                    "0.0",
                    "--min-production-pure-solved-rate",
                    "0.0",
                    "--min-research-raw-blank-cell-accuracy",
                    "0.0",
                    "--max-production-fast-board-ms",
                    "999.0",
                    "--max-production-pure-board-ms",
                    "999.0",
                    "--max-production-fast-solved-rate-drop",
                    "1.0",
                    "--max-production-pure-solved-rate-drop",
                    "1.0",
                    "--max-production-fast-board-ms-increase",
                    "1.0",
                    "--max-production-pure-board-ms-increase",
                    "1.0",
                    "--report",
                    str(report_path),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["passed"])
        self.assertIsNotNone(report["baseline"])
        self.assertEqual(report["baseline"]["path"], str(baseline_report_path))
        self.assertEqual(report["baseline"]["run_metadata"]["commit"], "baseline-sha")
        self.assertEqual(report["baseline"]["missing_presets"], [])
        self.assertEqual(len(report["baseline"]["comparisons"]), 3)
        self.assertEqual(len(report["gates"]), 9)
        self.assertTrue(all(gate["passed"] for gate in report["gates"]))

        baseline_comparison = {
            comparison["preset"]: comparison for comparison in report["baseline"]["comparisons"]
        }
        self.assertLessEqual(
            baseline_comparison["production_fast"]["latency"]["mean_board_duration_ms"]["delta"],
            1.0,
        )
        self.assertLessEqual(
            baseline_comparison["production_pure"]["latency"]["mean_board_duration_ms"]["delta"],
            1.0,
        )

    def test_release_check_requires_baseline_report_when_regression_threshold_is_set(self) -> None:
        with self.assertRaises(SystemExit) as context:
            release_check_main(
                [
                    "--checkpoint",
                    "dummy.pt",
                    "--max-production-fast-solved-rate-drop",
                    "0.01",
                ]
            )

        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

