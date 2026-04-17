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


if __name__ == "__main__":
    unittest.main()

