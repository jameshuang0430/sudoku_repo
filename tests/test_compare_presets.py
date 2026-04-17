import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai.compare_presets import main as compare_presets_main
from ai.dataset import SudokuDataset, sample_to_record
from ai.model import SudokuMLP


class ComparePresetsTests(unittest.TestCase):
    def test_compare_presets_main_writes_report(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "compare.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            report_path = Path(temp_dir) / "compare_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)

            compare_presets_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_path),
                    "--batch-size",
                    "2",
                    "--presets",
                    "production_fast",
                    "production_pure",
                    "research_raw",
                    "--benchmark-max-samples",
                    "4",
                    "--benchmark-warmup-batches",
                    "0",
                    "--benchmark-repeats",
                    "1",
                    "--report",
                    str(report_path),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("run_metadata", report)
        self.assertEqual(report["run_metadata"]["entrypoint"], "ai.compare_presets")
        self.assertEqual(report["run_metadata"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(report["run_metadata"]["dataset_path"], str(dataset_path))
        self.assertEqual(report["run_metadata"]["model_type"], "mlp")
        self.assertEqual(report["compare_config"]["presets"], ["production_fast", "production_pure", "research_raw"])
        self.assertEqual(report["compare_config"]["batch_size"], 2)
        self.assertEqual(len(report["comparisons"]), 3)
        self.assertEqual(
            [comparison["preset"] for comparison in report["comparisons"]],
            ["production_fast", "production_pure", "research_raw"],
        )
        self.assertTrue(all("metrics" in comparison for comparison in report["comparisons"]))
        self.assertTrue(all("latency" in comparison for comparison in report["comparisons"]))
        self.assertTrue(
            all(comparison["latency"]["mean_board_duration_ms"] >= 0.0 for comparison in report["comparisons"])
        )


if __name__ == "__main__":
    unittest.main()
