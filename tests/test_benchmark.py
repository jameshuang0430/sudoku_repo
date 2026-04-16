import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai.benchmark import main as benchmark_main
from ai.dataset import SudokuDataset, sample_to_record
from ai.model import SudokuMLP


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_main_writes_report(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "bench.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            report_path = Path(temp_dir) / "benchmark_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)

            benchmark_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_path),
                    "--batch-sizes",
                    "1",
                    "2",
                    "--decode-presets",
                    "argmax",
                    "iterative_strict",
                    "--max-samples",
                    "4",
                    "--warmup-batches",
                    "0",
                    "--repeats",
                    "1",
                    "--report",
                    str(report_path),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["benchmark_config"]["batch_sizes"], [1, 2])
        self.assertEqual(report["benchmark_config"]["decode_presets"], ["argmax", "iterative_strict"])
        self.assertEqual(len(report["results"]), 4)
        self.assertTrue(all(result["sample_count"] == 4 for result in report["results"]))
        self.assertTrue(all(result["throughput_boards_per_second"] >= 0.0 for result in report["results"]))


if __name__ == "__main__":
    unittest.main()
