import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import ai.product as product_module
from ai.infer import main as infer_main
from ai.model import SudokuMLP
from ai.product import main as product_main
from solver import generate_puzzle


class InferTests(unittest.TestCase):
    def _write_checkpoint(self, directory: str) -> Path:
        checkpoint_path = Path(directory) / "model.pt"
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def test_infer_main_supports_file_input(self) -> None:
        puzzle, _solution = generate_puzzle(blanks=10, seed=7)
        puzzle_text = "\n".join("".join(str(value) for value in row) for row in puzzle)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            puzzle_path = Path(temp_dir) / "puzzle.txt"
            puzzle_path.write_text(puzzle_text, encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                infer_main(
                    [
                        "--checkpoint",
                        str(checkpoint_path),
                        "--file",
                        str(puzzle_path),
                        "--decode-preset",
                        "production_pure",
                    ]
                )

        rendered = output.getvalue()
        self.assertIn(f"checkpoint={checkpoint_path}", rendered)
        self.assertIn("decode_preset=production_pure", rendered)
        self.assertIn("preset_profile=accuracy_oriented", rendered)
        self.assertIn("decode_mode=iterative", rendered)
        self.assertIn("iterative_threshold=0.75", rendered)
        self.assertIn("iterative_max_fills_per_round=2", rendered)
        self.assertIn("puzzle:", rendered)
        self.assertIn("prediction:", rendered)
        self.assertIn("is_valid=", rendered)

    def test_infer_main_defaults_to_production_fast(self) -> None:
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            with redirect_stdout(output):
                infer_main(
                    [
                        "--checkpoint",
                        str(checkpoint_path),
                        "--puzzle",
                        puzzle,
                    ]
                )

        rendered = output.getvalue()
        self.assertIn(f"checkpoint={checkpoint_path}", rendered)
        self.assertIn("decode_preset=production_fast", rendered)
        self.assertIn("preset_profile=latency_oriented", rendered)
        self.assertIn("decode_mode=solver_guided", rendered)

    def test_infer_main_supports_inline_puzzle(self) -> None:
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            with redirect_stdout(output):
                infer_main(
                    [
                        "--checkpoint",
                        str(checkpoint_path),
                        "--puzzle",
                        puzzle,
                        "--decode-mode",
                        "argmax",
                        "--decode-preset",
                        "research_raw",
                    ]
                )

        rendered = output.getvalue()
        self.assertIn(f"checkpoint={checkpoint_path}", rendered)
        self.assertIn("decode_preset=research_raw", rendered)
        self.assertIn("preset_profile=research", rendered)
        self.assertIn("decode_mode=argmax", rendered)
        self.assertIn("prediction:", rendered)

    def test_product_wrapper_defaults_to_fast_preset_and_checkpoint(self) -> None:
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            with patch.object(product_module, "DEFAULT_PRODUCTION_CHECKPOINT", checkpoint_path):
                with redirect_stdout(output):
                    product_main(["--puzzle", puzzle])

        rendered = output.getvalue()
        self.assertIn(f"checkpoint={checkpoint_path}", rendered)
        self.assertIn("decode_preset=production_fast", rendered)
        self.assertIn("preset_profile=latency_oriented", rendered)
        self.assertIn("decode_mode=solver_guided", rendered)

    def test_product_wrapper_maps_pure_preset(self) -> None:
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            with patch.object(product_module, "DEFAULT_PRODUCTION_CHECKPOINT", checkpoint_path):
                with redirect_stdout(output):
                    product_main(["--preset", "pure", "--puzzle", puzzle])

        rendered = output.getvalue()
        self.assertIn(f"checkpoint={checkpoint_path}", rendered)
        self.assertIn("decode_preset=production_pure", rendered)
        self.assertIn("preset_profile=accuracy_oriented", rendered)
        self.assertIn("decode_mode=iterative", rendered)


if __name__ == "__main__":
    unittest.main()
