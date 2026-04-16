import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai.infer import main as infer_main
from solver import generate_puzzle


class InferTests(unittest.TestCase):
    def test_infer_main_supports_file_input(self) -> None:
        puzzle, _solution = generate_puzzle(blanks=10, seed=7)
        puzzle_text = "\n".join("".join(str(value) for value in row) for row in puzzle)

        with tempfile.TemporaryDirectory() as temp_dir:
            puzzle_path = Path(temp_dir) / "puzzle.txt"
            puzzle_path.write_text(puzzle_text, encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                infer_main(
                    [
                        "--checkpoint",
                        r"ai\checkpoints\transformer_large_current.best.pt",
                        "--file",
                        str(puzzle_path),
                        "--decode-preset",
                        "production_pure",
                    ]
                )

        rendered = output.getvalue()
        self.assertIn("checkpoint=ai\\checkpoints\\transformer_large_current.best.pt", rendered)
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

        with redirect_stdout(output):
            infer_main(
                [
                    "--checkpoint",
                    r"ai\checkpoints\transformer_large_current.best.pt",
                    "--puzzle",
                    puzzle,
                ]
            )

        rendered = output.getvalue()
        self.assertIn("decode_preset=production_fast", rendered)
        self.assertIn("preset_profile=latency_oriented", rendered)
        self.assertIn("decode_mode=solver_guided", rendered)

    def test_infer_main_supports_inline_puzzle(self) -> None:
        puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        output = io.StringIO()

        with redirect_stdout(output):
            infer_main(
                [
                    "--checkpoint",
                    r"ai\checkpoints\transformer_large_current.best.pt",
                    "--puzzle",
                    puzzle,
                    "--decode-mode",
                    "argmax",
                    "--decode-preset",
                    "research_raw",
                ]
            )

        rendered = output.getvalue()
        self.assertIn("decode_preset=research_raw", rendered)
        self.assertIn("preset_profile=research", rendered)
        self.assertIn("decode_mode=argmax", rendered)
        self.assertIn("prediction:", rendered)


if __name__ == "__main__":
    unittest.main()
