import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from solver.cli import main, parse_puzzle_text


PUZZLE_TEXT = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

PRETTY_PUZZLE_TEXT = """5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
---------------------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
---------------------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9
"""


class SolverCLITests(unittest.TestCase):
    def test_parse_puzzle_text_supports_dots_and_whitespace(self) -> None:
        board = parse_puzzle_text("53.07....\n6..195...\n.98....6.\n8...6...3\n4..8.3..1\n7...2...6\n.6....28.\n...419..5\n....8..79")
        self.assertEqual(board[0][:3], [5, 3, 0])
        self.assertEqual(board[8][8], 9)

    def test_parse_puzzle_text_supports_pretty_printed_board(self) -> None:
        board = parse_puzzle_text(PRETTY_PUZZLE_TEXT)
        self.assertEqual(board[0][:3], [5, 3, 0])
        self.assertEqual(board[3][0], 8)

    def test_solve_command_prints_solution(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["solve", "--puzzle", PUZZLE_TEXT])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Solution:", rendered)
        self.assertIn("5 3 4", rendered)

    def test_solve_command_supports_file_input_and_uniqueness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            puzzle_path = Path(temp_dir) / "puzzle.txt"
            puzzle_path.write_text(PUZZLE_TEXT, encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["solve", "--file", str(puzzle_path), "--check-unique"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Unique solution: yes", rendered)

    def test_solve_command_supports_stdin(self) -> None:
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(PRETTY_PUZZLE_TEXT)):
            with redirect_stdout(output):
                exit_code = main(["solve", "--stdin", "--check-unique"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Unique solution: yes", rendered)
        self.assertIn("Solution:", rendered)

    def test_generate_command_can_print_solution(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["generate", "--blanks", "10", "--seed", "7", "--show-solution"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Puzzle:", rendered)
        self.assertIn("Solution:", rendered)

    def test_generate_command_can_write_puzzle_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            puzzle_path = Path(temp_dir) / "generated_puzzle.txt"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["generate", "--blanks", "10", "--seed", "7", "--output", str(puzzle_path)])

            rendered = output.getvalue()
            written = puzzle_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("Saved puzzle to", rendered)
        self.assertIn("|", written)
        self.assertNotIn("Puzzle:", written)

    def test_generate_command_can_write_solution_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            puzzle_path = Path(temp_dir) / "generated_puzzle.txt"
            solution_path = Path(temp_dir) / "generated_solution.txt"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "generate",
                        "--blanks",
                        "10",
                        "--seed",
                        "7",
                        "--output",
                        str(puzzle_path),
                        "--solution-output",
                        str(solution_path),
                    ]
                )

            solution_text = solution_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("Saved solution to", output.getvalue())
        self.assertIn("2 5 4", solution_text)

    def test_export_dataset_command_writes_training_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset"
            manifest_path = Path(temp_dir) / "manifest.jsonl"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "export-dataset",
                        "--size",
                        "2",
                        "--blanks",
                        "10",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(output_dir),
                        "--manifest",
                        str(manifest_path),
                    ]
                )

            records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(exit_code, 0)
            self.assertIn("Exported 2 puzzle files", output.getvalue())
            self.assertEqual(len(records), 2)
            self.assertIn("puzzle", records[0])
            self.assertIn("solution", records[0])
            self.assertTrue((output_dir / "puzzle_00001.txt").exists())
            self.assertTrue((output_dir / "puzzle_00001_solution.txt").exists())

    def test_export_dataset_command_writes_split_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset"
            manifest_dir = Path(temp_dir) / "manifests"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "export-dataset",
                        "--train-size",
                        "2",
                        "--val-size",
                        "1",
                        "--test-size",
                        "1",
                        "--blanks",
                        "10",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(output_dir),
                        "--manifest-dir",
                        str(manifest_dir),
                    ]
                )

            train_records = [json.loads(line) for line in (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            val_records = [json.loads(line) for line in (manifest_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()]
            test_records = [json.loads(line) for line in (manifest_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(exit_code, 0)
            self.assertIn("Saved manifest to", output.getvalue())
            self.assertEqual(len(train_records), 2)
            self.assertEqual(len(val_records), 1)
            self.assertEqual(len(test_records), 1)
            self.assertEqual(train_records[0]["split"], "train")
            self.assertEqual(val_records[0]["split"], "val")
            self.assertEqual(test_records[0]["split"], "test")
            self.assertTrue((output_dir / "train" / "puzzle_00001.txt").exists())
            self.assertTrue((output_dir / "val" / "puzzle_00001.txt").exists())
            self.assertTrue((output_dir / "test" / "puzzle_00001.txt").exists())

    def test_parse_puzzle_text_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 81 cells"):
            parse_puzzle_text("123")


if __name__ == "__main__":
    unittest.main()
