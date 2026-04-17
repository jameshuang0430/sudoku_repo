import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from solver import (
    board_to_string,
    count_solutions,
    export_puzzle_dataset,
    export_puzzle_dataset_splits,
    generate_puzzle,
    generate_solved_board,
    solve_board,
    validate_board,
)


PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

UNSOLVABLE_PUZZLE = [
    [5, 3, 1, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


class SolverTests(unittest.TestCase):
    def test_validate_board_rejects_duplicates(self) -> None:
        invalid = [row[:] for row in PUZZLE]
        invalid[0][2] = 5

        with self.assertRaisesRegex(ValueError, "Duplicate value 5 found in row 0."):
            validate_board(invalid)

    def test_solve_board_returns_expected_solution(self) -> None:
        self.assertEqual(solve_board(PUZZLE), SOLUTION)

    def test_count_solutions_reports_unique_puzzle(self) -> None:
        self.assertEqual(count_solutions(PUZZLE), 1)

    def test_unsolved_board_string_uses_dots_for_blanks(self) -> None:
        rendered = board_to_string(PUZZLE)
        self.assertIn(". . .", rendered)
        self.assertIn("|", rendered)

    def test_unsolvable_puzzle_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "The puzzle has no valid solution."):
            solve_board(UNSOLVABLE_PUZZLE)

    def test_generate_solved_board_returns_valid_complete_board(self) -> None:
        board = generate_solved_board(seed=7)
        validate_board(board)
        self.assertTrue(all(all(value != 0 for value in row) for row in board))

    def test_generate_puzzle_returns_unique_training_pair(self) -> None:
        puzzle, solution = generate_puzzle(blanks=40, seed=7)

        validate_board(puzzle)
        validate_board(solution)
        self.assertEqual(sum(value == 0 for row in puzzle for value in row), 40)
        self.assertEqual(count_solutions(puzzle), 1)
        self.assertEqual(solve_board(puzzle), solution)

    def test_export_puzzle_dataset_writes_training_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            records = export_puzzle_dataset(
                output_dir=temp_dir,
                size=2,
                blanks=10,
                seed=7,
            )

            self.assertEqual(len(records), 2)
            first_puzzle = Path(records[0]["puzzle_path"])
            first_solution = Path(records[0]["solution_path"])
            self.assertTrue(first_puzzle.exists())
            self.assertTrue(first_solution.exists())
            self.assertIn("|", first_puzzle.read_text(encoding="utf-8"))
            self.assertIn("|", first_solution.read_text(encoding="utf-8"))

    def test_export_puzzle_dataset_splits_writes_split_files_and_manifests(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset"
            manifest_dir = Path(temp_dir) / "manifests"
            split_records = export_puzzle_dataset_splits(
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                train_size=2,
                val_size=1,
                test_size=1,
                blanks=10,
                seed=7,
            )

            self.assertEqual(len(split_records["train"]), 2)
            self.assertEqual(len(split_records["val"]), 1)
            self.assertEqual(len(split_records["test"]), 1)
            self.assertTrue((output_dir / "train" / "puzzle_00001.txt").exists())
            self.assertTrue((output_dir / "val" / "puzzle_00001.txt").exists())
            self.assertTrue((output_dir / "test" / "puzzle_00001.txt").exists())
            self.assertTrue((manifest_dir / "train.jsonl").exists())
            self.assertTrue((manifest_dir / "val.jsonl").exists())
            self.assertTrue((manifest_dir / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
