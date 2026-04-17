import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from ai.checkpoint import load_model_from_checkpoint
from ai.dataset import SudokuDataset, SudokuFileDataset, flat_to_board, sample_to_record
from ai.decode import compose_completed_boards, decode_completed_boards
from ai.eval import evaluate_model, main as eval_main, summarize_board_violations
from ai.model import SudokuMLP, SudokuTransformer
from ai.plot_results import main as plot_main
from ai.train import constraint_consistency_penalty, main as train_main
from solver import export_puzzle_dataset, solve_board_with_scores


class AITests(unittest.TestCase):
    def test_dataset_sample_shapes(self) -> None:
        dataset = SudokuDataset(size=2, blanks=10, seed=7)
        sample = dataset[0]

        self.assertEqual(sample["digits"].shape, (81,))
        self.assertEqual(sample["givens"].shape, (81,))
        self.assertEqual(sample["targets"].shape, (81,))
        self.assertEqual(sample["digits"].dtype, torch.long)
        self.assertEqual(sample["targets"].dtype, torch.long)

    def test_model_forward_shape(self) -> None:
        dataset = SudokuDataset(size=2, blanks=10, seed=7)
        batch_digits = torch.stack([dataset[0]["digits"], dataset[1]["digits"]])
        batch_givens = torch.stack([dataset[0]["givens"], dataset[1]["givens"]])

        model = SudokuMLP()
        logits = model(batch_digits, batch_givens)

        self.assertEqual(logits.shape, (2, 81, 9))

    def test_transformer_forward_shape(self) -> None:
        dataset = SudokuDataset(size=2, blanks=10, seed=7)
        batch_digits = torch.stack([dataset[0]["digits"], dataset[1]["digits"]])
        batch_givens = torch.stack([dataset[0]["givens"], dataset[1]["givens"]])

        model = SudokuTransformer(embed_dim=32, num_heads=4, depth=2, ff_dim=64, dropout=0.0)
        logits = model(batch_digits, batch_givens)

        self.assertEqual(logits.shape, (2, 81, 9))

    def test_sample_to_record_returns_9x9_boards(self) -> None:
        dataset = SudokuDataset(size=1, blanks=10, seed=7)
        record = sample_to_record(dataset[0])

        self.assertEqual(len(record["puzzle"]), 9)
        self.assertEqual(len(record["solution"]), 9)
        self.assertEqual(record["blank_count"], 10)

    def test_file_dataset_loads_inline_records(self) -> None:
        dataset = SudokuDataset(size=2, blanks=10, seed=7)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            file_dataset = SudokuFileDataset(dataset_path)

        self.assertEqual(len(file_dataset), 2)
        self.assertEqual(file_dataset[0]["digits"].shape, (81,))

    def test_file_dataset_loads_path_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "puzzles"
            records = export_puzzle_dataset(output_dir=export_dir, size=2, blanks=10, seed=7)
            dataset_path = Path(temp_dir) / "manifest.jsonl"
            with dataset_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            {
                                "puzzle_path": str(Path("puzzles") / Path(record["puzzle_path"]).name),
                                "solution_path": str(Path("puzzles") / Path(record["solution_path"]).name),
                            }
                        )
                    )
                    handle.write("\n")

            file_dataset = SudokuFileDataset(dataset_path)

        self.assertEqual(len(file_dataset), 2)
        self.assertTrue(torch.equal(file_dataset[0]["givens"], (file_dataset[0]["digits"] != 0).to(dtype=torch.float32)))

    def test_compose_completed_boards_preserves_givens(self) -> None:
        digits = torch.tensor([[5, 0, 0, 4]], dtype=torch.long)
        predictions = torch.tensor([[2, 3, 4, 1]], dtype=torch.long)

        completed = compose_completed_boards(digits, predictions)

        self.assertEqual(completed.tolist(), [[5, 4, 5, 4]])

    def test_decode_completed_boards_solver_guided_repairs_wrong_argmax(self) -> None:
        dataset = SudokuDataset(size=1, blanks=10, seed=7)
        sample = dataset[0]
        digits = sample["digits"].unsqueeze(0)
        givens = sample["givens"].unsqueeze(0)
        targets = sample["targets"]
        logits = torch.full((1, 81, 9), -5.0, dtype=torch.float32)

        for index, target_value in enumerate(targets.tolist()):
            logits[0, index, target_value] = 5.0

        blank_indices = (sample["digits"] == 0).nonzero(as_tuple=False).flatten().tolist()
        wrong_index = blank_indices[0]
        correct_class = targets[wrong_index].item()
        wrong_class = (correct_class + 1) % 9
        logits[0, wrong_index, correct_class] = 1.0
        logits[0, wrong_index, wrong_class] = 9.0

        class StaticModel(torch.nn.Module):
            def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
                return logits

        completed, postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
            StaticModel(),
            digits,
            givens,
            torch.device("cpu"),
            mode="solver_guided",
            initial_logits=logits,
        )

        self.assertEqual(completed[0].tolist(), (targets + 1).tolist())
        self.assertEqual(postprocess_change_counts, [1])
        self.assertEqual(decode_iteration_counts, [0])

    def test_decode_completed_boards_iterative_uses_multiple_rounds(self) -> None:
        dataset = SudokuDataset(size=1, blanks=10, seed=7)
        sample = dataset[0]
        digits = sample["digits"].unsqueeze(0)
        givens = sample["givens"].unsqueeze(0)
        targets = sample["targets"]

        class TwoStageOracleModel(torch.nn.Module):
            def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((digits.size(0), 81, 9), dtype=torch.float32, device=digits.device)
                for batch_index in range(digits.size(0)):
                    blank_indices = (digits[batch_index] == 0).nonzero(as_tuple=False).flatten().tolist()
                    for blank_offset, index in enumerate(blank_indices):
                        target_value = int(targets[index].item())
                        if len(blank_indices) > 1 and blank_offset > 0:
                            logits[batch_index, index, :] = 0.0
                        else:
                            logits[batch_index, index, target_value] = 6.0
                return logits

        completed, postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
            TwoStageOracleModel(),
            digits,
            givens,
            torch.device("cpu"),
            mode="iterative",
        )

        self.assertEqual(completed[0].tolist(), (targets + 1).tolist())
        self.assertGreaterEqual(decode_iteration_counts[0], 2)
        self.assertGreater(postprocess_change_counts[0], 0)

    def test_decode_completed_boards_iterative_max_fills_per_round_limits_progress(self) -> None:
        dataset = SudokuDataset(size=1, blanks=10, seed=7)
        sample = dataset[0]
        digits = sample["digits"].unsqueeze(0)
        givens = sample["givens"].unsqueeze(0)
        targets = sample["targets"]
        blank_count = int((sample["digits"] == 0).sum().item())

        class StaticOracleModel(torch.nn.Module):
            def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
                logits = torch.full((digits.size(0), 81, 9), -5.0, dtype=torch.float32, device=digits.device)
                for batch_index in range(digits.size(0)):
                    for index, target_value in enumerate(targets.tolist()):
                        logits[batch_index, index, target_value] = 5.0
                return logits

        completed, postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
            StaticOracleModel(),
            digits,
            givens,
            torch.device("cpu"),
            mode="iterative",
            iterative_max_fills_per_round=1,
        )

        self.assertEqual(completed[0].tolist(), (targets + 1).tolist())
        self.assertEqual(postprocess_change_counts, [0])
        self.assertEqual(decode_iteration_counts, [blank_count])

    def test_decode_completed_boards_iterative_rechecks_consistency_within_round(self) -> None:
        solved_digits = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9] + [4, 5, 6, 7, 8, 9, 1, 2, 3] * 8],
            dtype=torch.long,
        )
        digits = solved_digits.clone()
        digits[0, 1] = 0
        digits[0, 2] = 0
        givens = (digits != 0).to(dtype=torch.float32)
        initial_logits = torch.full((1, 81, 9), -5.0, dtype=torch.float32)
        initial_logits[0, 1, 1] = 5.0
        initial_logits[0, 2, 1] = 5.0
        initial_logits[0, 2, 2] = 4.0

        class TwoStageConflictModel(torch.nn.Module):
            def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
                logits = torch.full((digits.size(0), 81, 9), -5.0, dtype=torch.float32, device=digits.device)
                logits[:, 1, 1] = 5.0
                logits[:, 2, 1] = 5.0
                logits[:, 2, 2] = 4.0
                for batch_index in range(digits.size(0)):
                    if digits[batch_index, 1].item() == 2 and digits[batch_index, 2].item() == 0:
                        logits[batch_index, 2, 1] = 1.0
                        logits[batch_index, 2, 2] = 6.0
                return logits

        completed, _postprocess_change_counts, decode_iteration_counts = decode_completed_boards(
            TwoStageConflictModel(),
            digits,
            givens,
            torch.device("cpu"),
            mode="iterative",
            initial_logits=initial_logits,
            iterative_max_fills_per_round=2,
        )

        self.assertEqual(completed[0, 1].item(), 2)
        self.assertEqual(completed[0, 2].item(), 3)
        self.assertGreaterEqual(decode_iteration_counts[0], 2)

    def test_constraint_consistency_penalty_is_zero_for_valid_completed_board(self) -> None:
        dataset = SudokuDataset(size=1, blanks=10, seed=7)
        sample = dataset[0]
        solution_digits = (sample["targets"] + 1).unsqueeze(0)
        logits = torch.zeros((1, 81, 9), dtype=torch.float32)

        penalty = constraint_consistency_penalty(logits, solution_digits)

        self.assertAlmostEqual(penalty.item(), 0.0, places=6)

    def test_constraint_consistency_penalty_is_positive_for_duplicate_prediction(self) -> None:
        digits = torch.zeros((1, 81), dtype=torch.long)
        logits = torch.full((1, 81, 9), -5.0, dtype=torch.float32)
        duplicate_classes = torch.tensor([0] * 9 + [1] * 72, dtype=torch.long)
        logits[0, torch.arange(81), duplicate_classes] = 5.0

        penalty = constraint_consistency_penalty(logits, digits)

        self.assertGreater(penalty.item(), 0.0)

    def test_solve_board_with_scores_validates_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "81 score vectors"):
            solve_board_with_scores([[0] * 9 for _ in range(9)], [[0.0] * 9])

    def test_summarize_board_violations_counts_row_column_and_box_conflicts(self) -> None:
        board = [
            5, 5, 3, 4, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 4, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ]

        summary = summarize_board_violations(board)

        self.assertFalse(summary["is_valid"])
        self.assertGreaterEqual(summary["row_conflicts"], 2)
        self.assertGreaterEqual(summary["col_conflicts"], 1)
        self.assertGreaterEqual(summary["box_conflicts"], 1)
        self.assertEqual(
            summary["total_conflicts"],
            summary["row_conflicts"] + summary["col_conflicts"] + summary["box_conflicts"],
        )

    def test_evaluate_model_reports_mismatch_and_conflict_metrics(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, digits: torch.Tensor, givens: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((digits.size(0), 81, 9), dtype=torch.float32, device=digits.device)
                logits[:, :, 0] = 1.0
                return logits

        dataset = SudokuDataset(size=2, blanks=10, seed=7)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        metrics = evaluate_model(FixedModel(), dataloader, torch.device("cpu"), decode_mode="iterative")

        self.assertIn("mean_mismatch_count", metrics)
        self.assertIn("mean_total_conflicts", metrics)
        self.assertIn("mean_postprocess_change_count", metrics)
        self.assertIn("mean_decode_iteration_count", metrics)
        self.assertGreater(metrics["mean_mismatch_count"], 0.0)
        self.assertGreaterEqual(metrics["mean_total_conflicts"], 0.0)

    def test_load_model_from_checkpoint_restores_model(self) -> None:
        model = SudokuMLP(embed_dim=8, hidden_dim=64, depth=2, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save(payload, checkpoint_path)

            restored_model, restored_payload = load_model_from_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
            )

        self.assertIsInstance(restored_model, SudokuMLP)
        self.assertEqual(restored_payload["config"]["seed"], 7)
        self.assertEqual(restored_model.get_config()["embed_dim"], 8)

    def test_load_model_from_checkpoint_restores_transformer(self) -> None:
        model = SudokuTransformer(embed_dim=32, num_heads=4, depth=2, ff_dim=64, dropout=0.0)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "transformer",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            torch.save(payload, checkpoint_path)

            restored_model, restored_payload = load_model_from_checkpoint(
                checkpoint_path,
                torch.device("cpu"),
            )

        self.assertIsInstance(restored_model, SudokuTransformer)
        self.assertEqual(restored_payload["config"]["seed"], 7)
        self.assertEqual(restored_model.get_config()["num_heads"], 4)

    def test_train_main_supports_dataset_file(self) -> None:
        dataset = SudokuDataset(size=8, blanks=10, seed=7)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            best_checkpoint_path = Path(temp_dir) / "best.pt"
            metrics_path = Path(temp_dir) / "metrics.json"
            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            train_main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--val-size",
                    "2",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "2",
                    "--early-stopping-patience",
                    "1",
                    "--constraint-loss-weight",
                    "0.1",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--best-checkpoint",
                    str(best_checkpoint_path),
                    "--metrics-output",
                    str(metrics_path),
                ]
            )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(best_checkpoint_path.exists())
            self.assertGreaterEqual(len(metrics["epochs"]), 1)
            self.assertIn("run_metadata", metrics)
            self.assertEqual(metrics["run_metadata"]["entrypoint"], "ai.train")
            self.assertEqual(metrics["run_metadata"]["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(metrics["run_metadata"]["dataset_path"], str(dataset_path))
            self.assertEqual(metrics["run_metadata"]["model_type"], "mlp")
            self.assertIn("best_epoch_metrics", metrics)
            self.assertIn("train_constraint_loss", metrics["epochs"][0])
            self.assertEqual(payload["config"]["resolved_train_size"], 6)
            self.assertEqual(payload["config"]["resolved_val_size"], 2)
            self.assertEqual(payload["config"]["best_checkpoint"], str(best_checkpoint_path))
            self.assertEqual(payload["config"]["constraint_loss_weight"], 0.1)

    def test_train_main_supports_transformer_model(self) -> None:
        dataset = SudokuDataset(size=8, blanks=10, seed=7)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            train_main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--val-size",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--model",
                    "transformer",
                    "--transformer-embed-dim",
                    "32",
                    "--transformer-num-heads",
                    "4",
                    "--transformer-depth",
                    "2",
                    "--transformer-ff-dim",
                    "64",
                    "--dropout",
                    "0.0",
                    "--constraint-loss-weight",
                    "0.05",
                    "--checkpoint",
                    str(checkpoint_path),
                ]
            )

            payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.assertEqual(payload["model_type"], "transformer")
            self.assertIn("best_epoch", payload["config"])
            self.assertEqual(payload["config"]["constraint_loss_weight"], 0.05)

    def test_train_main_supports_separate_val_dataset(self) -> None:
        train_dataset = SudokuDataset(size=6, blanks=10, seed=7)
        val_dataset = SudokuDataset(size=2, blanks=10, seed=101)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.jsonl"
            val_path = Path(temp_dir) / "val.jsonl"
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"

            with train_path.open("w", encoding="utf-8") as handle:
                for index in range(len(train_dataset)):
                    handle.write(json.dumps(sample_to_record(train_dataset[index])))
                    handle.write("\n")

            with val_path.open("w", encoding="utf-8") as handle:
                for index in range(len(val_dataset)):
                    handle.write(json.dumps(sample_to_record(val_dataset[index])))
                    handle.write("\n")

            train_main(
                [
                    "--dataset",
                    str(train_path),
                    "--val-dataset",
                    str(val_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--checkpoint",
                    str(checkpoint_path),
                ]
            )

            payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(payload["config"]["resolved_train_size"], 6)
            self.assertEqual(payload["config"]["resolved_val_size"], 2)

    def test_eval_main_supports_dataset_file(self) -> None:
        dataset = SudokuDataset(size=4, blanks=10, seed=7)
        model = SudokuMLP()
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "model_config": model.get_config(),
            "config": {"seed": 7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            report_path = Path(temp_dir) / "eval_report.json"

            with dataset_path.open("w", encoding="utf-8") as handle:
                for index in range(len(dataset)):
                    handle.write(json.dumps(sample_to_record(dataset[index])))
                    handle.write("\n")

            torch.save(payload, checkpoint_path)

            eval_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_path),
                    "--batch-size",
                    "2",
                    "--decode-preset",
                    "production_pure",
                    "--report",
                    str(report_path),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("run_metadata", report)
        self.assertEqual(report["run_metadata"]["entrypoint"], "ai.eval")
        self.assertEqual(report["run_metadata"]["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(report["run_metadata"]["dataset_path"], str(dataset_path))
        self.assertEqual(report["run_metadata"]["decode_preset"], "production_pure")
        self.assertEqual(report["run_metadata"]["decode_mode"], "iterative")
        self.assertIn("metrics", report)
        self.assertIn("mean_total_conflicts", report["metrics"])
        self.assertIn("mean_postprocess_change_count", report["metrics"])
        self.assertIn("mean_decode_iteration_count", report["metrics"])
        self.assertEqual(report["evaluation_config"]["dataset"], str(dataset_path))
        self.assertEqual(report["evaluation_config"]["decode_preset"], "production_pure")
        self.assertEqual(report["evaluation_config"]["preset_profile"], "accuracy_oriented")
        self.assertEqual(report["evaluation_config"]["decode_mode"], "iterative")
        self.assertEqual(report["evaluation_config"]["iterative_threshold"], 0.75)
        self.assertEqual(report["evaluation_config"]["iterative_max_fills_per_round"], 2)

    def test_plot_main_supports_training_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "train_metrics.json"
            output_path = Path(temp_dir) / "train_metrics.png"
            input_path.write_text(
                json.dumps(
                    {
                        "epochs": [
                            {
                                "epoch": 1,
                                "train_loss": 2.2,
                                "blank_cell_accuracy": 0.1,
                                "board_solved_rate": 0.0,
                                "valid_board_rate": 0.0,
                            },
                            {
                                "epoch": 2,
                                "train_loss": 2.0,
                                "blank_cell_accuracy": 0.15,
                                "board_solved_rate": 0.01,
                                "valid_board_rate": 0.02,
                            },
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            plot_main(["--input", str(input_path), "--output", str(output_path)])

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_main_supports_evaluation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "eval_report.json"
            output_path = Path(temp_dir) / "eval_report.png"
            input_path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "blank_cell_accuracy": 0.12,
                            "board_solved_rate": 0.0,
                            "valid_board_rate": 0.03,
                            "mean_postprocess_change_count": 1.5,
                            "mean_decode_iteration_count": 2.0,
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            plot_main(["--input", str(input_path), "--output", str(output_path)])

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_flat_to_board_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 81 values"):
            flat_to_board([1, 2, 3])


if __name__ == "__main__":
    unittest.main()







