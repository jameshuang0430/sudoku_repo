import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai.release_gate as release_gate_module
from ai.release_gate import main as release_gate_main


class ReleaseGateTests(unittest.TestCase):
    def test_release_gate_compare_mode_uses_smoke_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "model.pt"
            dataset_path = temp_path / "test.jsonl"
            baseline_path = temp_path / "smoke_baseline.json"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            dataset_path.write_text("{}\n", encoding="utf-8")
            baseline_path.write_text("{}", encoding="utf-8")

            smoke_profile = release_gate_module.RELEASE_GATE_PROFILES["smoke"]
            patched_profile = release_gate_module.ReleaseGateProfile(
                batch_size=smoke_profile.batch_size,
                benchmark_max_samples=smoke_profile.benchmark_max_samples,
                benchmark_warmup_batches=smoke_profile.benchmark_warmup_batches,
                benchmark_repeats=smoke_profile.benchmark_repeats,
                compare_report=temp_path / "smoke_compare.json",
                baseline_report=baseline_path,
                min_production_fast_solved_rate=smoke_profile.min_production_fast_solved_rate,
                min_production_pure_solved_rate=smoke_profile.min_production_pure_solved_rate,
                min_research_raw_blank_cell_accuracy=smoke_profile.min_research_raw_blank_cell_accuracy,
                max_production_fast_board_ms=smoke_profile.max_production_fast_board_ms,
                max_production_pure_board_ms=smoke_profile.max_production_pure_board_ms,
                max_production_fast_solved_rate_drop=smoke_profile.max_production_fast_solved_rate_drop,
                max_production_pure_solved_rate_drop=smoke_profile.max_production_pure_solved_rate_drop,
                max_production_fast_board_ms_increase=smoke_profile.max_production_fast_board_ms_increase,
                max_production_pure_board_ms_increase=smoke_profile.max_production_pure_board_ms_increase,
            )

            with patch.dict(
                release_gate_module.RELEASE_GATE_PROFILES,
                {"smoke": patched_profile},
                clear=False,
            ):
                with patch.object(release_gate_module, "DEFAULT_PRODUCTION_CHECKPOINT", checkpoint_path):
                    with patch.object(release_gate_module, "DEFAULT_RELEASE_DATASET", dataset_path):
                        with patch.object(release_gate_module, "release_check_main") as release_check_main_mock:
                            release_gate_main([])

        forwarded = release_check_main_mock.call_args.args[0]
        self.assertIn("--baseline-report", forwarded)
        self.assertIn(str(baseline_path), forwarded)
        self.assertIn("--batch-size", forwarded)
        self.assertIn(str(patched_profile.batch_size), forwarded)
        self.assertIn("--benchmark-max-samples", forwarded)
        self.assertIn(str(patched_profile.benchmark_max_samples), forwarded)
        self.assertIn("--max-production-fast-solved-rate-drop", forwarded)
        self.assertIn(str(patched_profile.max_production_fast_solved_rate_drop), forwarded)
        self.assertIn("--report", forwarded)
        self.assertIn(str(patched_profile.compare_report), forwarded)

    def test_release_gate_baseline_mode_uses_full_baseline_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "model.pt"
            dataset_path = temp_path / "test.jsonl"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            dataset_path.write_text("{}\n", encoding="utf-8")

            full_profile = release_gate_module.RELEASE_GATE_PROFILES["full"]
            patched_profile = release_gate_module.ReleaseGateProfile(
                batch_size=full_profile.batch_size,
                benchmark_max_samples=full_profile.benchmark_max_samples,
                benchmark_warmup_batches=full_profile.benchmark_warmup_batches,
                benchmark_repeats=full_profile.benchmark_repeats,
                compare_report=temp_path / "full_compare.json",
                baseline_report=temp_path / "full_baseline.json",
                min_production_fast_solved_rate=full_profile.min_production_fast_solved_rate,
                min_production_pure_solved_rate=full_profile.min_production_pure_solved_rate,
                min_research_raw_blank_cell_accuracy=full_profile.min_research_raw_blank_cell_accuracy,
                max_production_fast_board_ms=full_profile.max_production_fast_board_ms,
                max_production_pure_board_ms=full_profile.max_production_pure_board_ms,
                max_production_fast_solved_rate_drop=full_profile.max_production_fast_solved_rate_drop,
                max_production_pure_solved_rate_drop=full_profile.max_production_pure_solved_rate_drop,
                max_production_fast_board_ms_increase=full_profile.max_production_fast_board_ms_increase,
                max_production_pure_board_ms_increase=full_profile.max_production_pure_board_ms_increase,
            )

            with patch.dict(
                release_gate_module.RELEASE_GATE_PROFILES,
                {"full": patched_profile},
                clear=False,
            ):
                with patch.object(release_gate_module, "DEFAULT_PRODUCTION_CHECKPOINT", checkpoint_path):
                    with patch.object(release_gate_module, "DEFAULT_RELEASE_DATASET", dataset_path):
                        with patch.object(release_gate_module, "release_check_main") as release_check_main_mock:
                            release_gate_main(["--profile", "full", "--mode", "baseline"])

        forwarded = release_check_main_mock.call_args.args[0]
        self.assertIn("--batch-size", forwarded)
        self.assertIn(str(patched_profile.batch_size), forwarded)
        self.assertIn("--benchmark-max-samples", forwarded)
        self.assertIn(str(patched_profile.benchmark_max_samples), forwarded)
        self.assertIn("--report", forwarded)
        self.assertIn(str(patched_profile.baseline_report), forwarded)
        self.assertNotIn("--baseline-report", forwarded)
        self.assertNotIn("--max-production-fast-solved-rate-drop", forwarded)


if __name__ == "__main__":
    unittest.main()
