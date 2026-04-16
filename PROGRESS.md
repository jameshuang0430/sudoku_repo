# Progress Log

## 2026-04-16 Transformer Upgrade

### Current Progress
- Completed deterministic solver, dataset export, fixed `train/val/test` manifests, training, evaluation, and PNG plotting workflows.
- Verified baseline MLP training and test evaluation end to end.
- Added a `SudokuTransformer` baseline, model factory, checkpoint support for multiple model types, and `--model transformer` training support.
- Added Transformer forward, checkpoint, and training-path tests.
- Verified the Transformer path with a smoke training run and a full regression pass.
- Ran a formal Transformer baseline experiment on the same fixed `train/val/test` split used by the MLP baseline.
- Ran a substantially larger Transformer experiment on a larger fixed split.
- Upgraded evaluation so it now reports mismatch counts and row/column/box conflict summaries.

### Issues Encountered
- `functions.apply_patch` failed on Windows with `windows sandbox: setup refresh failed with status exit code: 1`.
- A parallel read checked `PROGRESS.md` before the file write completed.
- PyTorch emitted a nested-tensor warning from `TransformerEncoder` when `norm_first=True` was enabled.
- A direct `python -m unittest tests.test_ai...` invocation failed because this repo currently relies on `discover -s tests` instead of package-style test imports.
- The first formal Transformer baseline underperformed the existing MLP baseline on the same 5-epoch, 512-sample setup.
- Even after scaling the Transformer experiment, `board_solved_rate` and `valid_board_rate` remained at `0.0000`.
- When training from `--dataset` / `--val-dataset`, the saved checkpoint metadata still reports the CLI default `train_size` / `val_size` fields rather than the resolved dataset lengths.

### Resolution
- Switched to bounded PowerShell file writes inside the repo as a fallback, while keeping changes limited to the intended files.
- Switched progress-log updates back to serial commands when read-after-write ordering matters.
- Removed `norm_first=True` from the encoder layer so the Transformer path runs cleanly under the current PyTorch build.
- Switched verification back to the repo's stable `python -m unittest discover -s tests -v` command.
- Logged the first Transformer result as a baseline comparison point instead of assuming the architecture change alone would help immediately.
- Increased scale to test whether the bottleneck was data volume rather than architecture wiring.
- Added conflict-level evaluation so the legality failure is now measurable instead of being a binary `valid` / `invalid` outcome only.
- Deferred the checkpoint metadata mismatch as a separate cleanup item so this richer-evaluation segment stays focused.

### Completed Segment
- Transformer baseline is implemented, tested, warning-free, and formally compared against the current MLP baseline.
- Larger-scale Transformer training/evaluation has now been run and logged.
- Constraint-aware evaluation metrics are now implemented and verified.

### Formal Comparison
- MLP validation blank-cell accuracy after 5 epochs: `0.1217`
- MLP test blank-cell accuracy: `0.1145`
- Transformer validation blank-cell accuracy after 5 epochs: `0.1063`
- Transformer test blank-cell accuracy: `0.1086`
- Both models still have `board_solved_rate=0.0000` and `valid_board_rate=0.0000` on this setup.

### Larger Transformer Experiment
- Scale:
  - `train=4096`
  - `val=512`
  - `test=512`
  - `epochs=10`
  - `batch_size=32`
- Validation blank-cell accuracy after 10 epochs: `0.8139`
- Test blank-cell accuracy: `0.8067`
- Validation `board_solved_rate=0.0000`, `valid_board_rate=0.0000`
- Test `board_solved_rate=0.0000`, `valid_board_rate=0.0000`

### New Constraint Metrics On Large Test Split
- `mean_mismatch_count=7.73`
- `mean_row_conflicts=5.02`
- `mean_col_conflicts=5.45`
- `mean_box_conflicts=4.48`
- `mean_total_conflicts=14.94`
- These new metrics confirm that the model is usually close on many cells, but still breaks multiple Sudoku constraints per board.

### Interpretation
- The larger Transformer clearly learns cell-level prediction much better than the small-scale runs.
- The remaining bottleneck is global consistency: the model predicts many correct cells but still fails to produce fully valid Sudoku boards.
- The next upgrade should target constraints and decoding behavior, not just raw scale.

### Next Steps
- Commit the richer evaluation segment.
- Consider iterative decoding, constraint-aware loss terms, richer error analysis, or post-processing with the solver.
- Fix checkpoint metadata so resolved dataset sizes are preserved when training from exported manifests.

## 2026-04-16 Hybrid Decoding Segment

### Current Progress
- Added `solver.solve_board_with_scores()` so the exact solver can be reused as a guided decoder rather than only as a standalone CLI utility.
- Added `ai/decode.py` with switchable `argmax` and `solver_guided` decode modes.
- Updated `ai.eval` and `ai.analyze_errors` so both tools can compare raw model outputs against solver-guided post-processing on the same checkpoint.
- Extended evaluation reports and PNG plots to include `mean_postprocess_change_count`.
- Verified the new path with the full repo test suite and a real run on the large Transformer checkpoint.

### Issues Encountered
- A solver-guided decoder can make unique generated datasets look trivially perfect because the solver can always recover the one valid completion from the givens.
- The earlier conflict metrics alone were not enough to show how much the solver had to override the model.

### Resolution
- Kept decoding explicit with `--decode-mode argmax|solver_guided` so raw prediction quality and repaired-board quality stay separable in reports.
- Added `mean_postprocess_change_count` so the hybrid path now reports how many blank-cell argmax guesses were changed by the solver on average.
- Documented the decode-mode caveat in the README instead of presenting solver-guided numbers as if they were raw model outputs.

### Completed Segment
- Hybrid decoding is now implemented, test-covered, documented, and reported.
- The repo can now measure both raw Sudoku consistency failures and a solver-repaired upper bound for the same checkpoint.

### Large Transformer Hybrid Evaluation
- Command:
  - `python -m ai.eval --checkpoint ai\checkpoints\transformer_large.pt --dataset data\manifests_large\test.jsonl --batch-size 32 --decode-mode solver_guided --report ai\reports\transformer_large_solver_guided_test_metrics.json`
- Results:
  - `blank_cell_accuracy=1.0000`
  - `board_solved_rate=1.0000`
  - `valid_board_rate=1.0000`
  - `mean_mismatch_count=0.00`
  - `mean_total_conflicts=0.00`
  - `mean_postprocess_change_count=7.73`
- Interpretation:
  - The solver-guided path fully repairs the large Transformer outputs on this fixed unique-solution test split.
  - The average `7.73` post-process changes lines up with the earlier raw `mean_mismatch_count`, which confirms the solver is repairing the same last-mile mistakes rather than the model already producing legal boards on its own.

### Next Steps
- Commit the hybrid decoding segment.
- Decide whether to keep solver-guided decoding as an evaluation-only upper bound or expose it as the default inference path.
- If raw model quality still matters most, move next to iterative decoding or constraint-aware training instead of relying only on solver repair.

## 2026-04-16 Iterative Decoding Segment

### Current Progress
- Added `iterative` as a third decode mode beside `argmax` and `solver_guided`.
- Refactored decoding so the evaluation and error-analysis tools can re-run the model during inference instead of only consuming one fixed logits tensor.
- Added `mean_decode_iteration_count` so iterative decoding reports how many refinement rounds were used on average.
- Verified the iterative path with new tests plus a real run on the large Transformer checkpoint.

### Issues Encountered
- The first version of the new test expected iterative decoding to keep `postprocess_change_count=0`, but the dummy model was intentionally changing low-confidence blanks across rounds, so that expectation was wrong.
- A parallel `plot_results` run attempted to open the iterative report before the JSON file had finished writing.

### Resolution
- Corrected the test to check the behavior that actually matters: the board is solved correctly and iterative decoding takes multiple rounds.
- Switched the report generation and plotting back to serial execution when read-after-write ordering matters.
- Kept `mean_postprocess_change_count` and added `mean_decode_iteration_count` so iterative decoding can be measured both as output drift from raw argmax and as refinement depth.

### Completed Segment
- Iterative decoding is now implemented, test-covered, documented, and measurable.
- The repo can now compare three inference modes on the same checkpoint: raw argmax, non-solver iterative refinement, and exact solver repair.

### Large Transformer Iterative Evaluation
- Command:
  - `python -m ai.eval --checkpoint ai\checkpoints\transformer_large.pt --dataset data\manifests_large\test.jsonl --batch-size 32 --decode-mode iterative --report ai\reports\transformer_large_iterative_test_metrics.json`
- Results:
  - `blank_cell_accuracy=0.8866`
  - `board_solved_rate=0.2480`
  - `valid_board_rate=0.2480`
  - `mean_mismatch_count=4.54`
  - `mean_total_conflicts=3.66`
  - `mean_postprocess_change_count=6.55`
  - `mean_decode_iteration_count=4.58`
- Interpretation:
  - This is a substantial improvement over raw argmax on the same checkpoint, which had `valid_board_rate=0.0000` and `mean_total_conflicts=14.94`.
  - Iterative decoding does not match the exact-solver upper bound, but it proves that a large part of the remaining quality gap is in inference behavior rather than only in the learned weights.

### Next Steps
- Commit the iterative decoding segment.
- Decide whether to expose iterative decoding as the default non-solver inference path.
- If the next goal is to raise raw-model quality further, move next to constraint-aware training or checkpoint-metadata cleanup.
