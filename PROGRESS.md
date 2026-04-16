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

## 2026-04-16 Early Stopping Segment

### Current Progress
- Added validation-driven early stopping to `ai.train`.
- Added separate best-checkpoint saving alongside the final checkpoint.
- Fixed checkpoint metadata so manifest-based training now records resolved train/validation dataset sizes.
- Added best-epoch metadata to the metrics report and checkpoint payload.

### Issues Encountered
- The previous training loop always saved only the final epoch, which made it impossible to tell whether the best validation state had already passed.
- Checkpoints saved from `--dataset` / `--val-dataset` runs still reported the CLI default `train_size` / `val_size`, which made later trace work misleading.

### Resolution
- Introduced `--early-stopping-patience` with validation ranking based on `board_solved_rate`, then `mean_total_conflicts`, then `blank_cell_accuracy`.
- Added `--best-checkpoint` plus a default `.best.pt` path when one is not supplied.
- Stored both requested and resolved dataset sizes in the checkpoint config so manifest-driven runs can be traced correctly.
- Wrote `best_epoch_metrics`, `best_epoch`, `final_epoch`, and `stopped_early` into the saved metadata.

### Completed Segment
- Training now stops when validation progress plateaus instead of always running the full requested epoch count.
- The repo now preserves both the final state and the best validation state for the same run.
- The earlier checkpoint-metadata mismatch for dataset-backed training is now fixed.

### Verification
- `python -m unittest discover -s tests -v`
- Result: `41` tests passed.
- The new training-path tests confirm:
  - best checkpoint files are written,
  - resolved dataset sizes are stored correctly,
  - early stopping triggers when patience is exhausted.

### Next Steps
- Commit the early-stopping segment.
- Use the best checkpoint rather than the final checkpoint in future formal evaluations.
- If the next focus is raw model quality, move next to constraint-aware training.

## 2026-04-16 Current Transformer Training Run

### Current Progress
- Ran a new formal Transformer training job on the fixed `data\manifests\train.jsonl` / `val.jsonl` split using the new early-stopping and best-checkpoint flow.
- Saved the final checkpoint, best checkpoint, JSON metrics report, and PNG charts for the run.
- Evaluated the best checkpoint on the fixed test split with both `argmax` and `iterative` decode modes.

### Issues Encountered
- Even with the improved training loop, the fixed 512-sample training split is still too small for `board_solved_rate` to move off zero under raw argmax decoding.
- Early stopping never triggered on this run because the validation ranking kept improving through conflict reduction and cell accuracy, even though solved-board rate stayed flat at zero.

### Resolution
- Preserved this run as a baseline experiment instead of forcing a larger refactor mid-run.
- Logged both raw `argmax` and `iterative` test results so the gap between learned weights and inference behavior stays measurable.
- Kept the best checkpoint at epoch 20 because validation metrics were still improving under the current ranking rule.

### Completed Segment
- A fresh Transformer run has now been trained and evaluated with the current training stack.
- The repo now has an up-to-date baseline for the smaller fixed split using the new best-checkpoint workflow.

### Training Run Summary
- Command:
  - `python -m ai.train --dataset data\manifests\train.jsonl --val-dataset data\manifests\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 20 --batch-size 32 --early-stopping-patience 5 --checkpoint ai\checkpoints\transformer_current.pt --best-checkpoint ai\checkpoints\transformer_current.best.pt --metrics-output ai\reports\transformer_current_metrics.json`
- Best epoch: `20`
- Validation metrics at best epoch:
  - `blank_cell_accuracy=0.6563`
  - `board_solved_rate=0.0000`
  - `valid_board_rate=0.0000`
  - `mean_total_conflicts=28.45`

### Test Evaluation Summary
- Raw argmax:
  - `blank_cell_accuracy=0.6621`
  - `board_solved_rate=0.0000`
  - `valid_board_rate=0.0000`
  - `mean_mismatch_count=13.52`
  - `mean_total_conflicts=27.90`
- Iterative decoding:
  - `blank_cell_accuracy=0.8016`
  - `board_solved_rate=0.0703`
  - `valid_board_rate=0.0703`
  - `mean_mismatch_count=7.94`
  - `mean_total_conflicts=5.88`
  - `mean_decode_iteration_count=7.10`

### Interpretation
- This run is materially better than the earliest small-split Transformer baseline, but it still has not crossed the threshold where raw argmax can solve full boards reliably.
- Iterative decoding remains the only non-solver path here that produces any non-zero solved-board rate.
- The next improvement should target either more scale or constraint-aware training, because pure supervised cell prediction on this small split is still not enough.

### Next Steps
- Commit this training/evaluation segment.
- Prefer evaluating `ai\checkpoints\transformer_current.best.pt` rather than the final checkpoint.
- If the next run should aim at much higher solved-board rate, increase training scale or move to constraint-aware training.

## 2026-04-17 Larger-Scale Transformer Run

### Current Progress
- Reused the existing larger fixed split in `data\manifests_large` to run a new formal Transformer experiment with the current training stack.
- Saved the final checkpoint, best checkpoint, training metrics report, and both raw/iterative test reports.
- Compared the new large-scale run against the older `transformer_large` baseline on the same split.

### Issues Encountered
- Early stopping still did not trigger on this run because validation metrics kept improving through epoch 20.
- Even at this larger scale, raw `argmax` decoding is still far behind iterative decoding on solved-board rate.

### Resolution
- Kept the full 20-epoch run because validation `board_solved_rate` and conflict metrics were still improving late in training.
- Preserved both raw `argmax` and `iterative` test reports so the inference gap remains explicit.
- Treated this run as the new large-scale baseline for the current training stack, since it uses the newer best-checkpoint and metadata flow.

### Completed Segment
- A larger-scale Transformer experiment has now been run end to end with the current training pipeline.
- The repo now has an updated large-split baseline that is directly comparable to the earlier large experiment.

### Training Run Summary
- Command:
  - `python -m ai.train --dataset data\manifests_large\train.jsonl --val-dataset data\manifests_large\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 20 --batch-size 32 --early-stopping-patience 5 --checkpoint ai\checkpoints\transformer_large_current.pt --best-checkpoint ai\checkpoints\transformer_large_current.best.pt --metrics-output ai\reports\transformer_large_current_metrics.json`
- Resolved scale:
  - `train=4096`
  - `val=512`
  - `test=512`
- Best epoch: `20`
- Validation metrics at best epoch:
  - `blank_cell_accuracy=0.8492`
  - `board_solved_rate=0.0117`
  - `valid_board_rate=0.0117`
  - `mean_total_conflicts=10.98`

### Test Evaluation Summary
- Raw argmax:
  - `blank_cell_accuracy=0.8448`
  - `board_solved_rate=0.0059`
  - `valid_board_rate=0.0059`
  - `mean_mismatch_count=6.21`
  - `mean_total_conflicts=11.68`
- Iterative decoding:
  - `blank_cell_accuracy=0.9063`
  - `board_solved_rate=0.2969`
  - `valid_board_rate=0.2969`
  - `mean_mismatch_count=3.75`
  - `mean_total_conflicts=3.50`
  - `mean_postprocess_change_count=5.17`
  - `mean_decode_iteration_count=4.10`

### Comparison To Earlier Large Baseline
- Earlier large raw argmax test:
  - `blank_cell_accuracy=0.8067`
  - `board_solved_rate=0.0000`
  - `mean_total_conflicts=14.94`
- Current large raw argmax test:
  - `blank_cell_accuracy=0.8448`
  - `board_solved_rate=0.0059`
  - `mean_total_conflicts=11.68`
- Earlier large iterative test:
  - `blank_cell_accuracy=0.8866`
  - `board_solved_rate=0.2480`
  - `mean_total_conflicts=3.66`
- Current large iterative test:
  - `blank_cell_accuracy=0.9063`
  - `board_solved_rate=0.2969`
  - `mean_total_conflicts=3.50`

### Interpretation
- This larger-scale run is a real improvement over both the recent small-split run and the earlier large-split baseline.
- Raw model quality is finally strong enough to produce a non-zero solved-board rate without post-processing on the large test split.
- Iterative decoding remains the most effective non-solver inference path so far, and it is now close to solving 3 out of 10 boards on this split.

### Next Steps
- Commit this larger-scale experiment segment.
- Use `ai\checkpoints\transformer_large_current.best.pt` as the default checkpoint for further comparison work.
- If the next goal is to push solved-board rate substantially higher, move next to either more scale again or constraint-aware training.

## 2026-04-17 Single-Puzzle Inference Segment

### Current Progress
- Added `ai.infer` for single-puzzle checkpoint inference.
- The new CLI supports `--file`, `--puzzle`, and `--stdin` input modes.
- It prints the original puzzle, decoded prediction, and board-validity summary, with optional raw argmax output when useful.
- Added a dedicated `tests/test_infer.py` so inference coverage stays separate from the training stack work.

### Issues Encountered
- The first version of the CLI used `solver.board_to_string()`, which validates boards before rendering them.
- That broke `argmax` inference output when the raw prediction was structurally invalid, which is a normal case for this project.

### Resolution
- Replaced strict rendering with a local `format_board()` helper that can print invalid boards safely.
- Kept the actual legality check explicit via `summarize_board_violations()` instead of hiding it inside board formatting.
- Added dedicated tests for both `--file` and inline `--puzzle` usage against the current best checkpoint.

### Completed Segment
- You can now point a checkpoint directly at a puzzle file and get a readable prediction without building a dataset or running full evaluation.
- The repo now has a lightweight interactive inference entry point for checkpoint sanity checks.

### Verification
- `python -m unittest discover -s tests -p "test_infer.py" -v`
- Result: `2` tests passed.

### Next Steps
- Commit the single-puzzle inference segment.
- If desired, add JSON output mode so `ai.infer` can feed other tooling without parsing terminal text.
- Return to the unfinished constraint-aware training work in a separate clean segment.

## 2026-04-17 Constraint-Aware Training Segment

### Current Progress
- Added optional constraint-aware training via `--constraint-loss-weight`.
- Training now logs `train_ce_loss` and `train_constraint_loss` separately.
- Added tests for the new consistency penalty and for constraint-loss metadata flowing into checkpoints and metrics.
- Ran a formal large-scale constraint-aware Transformer experiment on the fixed large split.

### Issues Encountered
- A small logging bug in the first training-loop revision passed `epoch` twice into `str.format()`, which broke the training-path tests.
- The first real constraint-aware experiment improved structural conflict slightly under iterative decoding, but underperformed the unconstrained large-scale baseline on solved-board rate.

### Resolution
- Fixed the duplicated `epoch` formatting bug and re-ran the full test suite.
- Preserved the constraint-aware run as a comparison point rather than over-tuning the weight immediately.
- Documented the new training flag and separated the constraint-loss statistics in the reports so future sweeps can be compared cleanly.

### Completed Segment
- Constraint-aware training is now implemented, test-covered, documented, and experimentally exercised.
- The repo can now compare unconstrained and constraint-aware large-scale Transformer runs under the same evaluation pipeline.

### Verification
- `python -m unittest discover -s tests -v`
- Result: `45` tests passed.

### Large Constraint-Aware Experiment
- Command:
  - `python -m ai.train --dataset data\manifests_large\train.jsonl --val-dataset data\manifests_large\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 20 --batch-size 32 --early-stopping-patience 5 --constraint-loss-weight 0.05 --checkpoint ai\checkpoints\transformer_large_constraint.pt --best-checkpoint ai\checkpoints\transformer_large_constraint.best.pt --metrics-output ai\reports\transformer_large_constraint_metrics.json`
- Best epoch: `8`
- Stopped early: `true`
- Validation metrics at best epoch:
  - `blank_cell_accuracy=0.8048`
  - `board_solved_rate=0.0020`
  - `valid_board_rate=0.0020`
  - `mean_total_conflicts=15.70`

### Test Evaluation Summary
- Raw argmax:
  - `blank_cell_accuracy=0.7975`
  - `board_solved_rate=0.0000`
  - `valid_board_rate=0.0000`
  - `mean_mismatch_count=8.10`
  - `mean_total_conflicts=16.19`
- Iterative decoding:
  - `blank_cell_accuracy=0.8825`
  - `board_solved_rate=0.2129`
  - `valid_board_rate=0.2129`
  - `mean_mismatch_count=4.70`
  - `mean_total_conflicts=3.38`
  - `mean_postprocess_change_count=6.96`
  - `mean_decode_iteration_count=4.73`

### Comparison To Current Large Baseline
- Current unconstrained large raw argmax test:
  - `blank_cell_accuracy=0.8448`
  - `board_solved_rate=0.0059`
  - `mean_total_conflicts=11.68`
- Constraint-aware large raw argmax test:
  - `blank_cell_accuracy=0.7975`
  - `board_solved_rate=0.0000`
  - `mean_total_conflicts=16.19`
- Current unconstrained large iterative test:
  - `blank_cell_accuracy=0.9063`
  - `board_solved_rate=0.2969`
  - `mean_total_conflicts=3.50`
- Constraint-aware large iterative test:
  - `blank_cell_accuracy=0.8825`
  - `board_solved_rate=0.2129`
  - `mean_total_conflicts=3.38`

### Interpretation
- This first constraint-aware run did not beat the unconstrained large baseline overall.
- It slightly improved iterative conflict totals, but it lost too much on solved-board rate and raw accuracy.
- The current conclusion is that `constraint_loss_weight=0.05` is too blunt for this setup, not that constraint-aware training is fundamentally useless.

### Next Steps
- Commit the constraint-aware training segment.
- If this direction is worth pursuing, run a smaller weight sweep such as `0.005`, `0.01`, and `0.02` instead of increasing the penalty further.
- Keep `ai\checkpoints\transformer_large_current.best.pt` as the default best-performing large checkpoint for now.

## 2026-04-17 Small Constraint-Weight Sweep

### Current Progress
- Ran a follow-up large-scale constraint-aware sweep with smaller weights: `0.005`, `0.01`, and `0.02`.
- Evaluated each best checkpoint on the fixed large test split with both `argmax` and `iterative` decoding.
- Compared all three weights against the current unconstrained large baseline and the earlier `0.05` constraint run.

### Issues Encountered
- Each small-weight run still took the full 20 epochs, so the sweep remained compute-heavy even though the penalties were gentler than `0.05`.
- None of the tested small weights beat the current unconstrained large baseline on solved-board rate.

### Resolution
- Kept the sweep focused on three weights instead of expanding further before seeing the trend.
- Used the same large split and same model hyperparameters for every run so the comparison stays clean.
- Treated `0.005`, `0.01`, and `0.02` as a first local sweep rather than continuing to larger or denser grids immediately.

### Completed Segment
- The repo now has a meaningful first sweep over small constraint-loss weights.
- We now know that lighter penalties are better than `0.05`, but still not better than the unconstrained large baseline overall.

### Sweep Summary
- Baseline unconstrained large test:
  - raw argmax: `blank_cell_accuracy=0.8448`, `board_solved_rate=0.0059`, `mean_total_conflicts=11.68`
  - iterative: `blank_cell_accuracy=0.9063`, `board_solved_rate=0.2969`, `mean_total_conflicts=3.50`
- Constraint `0.005` test:
  - raw argmax: `blank_cell_accuracy=0.8317`, `board_solved_rate=0.0020`, `mean_total_conflicts=12.64`
  - iterative: `blank_cell_accuracy=0.8985`, `board_solved_rate=0.2539`, `mean_total_conflicts=3.79`
- Constraint `0.01` test:
  - raw argmax: `blank_cell_accuracy=0.8286`, `board_solved_rate=0.0000`, `mean_total_conflicts=13.01`
  - iterative: `blank_cell_accuracy=0.9001`, `board_solved_rate=0.2578`, `mean_total_conflicts=3.79`
- Constraint `0.02` test:
  - raw argmax: `blank_cell_accuracy=0.8305`, `board_solved_rate=0.0059`, `mean_total_conflicts=12.84`
  - iterative: `blank_cell_accuracy=0.8987`, `board_solved_rate=0.2578`, `mean_total_conflicts=3.75`
- Earlier constraint `0.05` test:
  - raw argmax: `blank_cell_accuracy=0.7975`, `board_solved_rate=0.0000`, `mean_total_conflicts=16.19`
  - iterative: `blank_cell_accuracy=0.8825`, `board_solved_rate=0.2129`, `mean_total_conflicts=3.38`

### Interpretation
- `0.005` to `0.02` are clearly better than `0.05`, so the first penalty was too strong.
- Among the new runs, `0.02` is the least bad on raw argmax and ties the baseline raw solved-board rate, but it still does not beat the unconstrained baseline on accuracy or conflicts.
- `0.01` and `0.02` slightly improve over `0.005` on iterative solved-board rate, but all three remain below the unconstrained large iterative baseline of `0.2969`.
- The current best overall checkpoint is still `ai\checkpoints\transformer_large_current.best.pt`.

### Next Steps
- Commit the small-weight sweep results.
- If this direction is pursued further, the next most sensible tests are either a very small weight like `0.001` or a different constraint formulation, not simply more weights around `0.05`.
- If the goal is fastest progress on solved-board rate, return focus to scale and decoding rather than this constraint penalty alone.
