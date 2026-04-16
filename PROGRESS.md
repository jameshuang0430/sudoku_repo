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

### Issues Encountered
- `functions.apply_patch` failed on Windows with `windows sandbox: setup refresh failed with status exit code: 1`.
- A parallel read checked `PROGRESS.md` before the file write completed.
- PyTorch emitted a nested-tensor warning from `TransformerEncoder` when `norm_first=True` was enabled.
- A direct `python -m unittest tests.test_ai...` invocation failed because this repo currently relies on `discover -s tests` instead of package-style test imports.
- The first formal Transformer baseline underperformed the existing MLP baseline on the same 5-epoch, 512-sample setup.
- Even after scaling the Transformer experiment, `board_solved_rate` and `valid_board_rate` remained at `0.0000`.

### Resolution
- Switched to bounded PowerShell file writes inside the repo as a fallback, while keeping changes limited to the intended files.
- Switched progress-log updates back to serial commands when read-after-write ordering matters.
- Removed `norm_first=True` from the encoder layer so the Transformer path runs cleanly under the current PyTorch build.
- Switched verification back to the repo's stable `python -m unittest discover -s tests -v` command.
- Logged the first Transformer result as a baseline comparison point instead of assuming the architecture change alone would help immediately.
- Increased scale to test whether the bottleneck was data volume rather than architecture wiring.

### Completed Segment
- Transformer baseline is implemented, tested, warning-free, and formally compared against the current MLP baseline.
- Larger-scale Transformer training/evaluation has now been run and logged.

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

### Interpretation
- The larger Transformer clearly learns cell-level prediction much better than the small-scale runs.
- The remaining bottleneck is global consistency: the model predicts many correct cells but still fails to produce fully valid Sudoku boards.
- The next upgrade should target constraints and decoding behavior, not just raw scale.

### Next Steps
- Commit the larger Transformer experiment segment.
- Add richer evaluation for constraint violations or mismatch distributions.
- Consider iterative decoding, constraint-aware loss terms, or post-processing with the solver.
