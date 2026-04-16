# Progress Log

## 2026-04-16 Transformer Upgrade

### Current Progress
- Completed deterministic solver, dataset export, fixed `train/val/test` manifests, training, evaluation, and PNG plotting workflows.
- Verified baseline MLP training and test evaluation end to end.
- Added a `SudokuTransformer` baseline, model factory, checkpoint support for multiple model types, and `--model transformer` training support.
- Added Transformer forward, checkpoint, and training-path tests.
- Verified the Transformer path with a smoke training run and a full regression pass.
- Ran a formal Transformer baseline experiment on the same fixed `train/val/test` split used by the MLP baseline.

### Issues Encountered
- `functions.apply_patch` failed on Windows with `windows sandbox: setup refresh failed with status exit code: 1`.
- A parallel read checked `PROGRESS.md` before the file write completed.
- PyTorch emitted a nested-tensor warning from `TransformerEncoder` when `norm_first=True` was enabled.
- A direct `python -m unittest tests.test_ai...` invocation failed because this repo currently relies on `discover -s tests` instead of package-style test imports.
- The first formal Transformer baseline underperformed the existing MLP baseline on the same 5-epoch, 512-sample setup.

### Resolution
- Switched to bounded PowerShell file writes inside the repo as a fallback, while keeping changes limited to the intended files.
- Switched progress-log updates back to serial commands when read-after-write ordering matters.
- Removed `norm_first=True` from the encoder layer so the Transformer path runs cleanly under the current PyTorch build.
- Switched verification back to the repo's stable `python -m unittest discover -s tests -v` command.
- Logged the first Transformer result as a baseline comparison point instead of assuming the architecture change alone would help immediately.

### Completed Segment
- Transformer baseline is implemented, tested, warning-free, and formally compared against the current MLP baseline.

### Formal Comparison
- MLP validation blank-cell accuracy after 5 epochs: `0.1217`
- MLP test blank-cell accuracy: `0.1145`
- Transformer validation blank-cell accuracy after 5 epochs: `0.1063`
- Transformer test blank-cell accuracy: `0.1086`
- Both models still have `board_solved_rate=0.0000` and `valid_board_rate=0.0000` on this setup.

### Interpretation
- The current Transformer configuration is not yet better than the MLP baseline.
- The likely problem is not the model path itself, but the training scale and untuned hyperparameters.
- Next comparisons should increase data volume and/or tune Transformer dimensions, depth, learning rate, and epochs.

### Next Steps
- Commit the formal Transformer comparison segment.
- Run a larger Transformer experiment or a hyperparameter sweep against the same fixed split.
