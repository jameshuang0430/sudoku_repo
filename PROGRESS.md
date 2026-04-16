# Progress Log

## 2026-04-16 Transformer Upgrade

### Current Progress
- Completed deterministic solver, dataset export, fixed `train/val/test` manifests, training, evaluation, and PNG plotting workflows.
- Verified baseline MLP training and test evaluation end to end.
- Added a `SudokuTransformer` baseline, model factory, checkpoint support for multiple model types, and `--model transformer` training support.
- Added Transformer forward, checkpoint, and training-path tests.
- Verified the Transformer path with a smoke training run and a full regression pass.

### Issues Encountered
- `functions.apply_patch` failed on Windows with `windows sandbox: setup refresh failed with status exit code: 1`.
- A parallel read checked `PROGRESS.md` before the file write completed.
- PyTorch emitted a nested-tensor warning from `TransformerEncoder` when `norm_first=True` was enabled.
- A direct `python -m unittest tests.test_ai...` invocation failed because this repo currently relies on `discover -s tests` instead of package-style test imports.

### Resolution
- Switched to bounded PowerShell file writes inside the repo as a fallback, while keeping changes limited to the intended files.
- Switched progress-log updates back to serial commands when read-after-write ordering matters.
- Removed `norm_first=True` from the encoder layer so the Transformer path runs cleanly under the current PyTorch build.
- Switched verification back to the repo's stable `python -m unittest discover -s tests -v` command.

### Completed Segment
- Transformer baseline is now implemented, tested, warning-free, and smoke-trained successfully.

### Next Steps
- Commit the Transformer segment.
- Run a full Transformer training/evaluation comparison against the existing MLP baseline.
