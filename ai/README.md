# AI Stage Overview

This directory now contains the current model-training, decoding, evaluation, inference, and product-wrapper tooling for the Sudoku project.

## What is here

- `dataset.py`
  - Builds deterministic puzzle samples and exported records.
- `model.py`
  - Contains the baseline MLP and the current Transformer path.
- `train.py`
  - Supports generated-data training, manifest-backed training, early stopping, best-checkpoint saving, and optional constraint loss.
- `decode.py`
  - Implements raw argmax, iterative refinement, and solver-guided decoding.
- `presets.py`
  - Central registry for named decode presets.
- `infer.py`
  - Low-level single-puzzle inference CLI.
  - Defaults to `production_fast` when no preset is specified.
- `product.py`
  - Thin production wrapper around `ai.infer`.
  - Hides the current best checkpoint path and exposes `fast` / `pure` product presets.
- `eval.py`
  - Batch evaluation CLI with board-level metrics and decode-aware reporting.
- `benchmark.py`
  - Measures latency across decode presets and batch sizes.
- `release_check.py`
  - Runs the standard preset comparison plus release gates and optional baseline-regression checks.
- `release_gate.py`
  - Thin wrapper that packages fixed `smoke` and `full` release-gate profiles around `ai.release_check`.
- `analyze_errors.py`
  - Prints and optionally saves failure cases under a chosen decode configuration.
- `export_dataset.py`
  - Writes deterministic JSONL datasets.
- `plot_results.py`
  - Turns metrics JSON into PNG plots.

## Current preset model

Recommended user-facing presets:

- `production_fast`
  - Solver-guided exact repair.
  - Best current latency / accuracy tradeoff.
- `production_pure`
  - Strict iterative decoding.
  - Best current non-solver accuracy.
- `research_raw`
  - Raw argmax baseline for debugging.
- `research_iterative`
  - Unrestricted iterative baseline for comparison.

## Recommended commands

Default product inference:

```powershell
python -m ai.product --file data\dataset\train\puzzle_00001.txt
```

Explicit pure-model inference:

```powershell
python -m ai.product --preset pure --file data\dataset\train\puzzle_00001.txt
```

Low-level inference with explicit checkpoint control:

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --file data\dataset\train\puzzle_00001.txt --decode-preset production_fast
```

Product-fast evaluation:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --decode-preset production_fast --report ai\reports\production_fast_eval.json
```

Latency benchmark:

```powershell
python -m ai.benchmark --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --batch-sizes 1 32 --decode-presets research_raw research_iterative production_pure production_fast --max-samples 128 --repeats 3 --report ai\reports\transformer_large_generalization_latency.json
```

Release check:

```powershell
python -m ai.release_check --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --report ai\reports\release_check.json
```

Use `--baseline-report ai\reports\release_check.json` together with thresholds such as `--max-production-fast-solved-rate-drop` or `--max-production-fast-board-ms-increase` when you want to gate regressions against a saved prior run.

Release-gate wrapper:

```powershell
python -m ai.release_gate --profile smoke
python -m ai.release_gate --profile full
```

Current recommended release baseline:

```powershell
python -m ai.release_check --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --batch-size 1 --benchmark-max-samples 128 --benchmark-warmup-batches 1 --benchmark-repeats 3 --report ai\reports\release_check_generalization_baseline.json
```

Recommended future gate policy:
- baseline report: `ai\reports\release_check_generalization_baseline.json`
- `production_fast`: `min_solved_rate=0.999`, `max_board_ms=3.0`, `max_solved_rate_drop=0.001`, `max_board_ms_increase=1.0`
- `production_pure`: `min_solved_rate=0.99`, `max_board_ms=30.0`, `max_solved_rate_drop=0.01`, `max_board_ms_increase=6.0`
- `research_raw`: `min_blank_cell_accuracy=0.84`

Current recommended full-manifest baseline:

```powershell
python -m ai.release_gate --profile full --mode baseline
```

Recommended full-gate policy:
- baseline report: `ai\reports\release_check_generalization_full_baseline.json`
- `production_fast`: `min_solved_rate=0.999`, `max_board_ms=2.0`, `max_solved_rate_drop=0.001`, `max_board_ms_increase=1.0`
- `production_pure`: `min_solved_rate=0.995`, `max_board_ms=30.0`, `max_solved_rate_drop=0.01`, `max_board_ms_increase=6.0`
- `research_raw`: `min_blank_cell_accuracy=0.84`

## How to think about the current system

- Raw model quality and product quality are different questions.
- Raw argmax is still useful for measuring the model itself, but it is not the current recommended end-user path.
- `python -m ai.product` is the practical default today because it hides the checkpoint path and starts from the current recommended preset.
- `production_fast` is the practical default today because it gives exact repaired outputs with much better latency than strict iterative decoding on the measured CPU setup.
- `production_pure` is the strongest non-solver path and should be treated as the accuracy-first research or purity option.

## Known gaps

- The public docs should state even more explicitly when users should choose `fast` versus `pure`.
- Local checkpoints and datasets are runtime artifacts and should generally stay out of git.
- Future model work should be judged by solved-board rate and conflict reduction, not blank-cell accuracy alone.
