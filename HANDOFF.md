# Handoff Report

## Project Summary

This repo has two layers:

1. `solver/`: deterministic Sudoku validation, solving, generation, and CLI tooling.
2. `ai/`: a PyTorch training, decoding, benchmarking, and inference stack built on top of the solver.

The project is no longer in the early "replace the MLP with a Transformer" stage. The current state is a product-oriented Transformer pipeline with multiple decode paths, shared decode presets, benchmark tooling, and a thin production wrapper command that hides the checkpoint path for the recommended end-user workflow.

## Current Product Position

### Recommended entrypoint

Use the thin wrapper command for normal single-puzzle inference:

```powershell
python -m ai.product --file data\dataset\train\puzzle_00001.txt
```

Optional accuracy-first non-solver variant:

```powershell
python -m ai.product --preset pure --file data\dataset\train\puzzle_00001.txt
```

### Recommended presets

- `production_fast`
  - Solver-guided exact repair.
  - Best current latency / accuracy tradeoff.
  - This is the current default recommendation and the wrapper's default behavior.
- `production_pure`
  - Strict iterative decoding.
  - Best current non-solver accuracy.
  - Much slower than `production_fast`.
- `research_raw`
  - Raw argmax baseline.
  - Useful for debugging raw model quality.
- `research_iterative`
  - Older unrestricted iterative baseline kept for comparison.

### Current interpretation

- The raw model is useful but still weak without decode help.
- Product quality currently comes from the combination of a good Transformer checkpoint plus a decode preset, not from raw argmax alone.
- If exact solver post-processing is acceptable, `production_fast` remains the cleanest current deployment choice.

## Current State

### Solver layer

Implemented under `solver/`:

- `solver/validator.py`
  - Validates board shape, value range, and Sudoku legality.
  - Renders boards for readable CLI output.
- `solver/solver.py`
  - Backtracking exact solver with solution counting.
  - Exposes score-aware solving support used by solver-guided decoding.
- `solver/generator.py`
  - Generates solved boards and puzzle/solution pairs.
  - Supports uniqueness-preserving generation.
- `solver/cli.py`
  - Supports `generate`, `solve`, and dataset export workflows.
  - Accepts puzzle input from string, file, and stdin.
  - Accepts pretty-printed boards as well as compact 81-character strings.

### AI layer

Implemented under `ai/`:

- `ai/dataset.py`
  - Builds deterministic training samples and exported records.
- `ai/model.py`
  - Supports both the baseline MLP and the Transformer path.
- `ai/train.py`
  - Supports manifest-backed training, best-checkpoint saving, early stopping, and optional constraint loss.
- `ai/decode.py`
  - Supports raw argmax, iterative refinement, and solver-guided repair.
- `ai/presets.py`
  - Centralizes shared decode presets used across CLI tools.
- `ai/infer.py`
  - Low-level single-puzzle inference CLI.
  - Defaults to `production_fast`.
- `ai/product.py`
  - Thin wrapper around `ai.infer`.
  - Hides the current best production checkpoint path.
  - Exposes `--preset fast|pure` for the user-facing path.
- `ai/eval.py`
  - Reports cell accuracy, solved-board rate, validity, conflict counts, postprocess changes, and decode iteration counts.
- `ai/benchmark.py`
  - Benchmarks decode latency across presets and batch sizes.
- `ai/release_check.py`
  - Runs the standard preset comparison plus release gates and optional baseline-regression checks.
- `ai/analyze_errors.py`
  - Compares failure cases under a chosen decode configuration.
- `ai/export_dataset.py`
  - Exports deterministic JSONL datasets.
- `ai/plot_results.py`
  - Renders PNG plots from metrics reports.

## Most Important Verified Results

### Test status

Latest known full-suite result:

```powershell
python -m unittest discover -s tests -v
```

- `49` tests passed before the wrapper segment.
- The wrapper segment adds dedicated `ai.product` coverage on top of the existing inference tests.

### Generalization snapshot on fresh large split

Checkpoint:
- `ai\checkpoints\transformer_large_current.best.pt`

Dataset:
- `data\manifests_generalization\test.jsonl`

Results:
- Raw argmax (`research_raw` style):
  - `blank_cell_accuracy=0.8437`
  - `board_solved_rate=0.0039`
  - `valid_board_rate=0.0039`
  - `mean_total_conflicts=11.59`
- Strict iterative (`production_pure`):
  - `blank_cell_accuracy=0.9991`
  - `board_solved_rate=0.9980`
  - `valid_board_rate=0.9980`
  - `mean_total_conflicts=0.02`
- Solver-guided (`production_fast`):
  - Intended product recommendation because it keeps exact repair with better latency than strict iterative on this CPU setup.

### Latency snapshot

Benchmark command:

```powershell
python -m ai.benchmark --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --batch-sizes 1 32 --decode-presets research_raw research_iterative production_pure production_fast --max-samples 128 --repeats 3 --report ai\reports\transformer_large_generalization_latency.json
```

Key results on CPU:
- `research_raw`, `batch_size=1`: `1.05 ms` per board
- `production_fast`, `batch_size=1`: `1.54 ms` per board
- `production_pure`, `batch_size=1`: `23.15 ms` per board

Interpretation:
- `production_fast` is dramatically faster than `production_pure` while still giving exact repaired outputs.
- `production_pure` remains valuable as the strongest non-solver path.

## Recommended Commands

### Default single-puzzle inference

```powershell
python -m ai.product --file data\dataset\train\puzzle_00001.txt
```

### Explicit non-solver production inference

```powershell
python -m ai.product --preset pure --file data\dataset\train\puzzle_00001.txt
```

### Low-level inference with explicit checkpoint control

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --file data\dataset\train\puzzle_00001.txt --decode-preset production_fast
```

### Product-fast evaluation

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --decode-preset production_fast --report ai\reports\production_fast_eval.json
```

### Product-pure evaluation

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --decode-preset production_pure --report ai\reports\production_pure_eval.json
```

### Standard release check

```powershell
python -m ai.release_check --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --report ai\reports\release_check.json
```

To enforce regression limits against a saved prior run, add `--baseline-report ai\reports\release_check.json` plus thresholds such as `--max-production-fast-solved-rate-drop` or `--max-production-fast-board-ms-increase`.

### Official current baseline

Baseline artifact:
- `ai\reports\release_check_generalization_baseline.json`

Baseline command:

```powershell
python -m ai.release_check --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --batch-size 1 --benchmark-max-samples 128 --benchmark-warmup-batches 1 --benchmark-repeats 3 --report ai\reports\release_check_generalization_baseline.json
```

Recommended thresholds for comparison runs:
- `production_fast`: `min_solved_rate=0.999`, `max_board_ms=3.0`, `max_solved_rate_drop=0.001`, `max_board_ms_increase=1.0`
- `production_pure`: `min_solved_rate=0.99`, `max_board_ms=30.0`, `max_solved_rate_drop=0.01`, `max_board_ms_increase=6.0`
- `research_raw`: `min_blank_cell_accuracy=0.84`

## Repo Artifact Policy

Current intended policy:

- Commit source, tests, top-level docs, and milestone reports that matter for comparison.
- Keep local datasets, checkpoints, and cache directories out of git.
- Treat `data/` and `ai/checkpoints/` as runtime artifacts unless there is a specific reason to publish a snapshot.

## Known Gaps

- The repo still has untracked source/docs files that should be committed together so the documented product state matches the actual code state.
- Raw argmax quality is still much weaker than the product presets.
- The public docs should explain even more directly when users should choose `fast` versus `pure`.
- The release-check baseline is now chosen, but the next policy question is whether this 128-sample smoke gate is enough on its own or should be paired with a slower full-manifest comparison.

## Recommended Next Steps

### Highest-value next step

Commit the current untracked source, tests, and docs together after re-running verification.

Reason:
- The wrapper, preset system, benchmark flow, and product-oriented docs now form a coherent state.
- The immediate risk is repo drift, not missing core product behavior.

### After that

- Keep `production_fast` as the default path unless there is a strong non-solver-only requirement.
- Decide whether to keep the current 128-sample `ai.release_check` baseline as the only release gate or add a second slower full-manifest gate.
- If raw model quality becomes a real goal, evaluate future work by solved-board rate and conflict reduction, not by blank-cell accuracy alone.

## Notes For The Next Person

- Do not treat the old MLP-vs-Transformer roadmap as current; that work is already behind us.
- The main open question is product packaging clarity and repo stabilization, not whether decode presets work.
- If you need the best current end-user experience, start from `python -m ai.product`.
- If you need the best non-solver behavior for research or purity reasons, use `--preset pure` and accept the latency cost.
