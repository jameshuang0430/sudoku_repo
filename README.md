# Sudoku Project

This repo is split into two stages:

1. `solver/`: a reliable Sudoku solver, validator, data generator, and CLI.
2. `ai/`: a future training pipeline built on top of the solver.

## Stage 1 goals

- Validate whether a 9x9 board is structurally valid.
- Solve standard Sudoku puzzles with `0` representing blanks.
- Count solutions to check whether a puzzle is unique.
- Generate solved boards and labeled training pairs.
- Provide a direct CLI for generating and solving puzzles.
- Keep the implementation small enough to study and extend.

## Project layout

```text
solver/
  __init__.py
  cli.py
  generator.py
  solver.py
  validator.py
tests/
  test_ai.py
  test_cli.py
  test_infer.py
  test_solver.py
ai/
  analyze_errors.py
  checkpoint.py
  dataset.py
  decode.py
  eval.py
  export_dataset.py
  infer.py
  model.py
  plot_results.py
  train.py
```

## Product Presets

Recommended product-facing decode presets:
- `production_fast`: solver-guided exact repair with the best current latency / accuracy tradeoff.
- `production_pure`: strict non-solver iterative decoding for the best current pure-model accuracy.
- `research_raw`: raw argmax baseline for debugging and comparison.
- `research_iterative`: unrestricted iterative baseline for research comparison.

Recommended default commands:

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --file data\dataset\train\puzzle_00001.txt
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --decode-preset production_fast --report ai\reports\production_fast_eval.json
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --decode-preset production_pure --report ai\reports\production_pure_eval.json
```

`ai.infer` now defaults to `--decode-preset production_fast` when no preset is specified.
## Quick start

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Generate a puzzle to the terminal:

```powershell
python -m solver.cli generate --blanks 40 --seed 7 --show-solution
```

Generate a puzzle file:

```powershell
python -m solver.cli generate --blanks 40 --seed 7 --output puzzles\puzzle_001.txt
```

Generate both puzzle and solution files:

```powershell
python -m solver.cli generate --blanks 40 --seed 7 --output puzzles\puzzle_001.txt --solution-output puzzles\puzzle_001_solution.txt
```

Batch-export a single JSONL dataset manifest:

```powershell
python -m solver.cli export-dataset --size 640 --blanks 40 --seed 7 --output-dir data\puzzles --manifest data\puzzles.jsonl
```

Batch-export fixed train/val/test splits:

```powershell
python -m solver.cli export-dataset --train-size 512 --val-size 128 --test-size 128 --blanks 40 --seed 7 --output-dir data\dataset --manifest-dir data\manifests
```

Solve a puzzle from an 81-character string:

```powershell
python -m solver.cli solve --puzzle "530070000600195000098000060800060003400803001700020006060000280000419005000080079" --check-unique
```

Solve a pasted 9-line board from standard input:

```powershell
Get-Content puzzle.txt | python -m solver.cli solve --stdin --check-unique
```

The parser accepts digits, `0`, `.`, whitespace, and visual separators like `|` and `-`, so pretty-printed boards work too.

Train the MLP baseline with on-the-fly generated data:

```powershell
python -m ai.train --epochs 5 --train-size 512 --val-size 128 --batch-size 32 --blanks 40
```

Train from one exported JSONL dataset with an internal split and save epoch metrics:

```powershell
python -m ai.train --dataset data\puzzles.jsonl --val-size 128 --epochs 20 --batch-size 32 --early-stopping-patience 5 --checkpoint ai\checkpoints\from_export.pt --best-checkpoint ai\checkpoints\from_export.best.pt --metrics-output ai\reports\from_export_metrics.json
```

Train from fixed train/val manifests:

```powershell
python -m ai.train --dataset data\manifests\train.jsonl --val-dataset data\manifests\val.jsonl --epochs 20 --batch-size 32 --early-stopping-patience 5 --checkpoint ai\checkpoints\from_split_export.pt --best-checkpoint ai\checkpoints\from_split_export.best.pt --metrics-output ai\reports\from_split_export_metrics.json
```

Train the Transformer baseline:

```powershell
python -m ai.train --dataset data\manifests\train.jsonl --val-dataset data\manifests\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 20 --batch-size 32 --early-stopping-patience 5 --checkpoint ai\checkpoints\transformer.pt --best-checkpoint ai\checkpoints\transformer.best.pt --metrics-output ai\reports\transformer_metrics.json
```

Train with an additional constraint penalty:

```powershell
python -m ai.train --dataset data\manifests_large\train.jsonl --val-dataset data\manifests_large\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 20 --batch-size 32 --early-stopping-patience 5 --constraint-loss-weight 0.05 --checkpoint ai\checkpoints\transformer_large_constraint.pt --best-checkpoint ai\checkpoints\transformer_large_constraint.best.pt --metrics-output ai\reports\transformer_large_constraint_metrics.json
```

Training behavior:
- `ai.train` now saves both the final checkpoint and the best-validation checkpoint.
- Early stopping is driven by validation `board_solved_rate` first, then `mean_total_conflicts`, then `blank_cell_accuracy`.
- The training config stored in checkpoints now records both requested sizes and resolved dataset sizes.
- `--constraint-loss-weight` adds a soft Sudoku consistency penalty on top of the blank-cell cross-entropy loss.
- Training reports now include `train_ce_loss` and `train_constraint_loss` separately so the two objectives can be traced independently.

Evaluate a checkpoint on the fixed test manifest with raw argmax decoding:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\from_split_export.pt --dataset data\manifests\test.jsonl --batch-size 32 --decode-mode argmax --report ai\reports\test_metrics.json
```

Evaluate the same checkpoint with iterative decoding:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large.pt --dataset data\manifests_large\test.jsonl --batch-size 32 --decode-mode iterative --report ai\reports\transformer_large_iterative_test_metrics.json
```

Evaluate with the product-grade pure-model preset:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_large\test.jsonl --batch-size 32 --decode-preset production_pure --report ai\reports\transformer_large_iterative_tuned_test_metrics.json
```

Evaluate the same checkpoint with solver-guided post-processing:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\transformer_large.pt --dataset data\manifests_large\test.jsonl --batch-size 32 --decode-mode solver_guided --report ai\reports\transformer_large_solver_guided_test_metrics.json
```

Run single-puzzle inference from a file with the default production-fast preset:

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --file data\dataset\train\puzzle_00001.txt
```

Run single-puzzle inference with the product-grade pure-model preset:

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --file data\dataset\train\puzzle_00001.txt --decode-preset production_pure
```

Run single-puzzle inference from an inline puzzle string:

```powershell
python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --puzzle "530070000600195000098000060800060003400803001700020006060000280000419005000080079" --decode-mode iterative
```

Run single-puzzle inference from standard input:

```powershell
Get-Content data\dataset\train\puzzle_00001.txt | python -m ai.infer --checkpoint ai\checkpoints\transformer_large_current.best.pt --stdin --decode-mode iterative
```

`ai.infer` prints the original puzzle, optional raw argmax prediction, decoded prediction, and whether the final board is structurally valid.

Decode-mode interpretation:
- `argmax`: raw model output with no repair.
- `iterative`: re-runs the model while filling confident cells round by round, without using the exact solver.
- `solver_guided`: uses the exact solver as a post-processor, so treat it as an upper-bound reference rather than raw model quality.
- `--decode-preset production_pure`: expands to `decode_mode=iterative`, `iterative_threshold=0.75`, and `iterative_max_fills_per_round=2`.
- `--iterative-threshold`: minimum softmax confidence required before iterative decoding fills a blank without falling back.
- `--iterative-max-fills-per-round`: optionally limits how many confident blanks iterative decoding may lock in per refinement round.

`ai.eval` now reports not only accuracy and valid-board rate, but also:
- `mean_mismatch_count`
- `mean_row_conflicts`
- `mean_col_conflicts`
- `mean_box_conflicts`
- `mean_total_conflicts`
- `mean_postprocess_change_count`
- `mean_decode_iteration_count`

Interpretation notes:
- `mean_postprocess_change_count` shows how many blank-cell raw argmax guesses were overridden by the selected decode path on average.
- `mean_decode_iteration_count` is mainly useful for `--decode-mode iterative`, where it shows how many refinement rounds were needed per board on average.

Render the saved reports to PNG images:

```powershell
python -m ai.plot_results --input ai\reports\from_split_export_metrics.json --output ai\reports\from_split_export_metrics.png
python -m ai.plot_results --input ai\reports\transformer_large_iterative_test_metrics.json --output ai\reports\transformer_large_iterative_test_metrics.png
python -m ai.plot_results --input ai\reports\transformer_large_solver_guided_test_metrics.json --output ai\reports\transformer_large_solver_guided_test_metrics.png
```

Evaluation PNGs now visualize both the rate-style metrics and the conflict-style metrics, including decode-iteration counts and post-processing changes when present.

Export generated data directly to JSONL:

```powershell
python -m ai.export_dataset --size 32 --blanks 40 --seed 7 --output data/sudoku_dataset.jsonl
```

Analyze model failures on generated samples:

```powershell
python -m ai.analyze_errors --checkpoint ai/checkpoints/smoke.pt --dataset-size 16 --limit 2
```

Analyze a fixed manifest with iterative decoding:

```powershell
python -m ai.analyze_errors --checkpoint ai/checkpoints/transformer_large.pt --dataset data/manifests_large/test.jsonl --decode-mode iterative --limit 2
```

Analyze a fixed manifest with solver-guided decoding:

```powershell
python -m ai.analyze_errors --checkpoint ai/checkpoints/transformer_large.pt --dataset data/manifests_large/test.jsonl --decode-mode solver_guided --limit 2
```

Benchmark latency across decode presets:

```powershell
python -m ai.benchmark --checkpoint ai\checkpoints\transformer_large_current.best.pt --dataset data\manifests_generalization\test.jsonl --batch-sizes 1 32 --decode-presets research_raw research_iterative production_pure production_fast --max-samples 128 --repeats 3 --report ai\reports\transformer_large_generalization_latency.json
```

`ai.benchmark` reports mean total time, mean per-board time, mean per-batch time, and throughput for each decode preset / batch-size pair.

## Learning path

A good order for this repo is:

1. Understand how `solver/` guarantees correct solutions.
2. Use the CLI to generate and solve puzzles directly.
3. Inspect exported `puzzle -> solution` records.
4. Train the MLP baseline and save per-epoch metrics.
5. Train the Transformer baseline on the same splits.
6. Run `ai.eval` on the fixed test split and inspect the conflict metrics.
7. Compare `argmax`, `iterative`, and `solver_guided` decode modes to separate raw prediction quality, non-solver refinement, and exact-solver repair.
8. Run `ai.infer` on a single puzzle file when you want an interactive checkpoint sanity check.
9. Try `--constraint-loss-weight` if you want to bias training toward lower structural conflict.
10. Render the training and evaluation reports with `ai.plot_results`.
11. Use `analyze_errors.py` to inspect failure modes.





