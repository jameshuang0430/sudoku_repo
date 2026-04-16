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
  test_solver.py
ai/
  analyze_errors.py
  checkpoint.py
  dataset.py
  eval.py
  export_dataset.py
  model.py
  plot_results.py
  train.py
```

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
python -m ai.train --dataset data\puzzles.jsonl --val-size 128 --epochs 5 --batch-size 32 --checkpoint ai\checkpoints\from_export.pt --metrics-output ai\reports\from_export_metrics.json
```

Train from fixed train/val manifests:

```powershell
python -m ai.train --dataset data\manifests\train.jsonl --val-dataset data\manifests\val.jsonl --epochs 5 --batch-size 32 --checkpoint ai\checkpoints\from_split_export.pt --metrics-output ai\reports\from_split_export_metrics.json
```

Train the Transformer baseline:

```powershell
python -m ai.train --dataset data\manifests\train.jsonl --val-dataset data\manifests\val.jsonl --model transformer --transformer-embed-dim 128 --transformer-num-heads 8 --transformer-depth 4 --transformer-ff-dim 512 --epochs 5 --batch-size 32 --checkpoint ai\checkpoints\transformer.pt --metrics-output ai\reports\transformer_metrics.json
```

Evaluate a checkpoint on the fixed test manifest:

```powershell
python -m ai.eval --checkpoint ai\checkpoints\from_split_export.pt --dataset data\manifests\test.jsonl --batch-size 32 --report ai\reports\test_metrics.json
```

`ai.eval` now reports not only accuracy and valid-board rate, but also:
- `mean_mismatch_count`
- `mean_row_conflicts`
- `mean_col_conflicts`
- `mean_box_conflicts`
- `mean_total_conflicts`

Render the saved reports to PNG images:

```powershell
python -m ai.plot_results --input ai\reports\from_split_export_metrics.json --output ai\reports\from_split_export_metrics.png
python -m ai.plot_results --input ai\reports\test_metrics.json --output ai\reports\test_metrics.png
```

Evaluation PNGs now visualize both the rate-style metrics and the conflict-style metrics.

Export generated data directly to JSONL:

```powershell
python -m ai.export_dataset --size 32 --blanks 40 --seed 7 --output data/sudoku_dataset.jsonl
```

Analyze model failures:

```powershell
python -m ai.analyze_errors --checkpoint ai/checkpoints/smoke.pt --dataset-size 16 --limit 2
```

## Learning path

A good order for this repo is:

1. Understand how `solver/` guarantees correct solutions.
2. Use the CLI to generate and solve puzzles directly.
3. Inspect exported `puzzle -> solution` records.
4. Train the MLP baseline and save per-epoch metrics.
5. Train the Transformer baseline on the same splits.
6. Run `ai.eval` on the fixed test split and inspect the conflict metrics.
7. Render the training and evaluation reports with `ai.plot_results`.
8. Use `analyze_errors.py` to inspect failure modes.
