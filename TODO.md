# TODO

## P0: Clean Up Repo State

- [ ] Review the current untracked source/docs files and commit the ones that define the actual product state:
  - `solver/cli.py`
  - `solver/generator.py`
  - `solver/validator.py`
  - `ai/dataset.py`
  - `ai/export_dataset.py`
  - `ai/product.py`
  - `tests/test_cli.py`
  - `tests/test_solver.py`
  - `tests/test_infer.py`
  - `ai/README.md`
  - `HANDOFF.md`
  - `README.md`
  - `.gitignore`
- [ ] Keep runtime-only artifacts out of git:
  - `data/`
  - `ai/checkpoints/`
  - `__pycache__/`
- [ ] Review the remaining untracked reports and decide whether they are real experiment outputs worth committing or local scratch artifacts.
- [ ] Re-run the baseline verification commands after cleanup:
  - `python -m unittest discover -s tests -v`
  - `python -m solver.cli generate --blanks 10 --seed 7 --show-solution`
  - `python -m solver.cli solve --puzzle "530070000600195000098000060800060003400803001700020006060000280000419005000080079" --check-unique`
  - `python -m ai.product --file data\dataset\train\puzzle_00001.txt`

## P1: Clarify The Public Product Default

- [ ] Keep `python -m ai.product` as the primary documented single-puzzle entrypoint.
- [ ] Decide how explicitly the docs should recommend `fast` versus `pure`.
- [ ] Add one short comparison table or summary line to user-facing docs:
  - `fast`: exact repair and lower latency
  - `pure`: non-solver path and higher latency
- [ ] Consider whether the low-level `ai.infer` docs should be moved slightly lower in the README so the wrapper stays visually primary.

Acceptance criteria:
- A user can tell which command to run first without reading the research workflow.

## P2: Add A Release Check Workflow

- [x] Add one repeatable evaluation command or script that compares `production_fast` and `production_pure` on the same manifest.
- [x] Store the resulting metrics in a predictable report location.
- [x] Document how to use that comparison before changing decode defaults.
- [x] Decide which saved release-check report becomes the official baseline and what regression thresholds should fail CI or release review.
- [ ] Decide whether the current 128-sample smoke gate should remain the only release gate or be paired with a slower full-manifest comparison.

Acceptance criteria:
- There is one standard way to answer "did the recommended preset regress?"

## P3: Improve Raw Model Quality

- [ ] Treat decode-quality and raw-model-quality as separate tracks.
- [ ] If raw argmax quality matters, run new experiments aimed at reducing the current gap between raw decoding and product decoding.
- [ ] Prefer experiments that improve global consistency, not just blank-cell accuracy.
- [ ] Keep fresh-split evaluation in the loop so model changes are not judged on one fixed manifest only.

Acceptance criteria:
- New model work is judged by board validity and solved-board rate, not only cell accuracy.

## P4: Output And UX Polish

- [ ] Add a one-line recommendation when a user selects a research preset in an inference-oriented command.
- [ ] Consider a simplified end-user command for file-based solving in addition to puzzle inference.
- [ ] Keep benchmark and evaluation output easy to compare across presets.

## Recommended Order

1. Finish repo cleanup and commit the current product-facing state.
2. Keep `ai.product` as the top documented entrypoint and clarify `fast` versus `pure`.
3. Decide whether to keep the current smoke-gate baseline alone or add a second full-manifest release gate.
4. Only then return to deeper raw-model-quality work.
