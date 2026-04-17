from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import SudokuDataset, sample_to_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated Sudoku training data to JSONL.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--blanks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sudoku_dataset.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SudokuDataset(size=args.size, blanks=args.blanks, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            record = sample_to_record(dataset[index])
            record["index"] = index
            record["seed"] = args.seed + index
            handle.write(json.dumps(record))
            handle.write("\n")

    print(f"exported_records={len(dataset)} output={args.output}")


if __name__ == "__main__":
    main()
