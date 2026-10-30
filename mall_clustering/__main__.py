"""Fit customer groups or assign a fictional point to the saved clusters."""

import argparse
import json
from pathlib import Path

from mall_clustering.model import DEFAULT_DATA, DEFAULT_OUTPUT, assign, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mall Customer Clustering with K-means")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--data", type=Path, default=DEFAULT_DATA)
    fitting.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    fitting.add_argument("--k", choices=["auto", *map(str, range(2, 9))], default="auto")
    prediction = commands.add_parser("assign")
    prediction.add_argument("--income", type=float, required=True, help="Annual income in thousands of synthetic USD")
    prediction.add_argument("--score", type=int, required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    args = parser.parse_args(argv)
    try:
        result = train(args.data, args.output_dir, args.k) if args.command == "train" else assign(args.income, args.score, args.model)
    except FileNotFoundError as error:
        parser.error(f"File not found: {error.filename}. Run training first if the model is missing.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
