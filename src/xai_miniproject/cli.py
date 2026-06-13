from __future__ import annotations

import argparse

from xai_miniproject.analyze import run_analysis
from xai_miniproject.config import load_config
from xai_miniproject.explain import run_explanations
from xai_miniproject.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xai-mini")
    parser.add_argument(
        "--config",
        default="configs/aifb.yaml",
        help="Path to a YAML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze", help="Compute RDF and task statistics.")
    subparsers.add_parser("train", help="Train and evaluate the R-GCN classifier.")
    subparsers.add_parser("explain", help="Run CELOE/EvoLearner explanations from GNN predictions.")
    subparsers.add_parser("run-all", help="Run analyze, train, and explain in sequence.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "analyze":
        run_analysis(config)
    elif args.command == "train":
        run_training(config)
    elif args.command == "explain":
        run_explanations(config)
    elif args.command == "run-all":
        run_analysis(config)
        run_training(config)
        run_explanations(config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
