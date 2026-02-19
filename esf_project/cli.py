"""Command-line interface for running ESF experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .trainer import run_experiments, train_single_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ESF experiments")
    parser.add_argument("--data-csv", type=Path, required=True, help="Path to the raw CSV file.")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target column name or index (default: last column).",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Feature column names (default: all columns except target).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="Custom delimiter passed to pandas.read_csv (default: auto).",
    )
    parser.add_argument(
        "--delim-whitespace",
        action="store_true",
        help="Treat consecutive whitespace as delimiter when loading the dataset.",
    )
    parser.add_argument(
        "--no-drop-index-column",
        dest="drop_index_column",
        action="store_false",
        help="Disable automatic removal of the first column (useful when CSV has no index column).",
    )
    parser.set_defaults(drop_index_column=True)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples when performing train/validation split (default: sequential split).",
    )
    parser.add_argument(
        "--split-random-state",
        type=int,
        default=None,
        help="Random seed used for shuffled train/val split (effective only when --shuffle).",
    )
    parser.add_argument(
        "--temporal-aggregations",
        nargs="*",
        default=None,
        help=(
            "Temporal summary statistics applied on each sliding window to reduce sequence length. "
            "Supported values include mean, max, min, std, median, slope, first, last, range."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=14, help="Sequence length for ESF.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--n-splines", type=int, default=5, help="Number of GAM splines per feature.")
    parser.add_argument("--lam", type=float, default=0.25, help="Regularization parameter for GAM smoothing.")
    parser.add_argument("--input-size", type=int, default=None, help="Number of variables per timestep (default: infer from data).")
    parser.add_argument("--hidden-size", type=int, default=192, help="Hidden size of ESF heads.")
    parser.add_argument("--dim-feedforward", type=int, default=1200, help="Transformer FF dimension.")
    parser.add_argument("--nhead", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.04, help="Dropout probability.")
    parser.add_argument("--batch-size", type=int, default=20, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=8e-05, help="Optimizer weight decay.")
    parser.add_argument("--max-epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--delta", type=float, default=1e-16, help="Early stopping delta.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(range(323, 333)),
        help="Random seeds for repeated runs.",
    )
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=["esf"],
        help="Model architectures to evaluate (only 'esf' is supported).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only the first architecture and first seed for a quick sanity check.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    def _parse_target(value: str | None) -> int | str:
        if value is None:
            return -1
        try:
            return int(value)
        except ValueError:
            return value

    data_cfg = DataConfig(
        csv_path=args.data_csv,
        timesteps=args.timesteps,
        train_ratio=args.train_ratio,
        n_splines=args.n_splines,
        lam=args.lam,
        target_column=_parse_target(args.target_column),
        feature_columns=args.feature_columns,
        drop_index_column=args.drop_index_column,
        delimiter=args.delimiter,
        delim_whitespace=args.delim_whitespace,
        shuffle=args.shuffle,
        split_random_state=args.split_random_state,
        temporal_aggregations=args.temporal_aggregations,
    )

    model_cfg = ModelConfig(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        delta=args.delta,
        seeds=args.seeds,
    )

    exp_cfg = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        results_dir=args.results_dir,
    )

    if args.single_run:
        metrics = train_single_run(exp_cfg, architecture=args.architectures[0], seed=args.seeds[0])
        print(metrics)
    else:
        summaries = run_experiments(exp_cfg, architectures=args.architectures)
        for summary in summaries:
            print(summary)


if __name__ == "__main__":  # pragma: no cover
    main()


