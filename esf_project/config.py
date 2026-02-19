"""Configuration objects for ESF experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    csv_path: Path
    timesteps: int = 14
    train_ratio: float = 0.8
    n_splines: int = 5
    lam: float = 0.25
    target_column: int | str = -1
    feature_columns: List[str] | None = None
    drop_index_column: bool = True
    delimiter: str | None = None
    delim_whitespace: bool = False
    shuffle: bool = False
    split_random_state: int | None = None
    temporal_aggregations: List[str] | None = None


@dataclass
class ModelConfig:
    input_size: int | None = None
    hidden_size: int = 192
    dim_feedforward: int = 1200
    nhead: int = 1
    num_layers: int = 2
    dropout: float = 0.04


@dataclass
class TrainConfig:
    batch_size: int = 20
    max_epochs: int = 200
    learning_rate: float = 0.0003
    weight_decay: float = 8e-05
    patience: int = 100
    delta: float = 1e-16
    seeds: List[int] = field(default_factory=lambda: list(range(323, 333)))


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    results_dir: Path = Path("results")

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)


