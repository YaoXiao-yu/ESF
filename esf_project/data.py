"""Data utilities for the ESF project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pygam import LinearGAM, s


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_gam_train: torch.Tensor
    X_gam_val: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    gam: LinearGAM
    raw_sequences: np.ndarray
    feature_names: list[str]
    target_name: str
    sequence_labels: list[str]


def _slide_windows(array: np.ndarray, timesteps: int) -> np.ndarray:
    windows = []
    for idx in range(len(array) - timesteps + 1):
        windows.append(array[idx : idx + timesteps])
    return np.asarray(windows)


def _apply_temporal_aggregations(
    sequences: np.ndarray, aggregations: list[str]
) -> tuple[np.ndarray, list[str]]:
    if sequences.size == 0:
        return sequences, []

    supported = {
        "mean": lambda x: np.mean(x, axis=1),
        "max": lambda x: np.max(x, axis=1),
        "min": lambda x: np.min(x, axis=1),
        "std": lambda x: np.std(x, axis=1, ddof=0),
        "median": lambda x: np.median(x, axis=1),
        "last": lambda x: x[:, -1, :],
        "first": lambda x: x[:, 0, :],
        "range": lambda x: np.max(x, axis=1) - np.min(x, axis=1),
        "amplitude": lambda x: np.max(x, axis=1) - np.min(x, axis=1),
    }

    def _slope(x: np.ndarray) -> np.ndarray:
        if x.shape[1] < 2:
            return np.zeros((x.shape[0], x.shape[2]), dtype=x.dtype)
        indices = np.arange(x.shape[1], dtype=x.dtype)
        indices = indices.reshape(1, -1, 1)
        mean_t = np.mean(indices, axis=1, keepdims=True)
        mean_x = np.mean(x, axis=1, keepdims=True)
        numerator = np.sum((indices - mean_t) * (x - mean_x), axis=1)
        denominator = np.sum((indices - mean_t) ** 2, axis=1)
        denominator[denominator == 0] = 1.0
        return numerator / denominator

    supported["slope"] = _slope

    aggregated_blocks: list[np.ndarray] = []
    labels: list[str] = []

    for agg_name in aggregations:
        key = agg_name.lower()
        if key not in supported:
            raise ValueError(
                f"Unsupported aggregation '{agg_name}'. "
                f"Available options: {sorted(supported.keys())}."
            )
        agg_func = supported[key]
        block = agg_func(sequences)
        aggregated_blocks.append(block[:, np.newaxis, :])
        labels.append(agg_name)

    aggregated = np.concatenate(aggregated_blocks, axis=1)
    return aggregated, labels


def _build_gam_terms(n_features: int, n_splines: int, lam: float) -> LinearGAM:
    # Dynamically adjust spline order: if n_splines=3, order must be 2 (quadratic spline)
    spline_order = 3 if n_splines > 3 else 2
    terms = [s(i, n_splines=n_splines, lam=lam, spline_order=spline_order) for i in range(n_features)]
    if not terms:
        raise ValueError("GAM requires at least one feature term")
    term_sum = terms[0]
    for term in terms[1:]:
        term_sum = term_sum + term
    return LinearGAM(term_sum)


def prepare_datasets(
    csv_path: Path,
    timesteps: int,
    train_ratio: float,
    n_splines: int,
    lam: float,
    batch_size: int,
    device: torch.device,
    *,
    target_column: int | str = -1,
    feature_columns: list[str] | None = None,
    drop_index_column: bool = True,
    delimiter: str | None = None,
    delim_whitespace: bool = False,
    shuffle: bool = False,
    split_random_state: int | None = None,
    temporal_aggregations: list[str] | None = None,
) -> DatasetBundle:
    read_kwargs: dict[str, object] = {}
    if delim_whitespace:
        read_kwargs["delim_whitespace"] = True
    if delimiter is not None:
        read_kwargs["sep"] = delimiter

    data = pd.read_csv(csv_path, **read_kwargs)

    if drop_index_column and data.shape[1] > 0:
        data = data.iloc[:, 1:]

    if isinstance(target_column, str):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        target_series = data[target_column]
        feature_df = data.drop(columns=[target_column])
        target_name = target_column
    else:
        idx = target_column
        if idx < 0:
            idx = data.shape[1] + idx
        if idx < 0 or idx >= data.shape[1]:
            raise ValueError(f"Target column index {target_column} is out of bounds.")
        target_series = data.iloc[:, idx]
        target_name = target_series.name if target_series.name is not None else f"target_{idx}"
        feature_df = data.drop(columns=data.columns[idx])

    if feature_columns:
        missing = [col for col in feature_columns if col not in feature_df.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found in dataset.")
        feature_df = feature_df[feature_columns]

    feature_names = list(feature_df.columns)
    features = feature_df.values
    targets = target_series.values

    gam_features = features[timesteps - 1 :]
    gam_targets = targets[timesteps - 1 :]

    gam_model = _build_gam_terms(gam_features.shape[1], n_splines, lam)
    gam_model.fit(gam_features, gam_targets)

    gam_train_matrix = gam_model._modelmat(gam_features).toarray()

    raw_sequence_windows = _slide_windows(features, timesteps)
    sequence_labels = [f"t{idx + 1}" for idx in range(timesteps)]

    if temporal_aggregations:
        sequence_data, sequence_labels = _apply_temporal_aggregations(
            raw_sequence_windows, temporal_aggregations
        )
    else:
        sequence_data = raw_sequence_windows
    target_aligned = targets[timesteps - 1 :]

    X_train_seq, X_val_seq, y_train, y_val, X_train_gam, X_val_gam = train_test_split(
        sequence_data,
        target_aligned,
        gam_train_matrix,
        train_size=train_ratio,
        shuffle=shuffle,
        random_state=split_random_state,
    )

    X_train_seq_t = torch.from_numpy(X_train_seq).float().to(device)
    X_val_seq_t = torch.from_numpy(X_val_seq).float().to(device)
    X_train_gam_t = torch.from_numpy(X_train_gam).float().to(device)
    X_val_gam_t = torch.from_numpy(X_val_gam).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    train_dataset = TensorDataset(X_train_seq_t, X_train_gam_t, y_train_t)
    val_dataset = TensorDataset(X_val_seq_t, X_val_gam_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        X_train=X_train_seq_t,
        X_val=X_val_seq_t,
        X_gam_train=X_train_gam_t,
        X_gam_val=X_val_gam_t,
        y_train=y_train_t,
        y_val=y_val_t,
        gam=gam_model,
        raw_sequences=raw_sequence_windows,
        feature_names=feature_names,
        target_name=target_name,
        sequence_labels=sequence_labels,
    )


