"""Metric utilities for the ESF project."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def direction_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    agreement = (
        np.sign(np.diff(actual)) * np.sign(np.diff(predicted))
    ) >= 0
    return float(agreement.sum() / agreement.shape[0])


def nmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mse = mean_squared_error(actual, predicted)
    variance = np.var(actual)
    return float(mse / variance) if variance > 0 else float("nan")


def regression_report(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    return {
        "r2": r2_score(actual, predicted),
        "rmse": rmse,
    }


def torch_nmse(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    mse = torch.mean((predicted - actual) ** 2).item()
    variance = torch.var(actual).item()
    return float(mse / variance) if variance > 0 else float("nan")


