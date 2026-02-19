"""Unified plotting utilities for fit curves and gate values."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_fit_curve(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot prediction fit curve and save image.

    Parameters
    ----------
    y_true / y_pred:
        One-dimensional true and predicted value sequences, should have the same length.
    output_path:
        Target file path (supports no suffix, defaults to `.png`).
    title:
        Optional title string.
    """

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred have inconsistent dimensions, cannot plot.")

    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    indices = np.arange(y_true_arr.size)
    ax.plot(indices, y_true_arr, label="Ground Truth", linewidth=1.5, color="#d62728")
    ax.plot(indices, y_pred_arr, label="Prediction", linewidth=1.5, color="#1f77b4")

    path_signature = " ".join(part.lower() for part in output_path.parts)
    if "sru" in path_signature:
        y_max = 0.8
    else:
        y_max = 1.2

    ax.set_ylim(-0.05, y_max)
    ax.set_yticks(np.arange(0.0, y_max + 1e-6, 0.2))
    ax.set_xlim(0, y_true_arr.size - 1 if y_true_arr.size else 1)

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_gate_values(
    gate_values: np.ndarray,
    output_path: Path,
    *,
    y_true: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot gate value curve.

    Parameters
    ----------
    gate_values:
        Gate value array, shape (n_samples, hidden_size) or (n_samples,)
        If multi-dimensional, will compute mean or plot separately for each dimension
    output_path:
        Target file path (supports no suffix, defaults to `.png`).
    y_true:
        Deprecated, no longer used
    title:
        Deprecated, no longer used
    figsize:
        Figure size, default (10, 6)
    """
    gate_values = np.asarray(gate_values, dtype=np.float32)
    
    # If gate_values is multi-dimensional, compute mean
    if gate_values.ndim > 1:
        if gate_values.shape[1] > 1:
            # Multiple dimensions, compute mean
            gate_mean = gate_values.mean(axis=1)
        else:
            gate_mean = gate_values.squeeze()
    else:
        gate_mean = gate_values
    
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    n_samples = gate_mean.size
    indices = np.arange(n_samples)

    # Single Y-axis plot: gate values only (blue)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(indices, gate_mean, linestyle='-', 
            linewidth=2, color='#1f77b4', markersize=4)  # blue
    ax.set_xlabel('Sample', fontsize=14)
    ax.set_ylabel('Gate Value', fontsize=14)
    
    # Set Y-axis tick format to three decimal places
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_gate_distribution(
    gate_values: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot gate value distribution histogram.

    Parameters
    ----------
    gate_values:
        Gate value array, shape (n_samples, hidden_size) or (n_samples,)
    output_path:
        Target file path (supports no suffix, defaults to `.png`).
    title:
        Optional title string
    figsize:
        Figure size, default (8, 6)
    """
    gate_values = np.asarray(gate_values, dtype=np.float32)
    
    # Flatten multi-dimensional array
    if gate_values.ndim > 1:
        gate_flat = gate_values.flatten()
    else:
        gate_flat = gate_values
    
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(gate_flat, bins=50, alpha=0.7, color='tab:blue', edgecolor='black')
    ax.set_xlabel('Gate Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add statistics
    mean_val = gate_flat.mean()
    std_val = gate_flat.std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.3f}')
    ax.legend()
    
    if title:
        plt.title(title, fontsize=16)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

