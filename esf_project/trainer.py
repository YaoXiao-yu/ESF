"""Training utilities for ESF experiments."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from utils.pytorchtools import EarlyStopping

from .config import ExperimentConfig
from .data import DatasetBundle, prepare_datasets
from .metrics import regression_report
from .models import build_model
from .utils import ensure_dirs, resolve_device, set_seed
from utils.plotting import plot_fit_curve, plot_gate_values, plot_gate_distribution


def _epoch_step(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    for seq_x, gam_x, labels in loader:
        seq_x = seq_x.to(device)
        gam_x = gam_x.to(device)
        labels = labels.to(device)

        if train:
            optimizer = _epoch_step.optimizer
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds, _ = model(seq_x, gam_x)
            loss = criterion(preds, labels)
            if train:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item() * labels.size(0)

    return epoch_loss / len(loader.dataset)


def train_single_run(
    config: ExperimentConfig,
    architecture: str = "esf",
    seed: int = 323,
) -> Dict[str, float]:
    set_seed(seed)
    device = resolve_device()
    config.ensure_dirs()

    dataset = prepare_datasets(
        csv_path=config.data.csv_path,
        timesteps=config.data.timesteps,
        train_ratio=config.data.train_ratio,
        n_splines=config.data.n_splines,
        lam=config.data.lam,
        batch_size=config.train.batch_size,
        device=device,
        target_column=config.data.target_column,
        feature_columns=config.data.feature_columns,
        drop_index_column=config.data.drop_index_column,
        delimiter=config.data.delimiter,
        delim_whitespace=config.data.delim_whitespace,
        shuffle=config.data.shuffle,
        split_random_state=config.data.split_random_state,
        temporal_aggregations=config.data.temporal_aggregations,
    )

    effective_timesteps = dataset.X_train.shape[1]
    if effective_timesteps != config.data.timesteps:
        print(
            f"[info] Original time window timesteps={config.data.timesteps}, "
            f"actual input length after aggregation={effective_timesteps}."
        )

    inferred_input_size = dataset.X_train.shape[-1]
    input_size = (
        config.model.input_size if config.model.input_size is not None else inferred_input_size
    )
    if input_size != inferred_input_size:
        print(
            f"[warning] Config input_size={config.model.input_size} does not match dataset feature count "
            f"{inferred_input_size}, automatically using dataset dimension."
        )
        input_size = inferred_input_size

    model = build_model(
        architecture,
        input_size=input_size,
        timesteps=effective_timesteps,
        gam_feature_dim=dataset.X_gam_train.shape[1],
        hidden_size=config.model.hidden_size,
        dim_feedforward=config.model.dim_feedforward,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    checkpoint_path = config.results_dir / "checkpoints" / f"{architecture}_seed{seed}.pt"
    early_stopping = EarlyStopping(
        patience=config.train.patience,
        delta=config.train.delta,
        path=str(checkpoint_path),
        verbose=False,
    )

    _epoch_step.optimizer = optimizer  # type: ignore[attr-defined]

    history: List[Dict[str, float]] = []

    for epoch in range(1, config.train.max_epochs + 1):
        train_loss = _epoch_step(model, dataset.train_loader, criterion, device, train=True)
        val_loss = _epoch_step(model, dataset.val_loader, criterion, device, train=False)
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[Seed {seed}] Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch} for seed {seed}.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds, outputs = model(dataset.X_val, dataset.X_gam_val)
        # Extract gate values
        gate_values = outputs.get("gate", None)
        seq_norm = outputs.get("seq_norm", None)
        gam_norm = outputs.get("gam_norm", None)
        mix_ratio = outputs.get("mix_ratio", None)
        if gate_values is not None:
            gate_values_np = gate_values.detach().cpu().numpy()
        else:
            gate_values_np = None
        seq_norm_np = seq_norm.detach().cpu().numpy() if seq_norm is not None else None
        gam_norm_np = gam_norm.detach().cpu().numpy() if gam_norm is not None else None
        mix_ratio_np = mix_ratio.detach().cpu().numpy() if mix_ratio is not None else None

    preds_np = preds.detach().cpu().numpy()
    y_val_np = dataset.y_val.detach().cpu().numpy()

    metrics = regression_report(y_val_np, preds_np)
    metrics["seed"] = seed
    metrics["architecture"] = architecture

    dataset_label = config.data.csv_path.stem if config.data.csv_path else "dataset"
    safe_target = re.sub(r"[^\w\.-]+", "_", dataset.target_name)
    shared_dir = Path("results/predictions") / f"{dataset_label}_{safe_target}"
    shared_dir.mkdir(parents=True, exist_ok=True)

    y_true_flat = y_val_np.reshape(-1)
    y_pred_flat = preds_np.reshape(-1)

    shared_pred_path = shared_dir / f"{architecture}_predictions_r2_{metrics['r2']:.4f}.csv"
    pd.DataFrame(
        {
            "index": np.arange(y_true_flat.size),
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
        }
    ).to_csv(shared_pred_path, index=False)

    shared_fig_path = shared_dir / f"{architecture}_fit_curve_r2_{metrics['r2']:.4f}.png"
    plot_fit_curve(
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        output_path=shared_fig_path,
        title=f"{architecture.upper()} Validation Fit (R$^2$ = {metrics['r2']:.3f})",
    )

    predictions_dir = config.results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / f"{architecture}_seed{seed}.npy"
    save_dict = {"pred": preds_np, "true": y_val_np}
    if gate_values_np is not None:
        save_dict["gate"] = gate_values_np
    if seq_norm_np is not None:
        save_dict["seq_norm"] = seq_norm_np
    if gam_norm_np is not None:
        save_dict["gam_norm"] = gam_norm_np
    if mix_ratio_np is not None:
        save_dict["mix_ratio"] = mix_ratio_np
    np.save(predictions_file, save_dict)
    
    # Save and plot gate values (if present)
    if gate_values_np is not None:
        # Save gate values to CSV
        gate_csv_path = shared_dir / f"{architecture}_gate_values_r2_{metrics['r2']:.4f}.csv"
        gate_df = pd.DataFrame({
            "index": np.arange(gate_values_np.shape[0]),
            "gate_mean": gate_values_np.mean(axis=1) if gate_values_np.ndim > 1 else gate_values_np,
        })
        # Save branch contribution diagnostics (for "temporal vs spatial" interpretation)
        if mix_ratio_np is not None:
            gate_df["mix_ratio"] = mix_ratio_np.reshape(-1)
        if seq_norm_np is not None:
            gate_df["seq_norm"] = seq_norm_np.reshape(-1)
        if gam_norm_np is not None:
            gate_df["gam_norm"] = gam_norm_np.reshape(-1)
        # If multi-dimensional gate values, save each dimension
        if gate_values_np.ndim > 1 and gate_values_np.shape[1] > 1:
            for i in range(gate_values_np.shape[1]):
                gate_df[f"gate_dim_{i}"] = gate_values_np[:, i]
        gate_df.to_csv(gate_csv_path, index=False)
        print(f"[info] Gate values saved: {gate_csv_path}")
        
        # Plot gate value curve (gate values only)
        gate_fig_path = shared_dir / f"{architecture}_gate_curve_r2_{metrics['r2']:.4f}.png"
        plot_gate_values(
            gate_values=gate_values_np,
            output_path=gate_fig_path,
        )
        print(f"[info] Gate value curve saved: {gate_fig_path}")
        
        # Plot gate value distribution
        gate_dist_path = shared_dir / f"{architecture}_gate_distribution_r2_{metrics['r2']:.4f}.png"
        plot_gate_distribution(
            gate_values=gate_values_np,
            output_path=gate_dist_path,
            title=f"{architecture.upper()} Gate Value Distribution",
        )
        print(f"[info] Gate value distribution saved: {gate_dist_path}")
        
        # Copy to figures directory
        figures_dir = config.results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gate_fig_path, figures_dir / gate_fig_path.name)
        shutil.copy2(gate_dist_path, figures_dir / gate_dist_path.name)

    figures_dir = config.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_fig_path, figures_dir / shared_fig_path.name)

    log_path = shared_dir / f"{architecture}_log.json"
    log_payload = {
        "dataset": str(config.data.csv_path),
        "target": dataset.target_name,
        "architecture": architecture,
        "seed": seed,
        "metrics": metrics,
        "timesteps": config.data.timesteps,
        "train_ratio": config.data.train_ratio,
        "batch_size": config.train.batch_size,
        "max_epochs": config.train.max_epochs,
        "patience": config.train.patience,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "hidden_size": config.model.hidden_size,
        "dim_feedforward": config.model.dim_feedforward,
        "dropout": config.model.dropout,
        "n_splines": config.data.n_splines,
        "lam": config.data.lam,
    }
    log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    history_file = config.results_dir / f"{architecture}_seed{seed}_history.json"
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics_file = config.results_dir / f"{architecture}_seed{seed}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def run_experiments(
    config: ExperimentConfig,
    architectures: List[str],
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for architecture in architectures:
        arch_results: List[Dict[str, float]] = []
        for idx, seed in enumerate(config.train.seeds):
            metrics = train_single_run(config, architecture=architecture, seed=seed)
            arch_results.append(metrics)
        aggregated = _aggregate_metrics(arch_results)
        aggregated["architecture"] = architecture
        results.append(aggregated)
        summary_file = config.results_dir / f"summary_{architecture}.json"
        summary_file.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    return results


def _aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = [k for k in records[0].keys() if k not in {"seed", "architecture"}]
    aggregated = {}
    for key in keys:
        values = [record[key] for record in records]
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


