# ESF: Adaptive Spatiotemporal Feature Fusion for Soft Sensor Modeling

This repository provides the implementation of the **ESF (Enhanced Sensor Fusion)** model for soft sensor modeling in industrial processes. ESF adaptively fuses spatial (GAM-based) and temporal (Transformer-based) features through a learnable gate mechanism.

## Features

- **Adaptive Feature Fusion**: A gate mechanism that dynamically balances spatial and temporal feature contributions
- **Interpretability**: GAM-based response curves and deletion game analysis for feature importance
- **Positional Encoding**: Transformer encoder with positional encoding for temporal modeling
- **Comprehensive Evaluation**: Support for deletion game and GAM response curve visualization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training ESF Model

```bash
python -m esf_project.cli \
  --data-csv data/SRU_data.csv \
  --target-column y1 \
  --no-drop-index-column \
  --timesteps 14 \
  --n-splines 5 \
  --lam 0.25 \
  --hidden-size 192 \
  --dim-feedforward 1200 \
  --dropout 0.04 \
  --learning-rate 0.0003 \
  --weight-decay 8e-05 \
  --batch-size 20 \
  --max-epochs 200 \
  --patience 100 \
  --results-dir results
```

### Running Post-hoc Analysis (GAM Response Curves)

```bash
python run_posthoc.py \
  --data-csv data/SRU_data.csv \
  --checkpoint results/checkpoints/esf_seed323.pt \
  --target-column y1 \
  --no-drop-index-column \
  --input-size 5 \
  --timesteps 14 \
  --n-splines 5 \
  --lam 0.25 \
  --hidden-size 192 \
  --output-dir results/posthoc
```

### Running Deletion Game

```bash
python deletion_game.py \
  --data-csv data/SRU_data.csv \
  --checkpoint results/checkpoints/esf_seed323.pt \
  --target-column y1 \
  --no-drop-index-column \
  --timesteps 14 \
  --n-splines 5 \
  --lam 0.25 \
  --hidden-size 192 \
  --output-dir results/deletion_game
```

## Project Structure

```
esf_github/
├── esf_project/          # Core ESF implementation
│   ├── models.py         # ESF model architecture
│   ├── data.py           # Data loading and GAM feature construction
│   ├── trainer.py        # Training utilities
│   ├── config.py         # Configuration classes
│   ├── metrics.py        # Evaluation metrics
│   ├── utils.py          # Utility functions
│   └── cli.py            # Command-line interface
├── utils/                # Utility modules
│   ├── pytorchtools.py   # Early stopping
│   └── plotting.py       # Plotting utilities
├── data/                 # Dataset directory
│   └── SRU_data.csv      # SRU dataset
├── run_posthoc.py        # Post-hoc analysis script
├── deletion_game.py      # Deletion game script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Model Architecture

ESF consists of two main branches:

1. **Temporal Branch**: Transformer encoder with positional encoding processes sequential input
2. **Spatial Branch**: GAM (Generalized Additive Model) captures non-linear feature interactions

A learnable gate mechanism adaptively fuses the outputs from both branches based on input characteristics.

## Key Hyperparameters

- `timesteps`: Sequence length (default: 14)
- `n_splines`: Number of GAM splines per feature (default: 5)
- `lam`: GAM regularization parameter (default: 0.25)
- `hidden_size`: Hidden dimension (default: 192)
- `dim_feedforward`: Transformer feedforward dimension (default: 1200)
- `dropout`: Dropout rate (default: 0.04)

## Outputs

After training, the following outputs are generated:

- **Model checkpoints**: Saved in `results/checkpoints/`
- **Predictions**: CSV files with true and predicted values
- **Gate values**: CSV files with gate scores and branch contributions
- **Figures**: Fit curves, gate value curves, and distributions
- **Metrics**: JSON files with R², RMSE, MAE, etc.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{esf2024,
  title={ESF: Adaptive Spatiotemporal Feature Fusion for Soft Sensor Modeling},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

