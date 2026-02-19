"""Model definitions for ESF."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (batch-first).

    Notes
    - ESF uses `nn.MultiheadAttention(..., batch_first=True)`, so tensors are [B, T, D].
    - We keep dropout optional; to minimize perturbation to existing experiments,
      ESF wires this with dropout=0.0 by default.
    """

    def __init__(self, d_model: int, dropout: float = 0.01, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: [1, max_len, d_model] for batch-first inputs [B, T, D]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if x.ndim != 3:
            raise ValueError(f"PositionalEncoding expects a 3D tensor [B, T, D], got shape={tuple(x.shape)}")
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        query: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            query, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(feedforward)
        return self.norm2(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, query, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)


class ESF(nn.Module):
    """ESF model: Adaptive Spatiotemporal Feature Fusion for Soft Sensor Modeling."""

    def __init__(
        self,
        input_size: int,
        timesteps: int,
        gam_feature_dim: int,
        hidden_size: int = 10,
        dim_feedforward: int = 2000,
        nhead: int = 1,
        num_layers: int = 2,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        model_dim = hidden_size
        self.seq_proj = nn.Linear(input_size, model_dim)
        # Positional encoding is applied to the projected sequence before attention.
        # Keep PE dropout at 0.0 to minimize perturbation; other dropouts remain as configured.
        self.positional_encoding = PositionalEncoding(d_model=model_dim, dropout=0.0, max_len=timesteps + 5)
        self.transformer = TransformerEncoder(
            d_model=model_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_gru = nn.Linear(model_dim * timesteps, hidden_size)
        self.fc_gam_q = nn.Linear(gam_feature_dim, model_dim)
        self.fc_gam = nn.Linear(gam_feature_dim, hidden_size)
        self.gate_weight_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.final_fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(hidden_size)

    def forward(self, seq_inputs: torch.Tensor, gam_inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_encoded = self.seq_proj(seq_inputs)
        seq_encoded = self.positional_encoding(seq_encoded)
        query = self.fc_gam_q(gam_inputs)
        query = torch.stack([query] * self.timesteps, dim=1)
        transformer_output = self.transformer(seq_encoded, query)
        gru_like = self.fc_gru(transformer_output.reshape(seq_inputs.shape[0], -1))
        tanh_output = torch.tanh(gru_like)

        gam_output = self.fc_gam(gam_inputs)
        combined = torch.cat([gru_like, gam_output], dim=-1)
        gate = torch.sigmoid(self.gate_weight_fc(combined))

        # For interpretability:
        # - tanh_output: temporal/sequence branch contribution proxy
        # - gate * gam_output: spatial/GAM branch contribution proxy (scaled by gate)
        seq_contrib = tanh_output
        gam_contrib = gate * gam_output
        seq_norm = torch.norm(seq_contrib, p=2, dim=1)
        gam_norm = torch.norm(gam_contrib, p=2, dim=1)
        mix_ratio = gam_norm / (gam_norm + seq_norm + 1e-8)  # higher => more spatial/GAM

        fused = seq_contrib + gam_contrib
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        final = self.final_fc(fused)
        return final.view(-1), {"gate": gate, "seq_norm": seq_norm, "gam_norm": gam_norm, "mix_ratio": mix_ratio}


def build_model(
    architecture: str,
    *,
    input_size: int,
    timesteps: int,
    gam_feature_dim: int,
    hidden_size: int,
    dim_feedforward: int,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "esf":
        return ESF(
            input_size=input_size,
            timesteps=timesteps,
            gam_feature_dim=gam_feature_dim,
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown architecture: {architecture}. Only 'esf' is supported.")


