"""
Phase 4A Transformer Baseline Sequence-to-Sequence Architecture.

Refactored from the verified 19.7M-parameter Transformer Encoder baseline
executed by Achouri Youcef Islem during the May 2026 research study.

Architecture:
- Input Projection: Linear(768 -> 512)
- Sinusoidal Positional Encoding
- 6-layer Transformer Encoder (d_model=512, nheads=8, dim_feedforward=2048, dropout=0.1)
- Output Projection: Linear(512 -> 768)
- Exact parameter count: 19,702,016 parameters

Empirical Historical Performance:
- Trained for 45 epochs on 11,323 matched NAM/speech pairs
- EarlyStopping halted at Epoch 45, reloaded optimal Epoch 35 weights
- Evaluated Test Loss: 0.23171 MAE on 1,133 held-out utterances (speaker-dependent split)
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as defined in Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input representations.

        Args:
            x: Input tensor of shape (Batch, Seq_len, d_model).

        Returns:
            Position-encoded tensor of shape (Batch, Seq_len, d_model).
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerSeq2Seq(nn.Module):
    """
    6-layer Transformer Encoder mapping HuBERT representations from NAM to Speech.

    Both source (NAM) and target (Speech) reside in the 768-dimensional latent
    feature space extracted by self-supervised speech models (e.g. HuBERT).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for NAM-to-Speech translation.

        Args:
            x: Input NAM features of shape (Batch, Seq_len, 768).
            src_key_padding_mask: Optional boolean tensor of shape (Batch, Seq_len)
                where True indicates padded time frames to ignore in self-attention.
            mask: Optional alias for src_key_padding_mask.

        Returns:
            Predicted speech features of shape (Batch, Seq_len, 768).
        """
        if src_key_padding_mask is None and mask is not None:
            src_key_padding_mask = mask

        out = self.proj_in(x)
        out = self.pe(out)
        out = self.encoder(out, src_key_padding_mask=src_key_padding_mask)
        out = self.proj_out(out)
        return out

    def get_parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
