"""
Phase 4B StethoSpeech Feed-Forward Transformer (FFT) Architecture.

Adapted from the official StethoSpeech specification (Shah et al., ACM IMWUT 2024)
and FastSpeech 2 (Ren et al., 2021).

Architecture:
- Input Projection: Linear(768 -> 256) + ReLU
- 6 FFT Encoder layers (n_head=2, d_inner=1024, kernels=[9, 1])
- 4 FFT Decoder layers (n_head=2, d_inner=1024, kernels=[9, 1])
- Output Projection: Linear(256 -> 768)
- Exact parameter count: 29,263,360 parameters (executed base model)
- With CTC projection head (32 vocab): 29,485,344 parameters (+221,984 weights)

Scientific and Historical Note:
- In the historical May 2026 exploration, training was run for approximately two
  preliminary epochs before encountering cloud session runtime constraints.
- While the auxiliary CTC projection head was defined to match the paper's
  theoretical architecture, auxiliary CTC loss was NOT calculated in the
  executed training loop (which evaluated pure masked L1 loss).
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_table(n_pos: int, d: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding table."""
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer (FFT) block.

    Consists of Multi-Head Self-Attention followed by 1D Convolutions
    (kernel 9 for temporal receptive field expansion, followed by kernel 1 projection).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 2,
        d_inner: int = 1024,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 1]

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.conv1 = nn.Conv1d(
            d_model, d_inner, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            d_inner, d_model, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through FFT block.

        Args:
            x: Input tensor of shape (Batch, Seq_len, d_model).
            mask: Optional key padding mask (Batch, Seq_len).

        Returns:
            Output tensor of shape (Batch, Seq_len, d_model).
        """
        # Self-attention with residual and layernorm
        residual = x
        x_attn, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(residual + self.dropout(x_attn))

        # 1D Convolutional feed-forward network with residual and layernorm
        residual = x
        # Transpose to (Batch, Channels, Seq_len) for Conv1D
        x_conv = x.transpose(1, 2)
        x_conv = self.conv2(F.relu(self.conv1(x_conv)))
        x_conv = x_conv.transpose(1, 2)
        x = self.norm2(residual + self.dropout(x_conv))

        return x


class StethoSpeechModel(nn.Module):
    """
    StethoSpeech FFT Architecture (Executed Phase 4B Baseline).

    6 FFT Encoder blocks + 4 FFT Decoder blocks.
    Parameter count: exactly 29,263,360 parameters.
    """

    def __init__(
        self,
        hubert_dim: int = 768,
        d_model: int = 256,
        enc_layers: int = 6,
        dec_layers: int = 4,
        n_head: int = 2,
        d_inner: int = 1024,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        max_len: int = 1500,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 1]

        self.hubert_dim = hubert_dim
        self.d_model = d_model

        self.enc_proj = nn.Linear(hubert_dim, d_model)
        self.register_buffer("pos_enc", get_sinusoid_table(max_len + 1, d_model))

        self.encoder = nn.ModuleList([
            FFTBlock(d_model, n_head, d_inner, kernel_sizes, dropout)
            for _ in range(enc_layers)
        ])

        self.decoder = nn.ModuleList([
            FFTBlock(d_model, n_head, d_inner, kernel_sizes, dropout)
            for _ in range(dec_layers)
        ])

        self.out_proj = nn.Linear(d_model, hubert_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass mapping input HuBERT representations.

        Args:
            x: Input NAM features of shape (Batch, Seq_len, hubert_dim).
            mask: Optional key padding mask (Batch, Seq_len).
            src_key_padding_mask: Optional alias for mask.

        Returns:
            Predicted speech features of shape (Batch, Seq_len, hubert_dim).
        """
        if mask is None and src_key_padding_mask is not None:
            mask = src_key_padding_mask

        batch_size, seq_len, _ = x.shape

        x = F.relu(self.enc_proj(x))
        x = x + self.pos_enc[:, :seq_len, :].expand(batch_size, -1, -1)

        for enc_block in self.encoder:
            x = enc_block(x, mask)

        x = x + self.pos_enc[:, :seq_len, :].expand(batch_size, -1, -1)

        for dec_block in self.decoder:
            x = dec_block(x, mask)

        return self.out_proj(x)

    def get_parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StethoSpeechWithCTC(StethoSpeechModel):
    """
    StethoSpeech FFT Architecture with Auxiliary CTC Projection Head.

    Augments the base 29.26M FFT model with an intermediate CTC classification head
    projecting encoder states to character/phoneme vocabulary logits:
    Linear(256 -> 768) -> ReLU -> Linear(768 -> ctc_vocab)

    Parameter count: 29,485,344 parameters (with ctc_vocab=32).

    Scientific Disclaimer:
    In historical training runs, this CTC projection was instantiated for paper
    alignment but was decoupled from the loss function during executed optimization.
    """

    def __init__(
        self,
        ctc_vocab: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_vocab = ctc_vocab
        d_model = kwargs.get("d_model", 256)
        hubert_dim = kwargs.get("hubert_dim", 768)

        self.fc_ctc = nn.Linear(d_model, hubert_dim)
        self.ctc_head = nn.Linear(hubert_dim, ctc_vocab)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both predicted speech features and CTC logits.

        Args:
            x: Input NAM features of shape (Batch, Seq_len, hubert_dim).
            mask: Optional key padding mask (Batch, Seq_len).
            src_key_padding_mask: Optional alias for mask.

        Returns:
            Tuple of (predicted_speech_features, ctc_logits).
        """
        if mask is None and src_key_padding_mask is not None:
            mask = src_key_padding_mask

        batch_size, seq_len, _ = x.shape

        x = F.relu(self.enc_proj(x))
        x = x + self.pos_enc[:, :seq_len, :].expand(batch_size, -1, -1)

        for enc_block in self.encoder:
            x = enc_block(x, mask)

        # Compute CTC logits from encoder representation
        ctc_features = F.relu(self.fc_ctc(x))
        ctc_logits = self.ctc_head(ctc_features)

        x = x + self.pos_enc[:, :seq_len, :].expand(batch_size, -1, -1)

        for dec_block in self.decoder:
            x = dec_block(x, mask)

        out_features = self.out_proj(x)
        return out_features, ctc_logits
