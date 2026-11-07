"""
Unit HiFi-GAN Neural Vocoder Generator (Pure PyTorch Implementation).

Native PyTorch port of Meta AI's Unit HiFi-GAN vocoder generator
(mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur).

This reimplementation eliminates external fairseq C++ dependencies,
allowing direct model instantiation and weight loading on Python 3.10–3.13.

Architecture:
- Unit Embedding: 1,000 discrete unit IDs -> 128-dim embeddings
- Pre-convolution: Conv1D (128 -> 512, kernel=7)
- 5 Upsampling Transposed Convolutions: rates [5, 4, 4, 2, 2] -> 320x total temporal expansion
  (Converts 50 Hz discrete frame sequences into 16,000 Hz continuous audio waveforms)
- Multi-Receptive Field (MRF) fusion modules across kernel sizes [3, 7, 11]
- Post-convolution: Conv1D (16 -> 1, kernel=7) + Tanh activation
- Exact parameter count: 13,375,809 parameters
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1(nn.Module):
    """
    Residual block with dilated 1D convolutions for multi-scale receptive fields.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class CodeHiFiGANGenerator(nn.Module):
    """
    Discrete Unit HiFi-GAN Vocoder Generator.

    Converts discrete unit indices [0, num_units-1] into continuous 16kHz audio.
    Total temporal upsampling factor is 5 * 4 * 4 * 2 * 2 = 320x.
    """

    def __init__(
        self,
        num_units: int = 1000,
        emb_dim: int = 128,
        upsample_initial_channel: int = 512,
        upsample_rates: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        resblock_kernel_sizes: Optional[List[int]] = None,
        resblock_dilation_sizes: Optional[List[Tuple[int, int, int]]] = None,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [5, 4, 4, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 8, 8, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]

        self.num_units = num_units
        self.emb_dim = emb_dim
        self.upsample_rates = upsample_rates
        self.total_upsample_factor = 1
        for r in upsample_rates:
            self.total_upsample_factor *= r

        self.dict = nn.Embedding(num_units, emb_dim)
        self.conv_pre = nn.Conv1d(emb_dim, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        curr_channel = upsample_initial_channel
        for r, k in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(
                nn.ConvTranspose1d(
                    curr_channel, curr_channel // 2, k, r, padding=(k - r) // 2
                )
            )
            curr_channel //= 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = nn.Conv1d(curr_channel, 1, 7, 1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthesize audio waveform from discrete unit tokens.

        Args:
            x: LongTensor of discrete unit IDs of shape (Batch, Seq_len).

        Returns:
            Audio waveform tensor of shape (Batch, 1, Seq_len * 320)
            with normalized amplitudes bounded in [-1.0, 1.0].
        """
        # x: (Batch, Seq_len) -> (Batch, emb_dim, Seq_len)
        x = self.dict(x).transpose(1, 2)
        x = self.conv_pre(x)

        num_resblocks_per_stage = 3
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(num_resblocks_per_stage):
                idx = i * num_resblocks_per_stage + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / num_resblocks_per_stage

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def get_parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
