"""
Tests for neural translation architectures and invariants.

Validates exact parameter counts, tensor dimension preservation, and attention invariants:
- Phase 4A Transformer Seq2Seq: exactly 19,702,016 parameters
- Phase 4B StethoSpeech FFT: exactly 29,263,360 parameters
- Phase 4B with CTC Projection Head: exactly 29,485,344 parameters
"""

import pytest
import torch
from stethospeech.models.transformer_baseline import TransformerSeq2Seq, PositionalEncoding
from stethospeech.models.stethospeech_fft import StethoSpeechModel, StethoSpeechWithCTC, FFTBlock


def test_phase_4a_parameter_count():
    """Verify Phase 4A Transformer baseline matches historical 19.7M parameter count."""
    model = TransformerSeq2Seq(
        input_dim=768,
        hidden_dim=512,
        nheads=8,
        num_layers=6,
        dim_ff=2048,
        dropout=0.1,
    )
    total_params = model.get_parameter_count()
    assert total_params == 19_702_016, (
        f"Phase 4A parameter count discrepancy: expected exactly 19,702,016, got {total_params:,}"
    )


def test_phase_4a_forward_pass_dimensions(dummy_nam_batch):
    """Verify Phase 4A forward pass preserves batch and temporal dimensions."""
    model = TransformerSeq2Seq()
    model.eval()

    batch_size, seq_len, feat_dim = dummy_nam_batch.shape
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, 40:] = True  # Padded frames

    with torch.no_grad():
        out = model(dummy_nam_batch, src_key_padding_mask=padding_mask)

    assert out.shape == (batch_size, seq_len, feat_dim), (
        f"Expected output shape {(batch_size, seq_len, feat_dim)}, got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output tensor contains NaNs"


def test_positional_encoding_shape_invariance():
    """Verify positional encoding preserves input tensor shape."""
    pe = PositionalEncoding(d_model=512, max_len=1000)
    x = torch.randn(4, 75, 512)
    out = pe(x)
    assert out.shape == (4, 75, 512)
    assert not torch.allclose(out, x), "Positional encoding must alter representations"


def test_fft_block_temporal_invariance():
    """Verify FFT block preserves temporal length and feature dimension."""
    fft_block = FFTBlock(d_model=256, n_head=2, d_inner=1024, kernel_sizes=[9, 1], dropout=0.1)
    fft_block.eval()

    x = torch.randn(2, 64, 256)
    with torch.no_grad():
        out = fft_block(x)

    assert out.shape == (2, 64, 256), f"Expected shape (2, 64, 256), got {out.shape}"
    assert not torch.isnan(out).any()


def test_phase_4b_executed_model_parameter_count():
    """Verify Phase 4B StethoSpeech executed model matches historical 29,263,360 parameters."""
    model = StethoSpeechModel(
        hubert_dim=768,
        d_model=256,
        enc_layers=6,
        dec_layers=4,
        n_head=2,
        d_inner=1024,
        kernel_sizes=[9, 1],
        dropout=0.1,
    )
    total_params = model.get_parameter_count()
    assert total_params == 29_263_360, (
        f"Phase 4B parameter count discrepancy: expected exactly 29,263,360, got {total_params:,}"
    )


def test_phase_4b_forward_pass_dimensions():
    """Verify Phase 4B forward pass shape."""
    model = StethoSpeechModel()
    model.eval()

    dummy_input = torch.randn(2, 60, 768)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 60, 768), f"Expected (2, 60, 768), got {out.shape}"
    assert not torch.isnan(out).any()


def test_phase_4b_with_ctc_head():
    """
    Verify Phase 4B with auxiliary CTC projection head:
    29,485,344 parameters (29,263,360 base + 221,984 CTC weights).
    """
    model_ctc = StethoSpeechWithCTC(
        hubert_dim=768,
        d_model=256,
        enc_layers=6,
        dec_layers=4,
        n_head=2,
        d_inner=1024,
        kernel_sizes=[9, 1],
        dropout=0.1,
        ctc_vocab=32,
    )
    total_params = model_ctc.get_parameter_count()
    assert total_params == 29_485_344, (
        f"Phase 4B with CTC parameter count discrepancy: expected exactly 29,485,344, got {total_params:,}"
    )

    dummy_input = torch.randn(2, 50, 768)
    model_ctc.eval()
    with torch.no_grad():
        out_features, ctc_logits = model_ctc(dummy_input)

    assert out_features.shape == (2, 50, 768), f"Expected features shape (2, 50, 768), got {out_features.shape}"
    assert ctc_logits.shape == (2, 50, 32), f"Expected CTC logits shape (2, 50, 32), got {ctc_logits.shape}"
