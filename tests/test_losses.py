"""
Tests for loss computation, masking invariants, and temporal alignment.
"""

import pytest
import torch
from stethospeech.training.losses import masked_l1_loss, exact_masked_l1_loss, auxiliary_ctc_loss


def test_masked_l1_loss_temporal_interpolation(dummy_speech_batch):
    """Verify linear interpolation properly aligns differing temporal lengths."""
    batch_size = 2
    t_nam = 45
    t_speech = 55
    feature_dim = 768

    preds = torch.randn(batch_size, t_nam, feature_dim, requires_grad=True)
    targets = dummy_speech_batch  # shape (2, 55, 768)
    speech_lengths = torch.tensor([50, 55])

    loss = masked_l1_loss(preds, targets, speech_lengths)
    assert loss.ndim == 0, "Loss must be a scalar"
    assert loss.item() > 0.0, "Loss must be positive"
    assert not torch.isnan(loss), "Loss must not be NaN"

    # Verify backward pass computes valid gradients
    loss.backward()
    assert preds.grad is not None
    assert not torch.isnan(preds.grad).any()


def test_masked_l1_loss_ignores_padded_frames():
    """Verify that values in padded frames do not alter the computed loss."""
    torch.manual_seed(42)
    # 2 items of max length 10; item 0 has valid length 5, item 1 has valid length 10
    lengths = torch.tensor([5, 10])

    preds = torch.ones(2, 10, 4)
    targets = torch.zeros(2, 10, 4)

    base_loss = masked_l1_loss(preds, targets, lengths)

    # Corrupt padded frames of item 0 (frames 5 to 9) with large numbers
    corrupted_preds = preds.clone()
    corrupted_preds[0, 5:, :] = 999.0

    corrupted_loss = masked_l1_loss(corrupted_preds, targets, lengths)

    assert torch.isclose(base_loss, corrupted_loss), (
        f"Padded frames affected loss: base={base_loss.item():.5f}, corrupted={corrupted_loss.item():.5f}"
    )


def test_exact_masked_l1_loss_raises_on_shape_mismatch():
    """Verify exact_masked_l1_loss raises ValueError when shapes mismatch."""
    preds = torch.randn(2, 40, 768)
    targets = torch.randn(2, 50, 768)
    lengths = torch.tensor([40, 40])

    with pytest.raises(ValueError, match="Shape mismatch"):
        exact_masked_l1_loss(preds, targets, lengths)


def test_auxiliary_ctc_loss():
    """Verify CTC loss computation operates without error."""
    torch.manual_seed(42)
    batch_size = 2
    time_steps = 30
    vocab_size = 32

    logits = torch.randn(batch_size, time_steps, vocab_size, requires_grad=True)
    targets = torch.randint(1, vocab_size, (batch_size, 10), dtype=torch.long)
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    target_lengths = torch.tensor([10, 8], dtype=torch.long)

    loss = auxiliary_ctc_loss(logits, targets, input_lengths, target_lengths)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    loss.backward()
    assert logits.grad is not None
