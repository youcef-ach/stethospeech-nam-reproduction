"""
Loss functions for sequence-to-sequence acoustic feature translation.

Implements masked L1 loss (Mean Absolute Error) with dynamic temporal
interpolation, aligning with the empirical methodology used during the
historical StethoSpeech experiments.
"""

import torch
import torch.nn.functional as F


def masked_l1_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked L1 loss with linear time interpolation for length alignment.

    When source and target have different frame counts (e.g. Due to silent
    padding variations between NAM and speech channels), linear interpolation
    resamples predictions to match target temporal length before computing loss.

    Args:
        preds: Predicted features of shape (Batch, T_pred, Feature_dim).
        targets: Target speech features of shape (Batch, T_target, Feature_dim).
        target_lengths: Valid unpadded lengths of target items (Batch,).

    Returns:
        Scalar tensor containing the masked mean absolute error.
    """
    if preds.shape[1] != targets.shape[1]:
        # Resample temporal dimension: (B, T_pred, D) -> (B, D, T_pred)
        preds_t = preds.transpose(1, 2)
        preds_interp = F.interpolate(
            preds_t,
            size=targets.shape[1],
            mode="linear",
            align_corners=False,
        )
        preds = preds_interp.transpose(1, 2)

    # Construct boolean mask for valid time frames
    # mask: (Batch, T_target) -> True for valid, False for padded
    batch_size, max_len, _ = targets.shape
    time_indices = torch.arange(max_len, device=targets.device).unsqueeze(0)
    valid_mask = time_indices < target_lengths.unsqueeze(1)
    mask = valid_mask.unsqueeze(-1).expand_as(targets)

    loss = torch.abs(preds - targets) * mask
    total_elements = mask.sum()
    if total_elements == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    return loss.sum() / total_elements


def exact_masked_l1_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked L1 loss assuming identical sequence lengths.

    Args:
        preds: Predicted features of shape (Batch, T, Feature_dim).
        targets: Target features of shape (Batch, T, Feature_dim).
        lengths: Valid unpadded sequence lengths of shape (Batch,).

    Returns:
        Scalar tensor containing the masked mean absolute error.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}. "
            "Use masked_l1_loss for dynamic temporal interpolation."
        )

    batch_size, max_len, _ = targets.shape
    time_indices = torch.arange(max_len, device=targets.device).unsqueeze(0)
    valid_mask = time_indices < lengths.unsqueeze(1)
    mask = valid_mask.unsqueeze(-1).expand_as(targets)

    loss = torch.abs(preds - targets) * mask
    total_elements = mask.sum()
    if total_elements == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    return loss.sum() / total_elements


def auxiliary_ctc_loss(
    ctc_logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """
    Compute Connectionist Temporal Classification (CTC) loss for auxiliary head.

    Note: In historical Phase 4B experiments, this auxiliary objective was
    decoupled and omitted from active training runs. Provided here for completeness
    and future multi-task research.

    Args:
        ctc_logits: Logits of shape (Batch, Time, Vocab_size).
        targets: Target labels of shape (Batch, Target_len).
        input_lengths: Valid length of inputs (Batch,).
        target_lengths: Valid length of targets (Batch,).
        blank: Blank token index (default: 0).

    Returns:
        Scalar CTC loss tensor.
    """
    # PyTorch ctc_loss expects log_probs of shape (Time, Batch, Vocab_size)
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )
