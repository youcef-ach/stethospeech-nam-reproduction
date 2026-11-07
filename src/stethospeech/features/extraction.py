"""
Self-supervised speech feature extraction (HuBERT / mHuBERT).

Distinguishes between:
- Phase 4A Baseline: English HuBERT Base (`facebook/hubert-base-ls960`), Layer 12 representation.
- Phase 4B Experimental: Multilingual mHuBERT Base (`mhubert_base_vp_en_es_fr_it3`),
  Layer 11 representation (matching Meta AI's Unit HiFi-GAN vocoder specification).
"""

from typing import Optional, Union
import numpy as np
import torch
import torchaudio


def extract_hubert_features(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    target_layer: int = 12,
    model_name: str = "facebook/hubert-base-ls960",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract self-supervised representations from raw audio waveforms.

    Args:
        waveform: 1D or 2D audio tensor (channels, time) or (time,).
        sample_rate: Audio sampling frequency in Hz (resampled to 16kHz if different).
        target_layer: HuBERT transformer layer to extract (Layer 12 for 4A, Layer 11 for 4B).
        model_name: HuggingFace model identifier or local directory.
        device: PyTorch device for execution.

    Returns:
        Tensor of shape (1, Time_frames, 768) at 50 Hz frame rate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resample if audio is not 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono if multi-channel
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Lazy-load transformers to avoid hard dependency on non-feature extraction paths
    from transformers import HubertModel

    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    waveform = waveform.to(device)
    with torch.no_grad():
        outputs = model(waveform, output_hidden_states=True)
        # hidden_states: tuple of (batch, time, 768) for embeddings + 12 transformer layers
        hidden_states = outputs.hidden_states
        if target_layer < 0 or target_layer >= len(hidden_states):
            raise ValueError(
                f"Invalid target layer {target_layer}. Model provides 0 to {len(hidden_states) - 1} states."
            )
        layer_features = hidden_states[target_layer]

    return layer_features
