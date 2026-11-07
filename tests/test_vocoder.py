"""
Tests for Unit HiFi-GAN vocoder generator and acoustic unit quantizer.
"""

import torch
import numpy as np
from stethospeech.models.vocoder import CodeHiFiGANGenerator, ResBlock1
from stethospeech.features.quantization import KMeansUnitQuantizer


def test_vocoder_parameter_count():
    """Verify Unit HiFi-GAN vocoder generator has exactly 13,375,809 parameters."""
    vocoder = CodeHiFiGANGenerator(num_units=1000, emb_dim=128)
    params = vocoder.get_parameter_count()
    assert params == 13_375_809, (
        f"Vocoder parameter count discrepancy: expected 13,375,809, got {params:,}"
    )


def test_vocoder_temporal_upsampling_320x(dummy_discrete_units):
    """
    Verify 320x temporal upsampling:
    50 frames (at 50 Hz = 1.00s) -> 16,000 audio samples (at 16kHz = 1.00s).
    """
    vocoder = CodeHiFiGANGenerator()
    vocoder.eval()

    num_frames = dummy_discrete_units.size(1)
    expected_samples = num_frames * 320  # 50 * 320 = 16,000

    with torch.no_grad():
        waveform = vocoder(dummy_discrete_units)

    assert waveform.shape == (1, 1, expected_samples), (
        f"Expected waveform shape (1, 1, {expected_samples}), got {waveform.shape}"
    )

    # Amplitudes must be bounded by tanh in [-1.0, 1.0]
    assert (waveform >= -1.0).all() and (waveform <= 1.0).all(), (
        "Audio waveform output must be normalized within [-1.0, 1.0]"
    )


def test_resblock1_receptive_fields():
    """Verify ResBlock1 processes channels and maintains temporal dimension."""
    resblock = ResBlock1(channels=256, kernel_size=3, dilation=(1, 3, 5))
    x = torch.randn(2, 256, 100)
    out = resblock(x)
    assert out.shape == (2, 256, 100)
    assert not torch.isnan(out).any()


def test_kmeans_unit_quantizer_fit_predict():
    """Verify K-Means quantizer fits on continuous embeddings and returns integer unit IDs."""
    np.random.seed(42)
    # Generate 150 frames of 768-dim features
    dummy_features = np.random.randn(150, 768).astype(np.float32)

    quantizer = KMeansUnitQuantizer(num_clusters=10, random_state=42)
    quantizer.fit(dummy_features, batch_size=50, max_iter=10)

    # Predict on test frames
    test_frames = np.random.randn(25, 768).astype(np.float32)
    units = quantizer.predict(test_frames)

    assert units.shape == (25,), f"Expected shape (25,), got {units.shape}"
    assert np.all(units >= 0) and np.all(units < 10), "Unit IDs must be within [0, 9]"

