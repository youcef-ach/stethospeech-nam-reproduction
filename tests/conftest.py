"""Pytest configuration and shared fixtures for CPU smoke tests."""

import os
import sys
import tempfile
import numpy as np
import pytest
import torch

# Ensure src directory is in sys.path for test discovery
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture
def dummy_nam_batch():
    """Create a synthetic NAM feature batch of shape (2, 45, 768)."""
    torch.manual_seed(42)
    return torch.randn(2, 45, 768)


@pytest.fixture
def dummy_speech_batch():
    """Create a synthetic Speech feature batch of shape (2, 55, 768)."""
    torch.manual_seed(42)
    return torch.randn(2, 55, 768)


@pytest.fixture
def dummy_discrete_units():
    """Create a synthetic discrete unit sequence (50 frames = 1.0s at 50 Hz)."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (1, 50), dtype=torch.long)


@pytest.fixture
def temp_feature_dirs():
    """Create temporary mock NAM and Speech feature directories with matching and non-matching files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nam_dir = os.path.join(temp_dir, "nam")
        speech_dir = os.path.join(temp_dir, "speech")
        os.makedirs(nam_dir)
        os.makedirs(speech_dir)

        # Create 10 matched files and 2 un-matched files (simulating raw 11,591 vs 11,323)
        for i in range(10):
            fid = f"s01_00{i}"
            arr_nam = np.random.randn(30 + i, 768).astype(np.float32)
            arr_spk = np.random.randn(30 + i, 768).astype(np.float32)
            np.save(os.path.join(nam_dir, f"{fid}.npy"), arr_nam)
            np.save(os.path.join(speech_dir, f"{fid}.npy"), arr_spk)

        # Unmatched in speech
        np.save(os.path.join(nam_dir, "s01_unmatched_1.npy"), np.random.randn(20, 768).astype(np.float32))
        np.save(os.path.join(nam_dir, "s01_unmatched_2.npy"), np.random.randn(20, 768).astype(np.float32))

        yield nam_dir, speech_dir
