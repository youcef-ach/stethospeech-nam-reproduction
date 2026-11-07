"""
Tests for dataset pairing, splitting logic, and batch collation.
"""

import os
import pytest
import numpy as np
import torch
from stethospeech.data.dataset import (
    find_matched_feature_pairs,
    create_speaker_dependent_splits,
    StethoSpeechPairDataset,
    collate_stethospeech_batch,
)


def test_find_matched_feature_pairs(temp_feature_dirs):
    """Verify that only intersected pairs present in both directories are retained."""
    nam_dir, speech_dir = temp_feature_dirs
    matched = find_matched_feature_pairs(nam_dir, speech_dir)

    # 10 matched files created in fixture, ignoring 2 unmatched nam files
    assert len(matched) == 10
    assert "s01_unmatched_1" not in matched
    assert "s01_000" in matched


def test_missing_feature_directory_raises():
    """Verify FileNotFoundError is raised when directory paths do not exist."""
    with pytest.raises(FileNotFoundError):
        find_matched_feature_pairs("non_existent_nam_dir", "non_existent_speech_dir")


def test_speaker_dependent_splits():
    """Verify 80/10/10 deterministic split ratios."""
    file_list = [f"utterance_{i:04d}" for i in range(100)]
    splits = create_speaker_dependent_splits(
        file_list,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_state=42,
    )

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    # Ensure mutual exclusivity across splits
    all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_assigned) == 100


def test_stethospeech_pair_dataset(temp_feature_dirs):
    """Verify dataset loads numpy arrays into float tensors."""
    nam_dir, speech_dir = temp_feature_dirs
    matched = find_matched_feature_pairs(nam_dir, speech_dir)

    dataset = StethoSpeechPairDataset(nam_dir, speech_dir, matched)
    assert len(dataset) == 10

    nam_tensor, speech_tensor = dataset[0]
    assert isinstance(nam_tensor, torch.Tensor)
    assert isinstance(speech_tensor, torch.Tensor)
    assert nam_tensor.dtype == torch.float32
    assert speech_tensor.dtype == torch.float32
    assert nam_tensor.shape[1] == 768
    assert speech_tensor.shape[1] == 768


def test_stethospeech_pair_dataset_missing_file_raises(temp_feature_dirs):
    """Verify FileNotFoundError is raised if a listed file is missing on disk."""
    nam_dir, speech_dir = temp_feature_dirs
    dataset = StethoSpeechPairDataset(nam_dir, speech_dir, ["non_existent_id"])

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_collate_stethospeech_batch():
    """Verify batch collate correctly pads variable-length sequences and produces boolean mask."""
    item1 = (torch.randn(30, 768), torch.randn(35, 768))
    item2 = (torch.randn(50, 768), torch.randn(45, 768))
    batch = [item1, item2]

    nam_padded, speech_padded, nam_lens, speech_lens, nam_mask = collate_stethospeech_batch(batch)

    assert nam_padded.shape == (2, 50, 768)
    assert speech_padded.shape == (2, 45, 768)
    assert torch.equal(nam_lens, torch.tensor([30, 50]))
    assert torch.equal(speech_lens, torch.tensor([35, 45]))

    # Padded frames in item 0 (indices 30 to 49) must be True in attention mask
    assert (nam_mask[0, 30:] == True).all()
    assert (nam_mask[0, :30] == False).all()
    # Item 1 has length 50, so none should be padded
    assert (nam_mask[1, :] == False).all()
