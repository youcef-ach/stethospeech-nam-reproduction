"""
PyTorch Dataset and Data Splitting for StethoSpeech Acoustic Features.

Dataset Accounting & Splitting:
- Historical Raw NAM files: 11,591
- FastSpeech2 Ground-Truth Speech files: 11,323
- Intersected matched pairs: 11,323 pairs (nam_files ∩ speech_files)
- Split strategy: 80% Train (9,058), 10% Val (1,132), 10% Test (1,133) using seed=42
- Classification: Speaker-Dependent (Closed-Speaker) Split. Utterances from
  familiar speakers appear across all subsets. Evaluates sentence generalization,
  not open-speaker cross-validation.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def find_matched_feature_pairs(
    nam_dir: str,
    speech_dir: str,
) -> List[str]:
    """
    Find matching base filenames (.npy) present in both NAM and Speech directories.

    Args:
        nam_dir: Directory containing input NAM feature arrays (*.npy).
        speech_dir: Directory containing target Speech feature arrays (*.npy).

    Returns:
        Sorted list of matched base identifiers (without file extensions).
    """
    if not os.path.exists(nam_dir):
        raise FileNotFoundError(f"NAM feature directory does not exist: {nam_dir}")
    if not os.path.exists(speech_dir):
        raise FileNotFoundError(f"Speech feature directory does not exist: {speech_dir}")

    nam_files = {
        os.path.splitext(f)[0]
        for f in os.listdir(nam_dir)
        if f.endswith(".npy")
    }
    speech_files = {
        os.path.splitext(f)[0]
        for f in os.listdir(speech_dir)
        if f.endswith(".npy")
    }

    matched = sorted(list(nam_files & speech_files))
    return matched


def create_speaker_dependent_splits(
    file_list: List[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """
    Partition matched file IDs into Train, Val, and Test splits.

    Replicates the exact historical 80/10/10 random split:
    - 11,323 total files -> 9,058 Train, 1,132 Val, 1,133 Test.

    Supports test_ratio = 0.0 (e.g. for Phase 4B 95/5 train/val setting).
    Validates that ratios sum to 1.0 and rejects datasets too small to partition.

    Args:
        file_list: List of matched base file IDs.
        train_ratio: Fraction for training (default: 0.80).
        val_ratio: Fraction for validation (default: 0.10).
        test_ratio: Fraction for testing (default: 0.10).
        random_state: Seed for reproducible splitting (default: 42).

    Returns:
        Dictionary mapping "train", "val", "test" to lists of file IDs.
    """
    total = len(file_list)
    if total < 2:
        raise ValueError(
            f"Dataset too small to partition: {total} items provided. At least 2 items required."
        )

    # Validate ratios
    if train_ratio <= 0.0 or val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError(
            f"Split ratios must be non-negative, and train_ratio must be positive. "
            f"Got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0 (within tolerance), got sum={ratio_sum:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    eval_ratio = val_ratio + test_ratio
    if eval_ratio <= 0.0:
        # All train
        return {"train": sorted(file_list), "val": [], "test": []}

    eval_count = int(round(total * eval_ratio))
    if eval_count == 0 or eval_count >= total:
        raise ValueError(
            f"Dataset of {total} items cannot be split with eval_ratio={eval_ratio:.3f}. "
            "Adjust split ratios or provide a larger dataset."
        )

    train_files, eval_files = train_test_split(
        file_list,
        test_size=eval_ratio,
        random_state=random_state,
        shuffle=True,
    )

    if test_ratio <= 0.0:
        # No test split requested (e.g. Phase 4B setting)
        val_files = eval_files
        test_files = []
    elif val_ratio <= 0.0:
        # No validation split requested
        val_files = []
        test_files = eval_files
    else:
        # Both val and test are non-zero
        relative_test_ratio = test_ratio / eval_ratio
        val_files, test_files = train_test_split(
            eval_files,
            test_size=relative_test_ratio,
            random_state=random_state,
            shuffle=True,
        )

    return {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }


class StethoSpeechPairDataset(Dataset):
    """
    Dataset yielding paired (NAM, Speech) acoustic feature tensors.

    Each sample is a 2D array of shape (Time_frames, 768) representing
    HuBERT self-supervised speech representations.
    """

    def __init__(
        self,
        nam_dir: str,
        speech_dir: str,
        file_ids: List[str],
    ):
        self.nam_dir = nam_dir
        self.speech_dir = speech_dir
        self.file_ids = file_ids

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id = self.file_ids[idx]
        nam_path = os.path.join(self.nam_dir, f"{file_id}.npy")
        speech_path = os.path.join(self.speech_dir, f"{file_id}.npy")

        if not os.path.exists(nam_path):
            raise FileNotFoundError(f"Missing NAM feature file: {nam_path}")
        if not os.path.exists(speech_path):
            raise FileNotFoundError(f"Missing Speech feature file: {speech_path}")

        nam_arr = np.load(nam_path)
        speech_arr = np.load(speech_path)

        return (
            torch.from_numpy(nam_arr).float(),
            torch.from_numpy(speech_arr).float(),
        )


def collate_stethospeech_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function with zero-padding and boolean attention masks.

    Args:
        batch: List of tuples (nam_tensor, speech_tensor).

    Returns:
        Tuple of:
        - nam_padded: (Batch, Max_T_nam, 768)
        - speech_padded: (Batch, Max_T_speech, 768)
        - nam_lengths: (Batch,)
        - speech_lengths: (Batch,)
        - nam_padding_mask: (Batch, Max_T_nam) where True indicates padded frames
    """
    nam_tensors, speech_tensors = zip(*batch)

    nam_lens = torch.tensor([x.size(0) for x in nam_tensors], dtype=torch.long)
    speech_lens = torch.tensor([x.size(0) for x in speech_tensors], dtype=torch.long)

    nam_padded = nn.utils.rnn.pad_sequence(nam_tensors, batch_first=True, padding_value=0.0)
    speech_padded = nn.utils.rnn.pad_sequence(speech_tensors, batch_first=True, padding_value=0.0)

    # Boolean key padding mask: True where frame is padding (time_idx >= len)
    max_nam_len = nam_padded.size(1)
    time_indices = torch.arange(max_nam_len).unsqueeze(0)
    nam_padding_mask = time_indices >= nam_lens.unsqueeze(1)

    return nam_padded, speech_padded, nam_lens, speech_lens, nam_padding_mask
