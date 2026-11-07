"""
Discrete unit quantization using MiniBatchKMeans.

Maps continuous 768-dimensional self-supervised speech representations
(mHuBERT Layer 11) into discrete cluster indices [0, 999] for consumption
by the Unit HiFi-GAN neural vocoder.
"""

import os
from typing import Optional, Union
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
import joblib


class KMeansUnitQuantizer:
    """
    Quantizer converting continuous HuBERT embeddings to discrete unit tokens.

    Args:
        num_clusters: Number of discrete acoustic units (default: 1000).
        random_state: Seed for initialization (default: 42).
    """

    def __init__(
        self,
        num_clusters: int = 1000,
        random_state: int = 42,
        n_clusters: Optional[int] = None,
    ):
        if n_clusters is not None:
            num_clusters = n_clusters
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.model: Optional[MiniBatchKMeans] = None

    def fit(self, features: np.ndarray, batch_size: int = 10000, max_iter: int = 100) -> "KMeansUnitQuantizer":
        """
        Train K-Means clustering on continuous speech feature frames.

        Args:
            features: 2D array of shape (Total_frames, 768).
            batch_size: Mini-batch size for clustering.
            max_iter: Maximum optimization iterations.

        Returns:
            self
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array (N_frames, Dim), got shape {features.shape}")

        self.model = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=self.random_state,
            n_init="auto",
        )
        self.model.fit(features)
        return self

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Assign frames to their nearest centroid.

        Args:
            features: Array or Tensor of shape (Time_frames, 768) or (Batch, Time, 768).

        Returns:
            1D or 2D array of discrete unit integers [0, num_clusters - 1].
        """
        if self.model is None:
            raise RuntimeError("Quantizer model has not been trained or loaded.")

        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = np.asarray(features)

        if features_np.ndim == 2:
            return self.model.predict(features_np)
        elif features_np.ndim == 3:
            b, t, d = features_np.shape
            flat = features_np.reshape(-1, d)
            preds = self.model.predict(flat)
            return preds.reshape(b, t)
        else:
            raise ValueError(f"Expected 2D or 3D feature array, got shape {features_np.shape}")

    def save(self, model_path: str) -> None:
        """Save fitted quantizer to disk via joblib."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted quantizer model.")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, model_path: str) -> "KMeansUnitQuantizer":
        """
        Load fitted quantizer from file.

        .. warning::
            Security Notice: joblib uses Python's pickle protocol internally and is
            NOT inherently secure against arbitrary code execution. Checkpoints must
            come strictly from trusted sources. Never load untrusted or unverified
            .joblib or .bin files.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"K-Means checkpoint file not found: {model_path}")
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.num_clusters = getattr(instance.model, "n_clusters", 1000)
        return instance
