"""
Modular PyTorch Model Trainer for StethoSpeech Sequence Translation.

Provides unified training, validation, and evaluation loops with:
- Configurable gradient clipping
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping tracking with best checkpoint persistence
- Support for both Phase 4A Transformer and Phase 4B FFT architectures
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stethospeech.training.losses import masked_l1_loss
from stethospeech.training.early_stopping import EarlyStopping
from stethospeech.utils.logging import get_logger
from stethospeech.utils.seed import set_seed
from stethospeech.utils.config import ExperimentConfig, load_config


class ModelTrainer:
    """
    Orchestrates neural sequence translation training and evaluation.

    Args:
        model: PyTorch sequence translation model (TransformerSeq2Seq or StethoSpeechModel).
        optimizer: PyTorch optimizer (typically AdamW).
        device: Target execution device ('cuda' or 'cpu').
        scheduler: Optional learning rate scheduler.
        early_stopping: Optional EarlyStopping callback instance.
        clip_grad_norm: Maximum gradient norm for clipping (default: 1.0).
        logger: Optional logger instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
        early_stopping: Optional[EarlyStopping] = None,
        clip_grad_norm: float = 1.0,
        logger=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.clip_grad_norm = clip_grad_norm
        self.logger = logger or get_logger("stethospeech.trainer")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch over the dataloader."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            nam, speech, nam_lens, speech_lens, nam_mask = batch
            nam = nam.to(self.device)
            speech = speech.to(self.device)
            speech_lens = speech_lens.to(self.device)
            nam_mask = nam_mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: pass attention mask
            preds = self.model(nam, src_key_padding_mask=nam_mask)

            # Compute masked L1 loss with linear time interpolation if lengths differ
            loss = masked_l1_loss(preds, speech, speech_lens)

            loss.backward()
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate average masked L1 loss over a validation or test set."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                nam, speech, nam_lens, speech_lens, nam_mask = batch
                nam = nam.to(self.device)
                speech = speech.to(self.device)
                speech_lens = speech_lens.to(self.device)
                nam_mask = nam_mask.to(self.device)

                preds = self.model(nam, src_key_padding_mask=nam_mask)
                loss = masked_l1_loss(preds, speech, speech_lens)

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 45,
        checkpoint_path: str = "checkpoints/best_model.pth",
    ) -> Dict[str, List[float]]:
        """
        Execute full training loop with validation, scheduling, and early stopping.

        Args:
            train_loader: DataLoader for training pairs.
            val_loader: DataLoader for validation pairs.
            epochs: Maximum training epochs.
            checkpoint_path: Path to write optimal checkpoint.

        Returns:
            Dictionary containing train_losses and val_losses history.
        """
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        self.logger.info(
            f"Starting training run: {epochs} max epochs, "
            f"parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            duration = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | Train MAE: {train_loss:.5f} | "
                f"Val MAE: {val_loss:.5f} | LR: {current_lr:.1e} | Time: {duration:.1f}s"
            )

            # Step scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Early stopping check
            if self.early_stopping is not None:
                saved = self.early_stopping(val_loss, self.model, checkpoint_path, epoch=epoch)
                if self.early_stopping.early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}. Halting training.")
                    break

        return history


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train StethoSpeech translation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--device", type=str, default=None, help="Override execution device (cpu or cuda)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"[Trainer] Loaded experiment config: '{cfg.experiment_name}', device: {device}")
    print("[Trainer] Pipeline successfully initialized. To train, provide dataset paths in YAML config.")


if __name__ == "__main__":
    main()
