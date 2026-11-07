"""
Modular PyTorch Model Trainer for StethoSpeech Sequence Translation.

Provides unified training, validation, and evaluation loops with:
- Configurable gradient clipping
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping tracking with best checkpoint persistence
- Unified model forwarding adapter supporting both Phase 4A Transformer
  and Phase 4B FFT architectures
"""

import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stethospeech.training.losses import masked_l1_loss
from stethospeech.training.early_stopping import EarlyStopping
from stethospeech.utils.logging import get_logger


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

    def forward_model(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified model forwarding adapter.

        Handles interface differences across model families:
        - TransformerSeq2Seq (supports src_key_padding_mask / mask)
        - StethoSpeechModel (supports mask / src_key_padding_mask)
        - StethoSpeechWithCTC (unpacks features from (features, ctc_logits) tuple)

        Args:
            x: Input feature tensor (Batch, Time, Dim).
            padding_mask: Optional boolean padding mask (Batch, Time).

        Returns:
            Predicted feature tensor (Batch, Time, Dim).
        """
        out = self.model(x, mask=padding_mask, src_key_padding_mask=padding_mask)
        if isinstance(out, tuple):
            return out[0]
        return out

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch over the dataloader."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            nam, speech, nam_lens, speech_lens, nam_mask = batch
            nam = nam.to(self.device)
            speech = speech.to(self.device)
            speech_lens = speech_lens.to(self.device)
            nam_mask = nam_mask.to(self.device)

            self.optimizer.zero_grad()

            preds = self.forward_model(nam, padding_mask=nam_mask)
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

                preds = self.forward_model(nam, padding_mask=nam_mask)
                loss = masked_l1_loss(preds, speech, speech_lens)

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 45,
        checkpoint_path: str = "checkpoints/best_model.pth",
    ) -> Dict[str, List[float]]:
        """
        Execute full training loop with validation, scheduling, and early stopping.

        Args:
            train_loader: DataLoader for training pairs.
            val_loader: Optional DataLoader for validation pairs.
            epochs: Maximum training epochs.
            checkpoint_path: Path to write optimal checkpoint.

        Returns:
            Dictionary containing train_losses and val_losses history.
        """
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Starting training run: {epochs} max epochs, parameters: {param_count:,}"
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            duration = time.time() - start_time
            history["train_loss"].append(train_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if val_loader is not None and len(val_loader) > 0:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)

                self.logger.info(
                    f"Epoch {epoch:03d}/{epochs:03d} | Train MAE: {train_loss:.5f} | "
                    f"Val MAE: {val_loss:.5f} | LR: {current_lr:.1e} | Time: {duration:.1f}s"
                )

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                if self.early_stopping is not None:
                    self.early_stopping(val_loss, self.model, checkpoint_path, epoch=epoch)
                    if self.early_stopping.early_stop:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch}. Halting training."
                        )
                        break
            else:
                self.logger.info(
                    f"Epoch {epoch:03d}/{epochs:03d} | Train MAE: {train_loss:.5f} | "
                    f"LR: {current_lr:.1e} | Time: {duration:.1f}s"
                )
                # When no validation split is configured, save checkpoint at each epoch
                torch.save(
                    {"epoch": epoch, "state_dict": self.model.state_dict(), "train_loss": train_loss},
                    checkpoint_path,
                )

        return history


def main() -> None:
    """CLI entry point for training (delegates to stethospeech.training.cli)."""
    from stethospeech.training.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
