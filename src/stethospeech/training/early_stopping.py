"""
Early stopping tracker and checkpoint manager.

Reproduces the exact convergence and checkpoint-saving behavior observed
during the historical 45-epoch baseline training run:
- Patience: 10 epochs
- Min delta: 0.001 MAE
- Best model saved at Epoch 35 (Val Loss 0.23185)
- Halted at Epoch 45 when subsequent val loss (0.23115) did not exceed the required delta
  improvement threshold (val_loss < 0.23085).
"""

from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping monitor based on validation loss improvement.

    Args:
        patience: Number of epochs to wait for an improvement exceeding delta.
        delta: Minimum change in monitored loss to qualify as an improvement.
        verbose: If True, prints messages upon saving checkpoints.
    """

    def __init__(self, patience: int = 10, delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        checkpoint_path: str,
        epoch: int = 0,
    ) -> bool:
        """
        Evaluate validation loss and update checkpoint state.

        Args:
            val_loss: Current epoch validation loss.
            model: PyTorch model whose state_dict should be saved.
            checkpoint_path: Destination path for best model checkpoint.
            epoch: Current epoch index.

        Returns:
            True if a new best checkpoint was saved, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint(val_loss, model, checkpoint_path)
            return True

        if val_loss < self.best_loss - self.delta:
            # Significant improvement observed
            if self.verbose:
                print(
                    f"  [EarlyStopping] Val loss improved from {self.best_loss:.5f} "
                    f"to {val_loss:.5f} (Δ > {self.delta}). Saving checkpoint..."
                )
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] Counter {self.counter}/{self.patience} "
                    f"(current: {val_loss:.5f}, best: {self.best_loss:.5f}, required: <{self.best_loss - self.delta:.5f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"  [EarlyStopping] Patience limit reached at epoch {epoch}. "
                        f"Best checkpoint preserved from epoch {self.best_epoch} with val loss {self.best_loss:.5f}."
                    )
            return False

    def _save_checkpoint(self, val_loss: float, model: nn.Module, path: str) -> None:
        """Persist model state_dict to disk."""
        torch.save(
            {
                "best_loss": val_loss,
                "best_epoch": self.best_epoch,
                "state_dict": model.state_dict(),
            },
            path,
        )
