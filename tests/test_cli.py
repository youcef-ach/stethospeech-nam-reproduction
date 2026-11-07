"""
Tests for configuration loading, CLI execution, and early stopping behavior.
"""

import os
import tempfile
import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stethospeech.utils.config import load_config
from stethospeech.training.early_stopping import EarlyStopping
from stethospeech.training.trainer import ModelTrainer
from stethospeech.models.transformer_baseline import TransformerSeq2Seq
from stethospeech.data.dataset import collate_stethospeech_batch


def test_yaml_config_loading():
    """Verify loading of YAML experiment configurations."""
    baseline_cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline_transformer.yaml")
    fft_cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experimental_fft.yaml")

    cfg_base = load_config(baseline_cfg_path)
    assert cfg_base.model.architecture == "transformer_baseline"
    assert cfg_base.model.num_layers == 6
    assert cfg_base.training.epochs == 45

    cfg_fft = load_config(fft_cfg_path)
    assert cfg_fft.model.architecture == "stethospeech_fft"
    assert cfg_fft.model.enc_layers == 6
    assert cfg_fft.model.dec_layers == 4


def test_early_stopping_mathematical_resolution():
    """
    Forensic test reproducing the historical Epoch 35 vs Epoch 45 behavior:
    - Best checkpoint saved at Epoch 35 with Val Loss 0.23185.
    - By Epoch 45, Val Loss was 0.23115.
    - Because 0.23115 > (0.23185 - 0.00100 = 0.23085), improvement was insufficient.
    - EarlyStopping halted at Epoch 45, preserving the Epoch 35 checkpoint.
    """
    model = TransformerSeq2Seq()
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, "best.pth")
        es = EarlyStopping(patience=10, delta=0.001, verbose=False)

        # Epoch 35: new best loss 0.23185
        saved_35 = es(0.23185, model, ckpt_path, epoch=35)
        assert saved_35 is True
        assert es.best_epoch == 35
        assert es.best_loss == 0.23185
        assert es.counter == 0

        # Epochs 36-44: shallow drops that fail to exceed delta (0.001)
        simulated_losses = [
            0.23180, 0.23175, 0.23160, 0.23150, 0.23140,
            0.23135, 0.23125, 0.23120, 0.23118,
        ]
        for idx, val in enumerate(simulated_losses, start=36):
            saved = es(val, model, ckpt_path, epoch=idx)
            assert saved is False
            assert es.best_epoch == 35  # Remains epoch 35

        # Epoch 45: reaches 0.23115 (10th consecutive epoch without >= 0.001 improvement)
        saved_45 = es(0.23115, model, ckpt_path, epoch=45)
        assert saved_45 is False
        assert es.counter == 10
        assert es.early_stop is True  # Halted!
        assert es.best_epoch == 35  # Best checkpoint correctly remains Epoch 35


def test_trainer_cpu_smoke_step():
    """Verify trainer executes a forward/backward/eval step on CPU without error."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Minimal 1-layer transformer for ultra-fast smoke test
    model = TransformerSeq2Seq(num_layers=1, hidden_dim=64, dim_ff=128, nheads=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic batch
    dummy_data = [
        (torch.randn(15, 768), torch.randn(18, 768)),
        (torch.randn(20, 768), torch.randn(20, 768)),
    ]
    loader = DataLoader(dummy_data, batch_size=2, collate_fn=collate_stethospeech_batch)

    trainer = ModelTrainer(model=model, optimizer=optimizer, device=device)
    train_loss = trainer.train_epoch(loader)
    eval_loss = trainer.evaluate(loader)

    assert train_loss > 0 and not torch.isnan(torch.tensor(train_loss))
    assert eval_loss > 0 and not torch.isnan(torch.tensor(eval_loss))
