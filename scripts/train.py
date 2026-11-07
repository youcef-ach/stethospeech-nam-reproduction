"""
Command-line interface to execute model training or smoke tests.
"""

import argparse
import sys
import torch
import torch.optim as optim

from stethospeech.utils.config import load_config
from stethospeech.utils.seed import set_seed
from stethospeech.models.transformer_baseline import TransformerSeq2Seq
from stethospeech.models.stethospeech_fft import StethoSpeechModel
from stethospeech.training.trainer import ModelTrainer
from stethospeech.training.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description="Train StethoSpeech translation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--device", type=str, default=None, help="Execution device: cpu or cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"=== StethoSpeech Experiment: {cfg.experiment_name} ===")
    print(f"Target Device: {device}")

    # Build model architecture based on config
    if cfg.model.architecture == "transformer_baseline":
        model = TransformerSeq2Seq(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            nheads=cfg.model.nheads,
            num_layers=cfg.model.num_layers,
            dim_ff=cfg.model.dim_ff,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.architecture == "stethospeech_fft":
        model = StethoSpeechModel(
            hubert_dim=cfg.model.input_dim,
            d_model=cfg.model.hidden_dim,
            enc_layers=cfg.model.enc_layers,
            dec_layers=cfg.model.dec_layers,
            n_head=cfg.model.nheads,
            d_inner=cfg.model.d_inner,
            kernel_sizes=cfg.model.kernel_sizes,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Instantiated model: {cfg.model.architecture} ({param_count:,} parameters)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_factor,
        patience=cfg.training.lr_patience,
    )

    early_stopping = EarlyStopping(
        patience=cfg.training.early_stop_patience,
        delta=cfg.training.early_stop_delta,
    )

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        clip_grad_norm=cfg.training.clip_grad_norm,
    )

    print("Trainer successfully initialized from configuration.")
    print("To execute training over the full corpus, verify feature directory paths in the configuration YAML.")


if __name__ == "__main__":
    main()
