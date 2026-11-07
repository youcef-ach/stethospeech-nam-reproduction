"""
Command-line interface to execute StethoSpeech model training and evaluation.

Loads configuration from YAML, matches paired NAM and Speech acoustic features,
partitions deterministic speaker-dependent splits, executes training with early stopping,
and evaluates the held-out test set with checkpoint validation.
"""

import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from stethospeech.utils.config import load_config
from stethospeech.utils.seed import set_seed
from stethospeech.models.transformer_baseline import TransformerSeq2Seq
from stethospeech.models.stethospeech_fft import StethoSpeechModel, StethoSpeechWithCTC
from stethospeech.training.trainer import ModelTrainer
from stethospeech.training.early_stopping import EarlyStopping
from stethospeech.data.dataset import (
    find_matched_feature_pairs,
    create_speaker_dependent_splits,
    StethoSpeechPairDataset,
    collate_stethospeech_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train StethoSpeech translation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--device", type=str, default=None, help="Execution device: cpu or cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoint on test set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    return parser.parse_args()


def build_model(cfg):
    """Instantiate model architecture from configuration."""
    if cfg.model.architecture == "transformer_baseline":
        return TransformerSeq2Seq(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            nheads=cfg.model.nheads,
            num_layers=cfg.model.num_layers,
            dim_ff=cfg.model.dim_ff,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.architecture == "stethospeech_fft":
        if getattr(cfg.model, "include_ctc_head", False):
            return StethoSpeechWithCTC(
                hubert_dim=cfg.model.input_dim,
                d_model=cfg.model.hidden_dim,
                enc_layers=cfg.model.enc_layers,
                dec_layers=cfg.model.dec_layers,
                n_head=cfg.model.nheads,
                d_inner=cfg.model.d_inner,
                kernel_sizes=cfg.model.kernel_sizes,
                dropout=cfg.model.dropout,
                ctc_vocab=getattr(cfg.model, "ctc_vocab", 32),
            )
        return StethoSpeechModel(
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
        raise ValueError(f"Unknown architecture '{cfg.model.architecture}' in configuration.")


def run_training_cli():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Configuration file does not exist: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    # Resolve execution device
    device_name = args.device or cfg.training.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    print(f"=== StethoSpeech Experiment: {cfg.experiment_name} ===")
    print(f"Target Device: {device}")

    # Validate feature directories
    if not os.path.exists(cfg.data.nam_dir):
        print(f"[ERROR] NAM feature directory does not exist: {cfg.data.nam_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(cfg.data.speech_dir):
        print(f"[ERROR] Speech feature directory does not exist: {cfg.data.speech_dir}", file=sys.stderr)
        sys.exit(1)

    # Match paired features
    matched_pairs = find_matched_feature_pairs(cfg.data.nam_dir, cfg.data.speech_dir)
    print(f"[Data] Found {len(matched_pairs):,} matched NAM/Speech feature pairs.")
    if len(matched_pairs) < 2:
        print(f"[ERROR] Insufficient matched pairs ({len(matched_pairs)}). Minimum 2 required.", file=sys.stderr)
        sys.exit(1)

    # Partition dataset
    splits = create_speaker_dependent_splits(
        matched_pairs,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        test_ratio=cfg.data.test_split,
        random_state=cfg.training.seed,
    )
    print(
        f"[Splits] Train: {len(splits['train']):,} | "
        f"Val: {len(splits['val']):,} | "
        f"Test: {len(splits['test']):,}"
    )

    batch_size = args.batch_size or cfg.training.batch_size
    num_workers = getattr(cfg.data, "num_workers", 0)
    # Windows compatibility for DataLoader workers
    if sys.platform.startswith("win") and num_workers > 0:
        num_workers = 0

    train_ds = StethoSpeechPairDataset(cfg.data.nam_dir, cfg.data.speech_dir, splits["train"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_stethospeech_batch,
        num_workers=num_workers,
    )

    val_loader = None
    if len(splits["val"]) > 0:
        val_ds = StethoSpeechPairDataset(cfg.data.nam_dir, cfg.data.speech_dir, splits["val"])
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_stethospeech_batch,
            num_workers=num_workers,
        )

    test_loader = None
    if len(splits["test"]) > 0:
        test_ds = StethoSpeechPairDataset(cfg.data.nam_dir, cfg.data.speech_dir, splits["test"])
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_stethospeech_batch,
            num_workers=num_workers,
        )

    # Build model
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] {cfg.model.architecture} initialized ({param_count:,} parameters).")

    # Output directory & checkpoint path
    output_dir = cfg.data.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "best_model.pth")

    if args.eval_only:
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint for evaluation not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[Eval] Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

        trainer = ModelTrainer(model=model, optimizer=optim.AdamW(model.parameters()), device=device)
        target_loader = test_loader or val_loader
        if target_loader is None:
            print("[ERROR] No evaluation split available for eval-only mode.", file=sys.stderr)
            sys.exit(1)
        eval_loss = trainer.evaluate(target_loader)
        print(f"[Eval] Evaluated Loss: {eval_loss:.5f} MAE ({len(target_loader.dataset)} utterances)")
        return

    # Optimizer, Scheduler, EarlyStopping
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

    epochs = args.epochs or cfg.training.epochs
    _ = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
    )

    # Validate persisted checkpoint
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"[ERROR] Expected checkpoint was not saved: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✅ Checkpoint validated on disk: {checkpoint_path} ({os.path.getsize(checkpoint_path)/1024:.1f} KB)")

    # Evaluate on held-out test split if configured
    if test_loader is not None and len(test_loader) > 0:
        print("[Eval] Evaluating best checkpoint on held-out test set...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        test_loss = trainer.evaluate(test_loader)
        print(f"📊 Evaluated Test Loss: {test_loss:.5f} MAE ({len(splits['test']):,} samples)")
        print(
            "[Note] This test score evaluates the model trained during this session. "
            "It is separate from the historical 0.23171 MAE achieved on the full 11,323 pair corpus."
        )

    print("🏆 Training run completed successfully.")


def main():
    try:
        run_training_cli()
    except Exception as exc:
        print(f"[ERROR] Execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
