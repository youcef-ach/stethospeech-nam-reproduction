"""
Tests for configuration loading, CLI execution, early stopping, and checkpoint hardening.
"""

import os
import shutil
import site
import subprocess
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from stethospeech.utils.config import load_config
from stethospeech.training.early_stopping import EarlyStopping
from stethospeech.training.trainer import ModelTrainer
from stethospeech.models.transformer_baseline import TransformerSeq2Seq
from stethospeech.models.vocoder import CodeHiFiGANGenerator
from stethospeech.features.quantization import KMeansUnitQuantizer
from stethospeech.data.dataset import collate_stethospeech_batch
from stethospeech.inference.synthesizer import (
    load_model_checkpoint,
    load_discrete_units,
    load_features,
)


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

    model = TransformerSeq2Seq(num_layers=1, hidden_dim=64, dim_ff=128, nheads=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

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


def test_e2e_cli_training_baseline():
    """End-to-end CLI integration test: train Transformer baseline on synthetic features."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "train.py")

    with tempfile.TemporaryDirectory() as temp_dir:
        nam_dir = os.path.join(temp_dir, "nam")
        spk_dir = os.path.join(temp_dir, "spk")
        out_dir = os.path.join(temp_dir, "run")
        os.makedirs(nam_dir)
        os.makedirs(spk_dir)

        # Generate 10 synthetic feature pairs
        for i in range(10):
            fid = f"s01_{i:03d}"
            np.save(os.path.join(nam_dir, f"{fid}.npy"), np.random.randn(25 + i, 768).astype(np.float32))
            np.save(os.path.join(spk_dir, f"{fid}.npy"), np.random.randn(25 + i, 768).astype(np.float32))

        cfg_dict = {
            "experiment_name": "cli_smoke_test_baseline",
            "model": {
                "architecture": "transformer_baseline",
                "input_dim": 768,
                "hidden_dim": 64,
                "output_dim": 768,
                "num_layers": 1,
                "nheads": 2,
                "dim_ff": 128,
                "dropout": 0.0,
            },
            "training": {
                "seed": 42,
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "clip_grad_norm": 1.0,
                "lr_patience": 2,
                "lr_factor": 0.5,
                "early_stop_patience": 5,
                "early_stop_delta": 0.001,
                "device": "cpu",
            },
            "data": {
                "nam_dir": nam_dir,
                "speech_dir": spk_dir,
                "output_dir": out_dir,
                "train_split": 0.80,
                "val_split": 0.10,
                "test_split": 0.10,
                "num_workers": 0,
            },
        }

        cfg_path = os.path.join(temp_dir, "test_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f)

        # Run training CLI via subprocess
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(repo_root, "src")
        cmd = [sys.executable, script_path, "--config", cfg_path, "--device", "cpu", "--epochs", "2"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)

        assert result.returncode == 0, f"Training CLI failed with code {result.returncode}:\n{result.stderr}\n{result.stdout}"
        assert "Training run completed successfully." in result.stdout

        # Verify checkpoint was written
        ckpt_path = os.path.join(out_dir, "best_model.pth")
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 0


def test_e2e_cli_training_fft_phase_4b():
    """End-to-end CLI integration test: train StethoSpeech FFT architecture with test_split=0.0."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "train.py")

    with tempfile.TemporaryDirectory() as temp_dir:
        nam_dir = os.path.join(temp_dir, "nam")
        spk_dir = os.path.join(temp_dir, "spk")
        out_dir = os.path.join(temp_dir, "run_fft")
        os.makedirs(nam_dir)
        os.makedirs(spk_dir)

        for i in range(10):
            fid = f"pair_{i:02d}"
            np.save(os.path.join(nam_dir, f"{fid}.npy"), np.random.randn(20, 768).astype(np.float32))
            np.save(os.path.join(spk_dir, f"{fid}.npy"), np.random.randn(20, 768).astype(np.float32))

        cfg_dict = {
            "experiment_name": "cli_smoke_test_fft",
            "model": {
                "architecture": "stethospeech_fft",
                "input_dim": 768,
                "hidden_dim": 64,
                "output_dim": 768,
                "enc_layers": 1,
                "dec_layers": 1,
                "nheads": 2,
                "d_inner": 128,
                "kernel_sizes": [9, 1],
                "dropout": 0.0,
            },
            "training": {
                "seed": 42,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "clip_grad_norm": 1.0,
                "lr_patience": 2,
                "lr_factor": 0.5,
                "early_stop_patience": 5,
                "early_stop_delta": 0.001,
                "device": "cpu",
            },
            "data": {
                "nam_dir": nam_dir,
                "speech_dir": spk_dir,
                "output_dir": out_dir,
                "train_split": 0.90,
                "val_split": 0.10,
                "test_split": 0.0,
                "num_workers": 0,
            },
        }

        cfg_path = os.path.join(temp_dir, "fft_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(repo_root, "src")
        cmd = [sys.executable, script_path, "--config", cfg_path, "--device", "cpu", "--epochs", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)

        assert result.returncode == 0, f"FFT CLI failed:\n{result.stderr}\n{result.stdout}"
        assert os.path.exists(os.path.join(out_dir, "best_model.pth"))


def test_cli_train_failure_cases():
    """Verify training CLI exits with non-zero status on invalid input or missing paths."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "train.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    # 1. Non-existent config
    res1 = subprocess.run([sys.executable, script_path, "--config", "non_existent.yaml"], capture_output=True, env=env)
    assert res1.returncode != 0

    # 2. Config pointing to non-existent feature directories
    with tempfile.TemporaryDirectory() as temp_dir:
        bad_cfg = {
            "experiment_name": "bad_dirs",
            "model": {"architecture": "transformer_baseline"},
            "training": {},
            "data": {"nam_dir": "non_existent_nam_path", "speech_dir": "non_existent_speech_path"},
        }
        cfg_file = os.path.join(temp_dir, "bad.yaml")
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(bad_cfg, f)

        res2 = subprocess.run([sys.executable, script_path, "--config", cfg_file], capture_output=True, env=env)
        assert res2.returncode != 0


def test_e2e_cli_synthesis_units():
    """End-to-end CLI integration test: synthesize audio from discrete unit sequence."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "synthesize.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save synthetic vocoder weights
        vocoder = CodeHiFiGANGenerator()
        vocoder_ckpt = os.path.join(temp_dir, "vocoder_weights.pt")
        torch.save({"generator": vocoder.state_dict()}, vocoder_ckpt)

        # Save synthetic unit sequence (25 tokens = 0.5s at 50 Hz)
        units = np.random.randint(0, 1000, size=25).astype(np.int64)
        units_file = os.path.join(temp_dir, "units.npy")
        np.save(units_file, units)

        out_wav = os.path.join(temp_dir, "output.wav")

        cmd = [
            sys.executable,
            script_path,
            "--mode", "units",
            "--input-file", units_file,
            "--output-wav", out_wav,
            "--vocoder-checkpoint", vocoder_ckpt,
            "--device", "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)

        assert result.returncode == 0, f"Synthesis CLI failed:\n{result.stderr}\n{result.stdout}"
        assert os.path.exists(out_wav)

        data, sr = sf.read(out_wav)
        assert sr == 16000
        # 25 tokens * 320 = 8000 samples (0.50s)
        assert len(data) == 8000


def test_cli_synthesis_failure_cases():
    """Verify synthesis CLI exits with non-zero status when checkpoints or inputs are missing."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "synthesize.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    # Missing input file
    res1 = subprocess.run([sys.executable, script_path, "--input-file", "missing.npy"], capture_output=True, env=env)
    assert res1.returncode != 0

    # Missing vocoder checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_input = os.path.join(temp_dir, "units.npy")
        np.save(fake_input, np.array([1, 2, 3]))
        res2 = subprocess.run(
            [sys.executable, script_path, "--mode", "units", "--input-file", fake_input, "--vocoder-checkpoint", "missing.pt"],
            capture_output=True,
            env=env,
        )
        assert res2.returncode != 0


def test_harden_checkpoint_loading():
    """Verify hardened checkpoint loading rejects missing, corrupted, or incompatible files."""
    model = TransformerSeq2Seq(num_layers=1, hidden_dim=64, dim_ff=128, nheads=2)
    device = torch.device("cpu")

    # 1. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint("non_existent_weights.pth", model, device)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 2. Corrupted file raises ValueError
        corrupted_path = os.path.join(temp_dir, "corrupted.pth")
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_OR_PICKLE_FILE_RAW_GARBAGE")

        with pytest.raises(ValueError, match="Failed to load checkpoint|Corrupted or unreadable"):
            load_model_checkpoint(corrupted_path, model, device)

        # 3. Empty file raises ValueError
        empty_path = os.path.join(temp_dir, "empty.pth")
        with open(empty_path, "wb") as f:
            pass
        with pytest.raises(ValueError, match="empty"):
            load_model_checkpoint(empty_path, model, device)

        # 4. Incompatible architecture weights raise KeyError
        incompatible_path = os.path.join(temp_dir, "incompatible.pth")
        vocoder = CodeHiFiGANGenerator()
        torch.save({"state_dict": vocoder.state_dict()}, incompatible_path)
        with pytest.raises(KeyError, match="missing"):
            load_model_checkpoint(incompatible_path, model, device)

        # 5. Partially populated checkpoint (missing keys) raises KeyError
        full_state = model.state_dict()
        subset_keys = list(full_state.keys())[:len(full_state) // 2]
        partial_state = {k: full_state[k] for k in subset_keys}
        partial_path = os.path.join(temp_dir, "partial.pth")
        torch.save({"state_dict": partial_state}, partial_path)
        with pytest.raises(KeyError, match="missing"):
            load_model_checkpoint(partial_path, model, device)

        # 6. Checkpoint with unexpected keys raises KeyError
        unexpected_state = dict(full_state)
        unexpected_state["bogus_layer.param"] = torch.randn(10)
        unexpected_path = os.path.join(temp_dir, "unexpected.pth")
        torch.save({"state_dict": unexpected_state}, unexpected_path)
        with pytest.raises(KeyError, match="unexpected"):
            load_model_checkpoint(unexpected_path, model, device)

        # 7. Checkpoint with incompatible tensor shapes raises ValueError
        mismatched_state = dict(full_state)
        for k in mismatched_state:
            if mismatched_state[k].dim() >= 2:
                mismatched_state[k] = torch.randn(7, 7)
                break
        mismatch_path = os.path.join(temp_dir, "mismatch.pth")
        torch.save({"state_dict": mismatched_state}, mismatch_path)
        with pytest.raises(ValueError, match="Incompatible tensor shapes"):
            load_model_checkpoint(mismatch_path, model, device)

        # 8. Vocoder partial weights rejected with KeyError
        v_full = vocoder.state_dict()
        v_partial = {k: v_full[k] for k in list(v_full.keys())[:-2]}  # drop 2 keys
        v_partial_path = os.path.join(temp_dir, "vocoder_partial.pt")
        torch.save({"generator": v_partial}, v_partial_path)
        with pytest.raises(KeyError, match="missing"):
            load_model_checkpoint(v_partial_path, vocoder, device, strict=True)


def test_e2e_cli_checkpoint_evaluation():
    """End-to-end CLI integration test: evaluate a saved checkpoint with --eval-only."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "train.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    with tempfile.TemporaryDirectory() as temp_dir:
        nam_dir = os.path.join(temp_dir, "nam")
        spk_dir = os.path.join(temp_dir, "spk")
        out_dir = os.path.join(temp_dir, "run_eval")
        os.makedirs(nam_dir)
        os.makedirs(spk_dir)

        for i in range(8):
            fid = f"eval_{i:02d}"
            np.save(os.path.join(nam_dir, f"{fid}.npy"), np.random.randn(20, 768).astype(np.float32))
            np.save(os.path.join(spk_dir, f"{fid}.npy"), np.random.randn(20, 768).astype(np.float32))

        cfg_dict = {
            "experiment_name": "eval_smoke_test",
            "model": {
                "architecture": "transformer_baseline",
                "input_dim": 768,
                "hidden_dim": 64,
                "output_dim": 768,
                "num_layers": 1,
                "nheads": 2,
                "dim_ff": 128,
                "dropout": 0.0,
            },
            "training": {
                "seed": 42,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "clip_grad_norm": 1.0,
                "lr_patience": 2,
                "lr_factor": 0.5,
                "early_stop_patience": 5,
                "early_stop_delta": 0.001,
                "device": "cpu",
            },
            "data": {
                "nam_dir": nam_dir,
                "speech_dir": spk_dir,
                "output_dir": out_dir,
                "train_split": 0.50,
                "val_split": 0.25,
                "test_split": 0.25,
                "num_workers": 0,
            },
        }

        cfg_path = os.path.join(temp_dir, "eval_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f)

        # 1. Train 1 epoch to save checkpoint
        cmd_train = [sys.executable, script_path, "--config", cfg_path, "--device", "cpu", "--epochs", "1"]
        res_train = subprocess.run(cmd_train, capture_output=True, text=True, cwd=repo_root, env=env)
        assert res_train.returncode == 0

        ckpt_path = os.path.join(out_dir, "best_model.pth")
        assert os.path.exists(ckpt_path)

        # 2. Run evaluation CLI with --eval-only
        cmd_eval = [
            sys.executable, script_path, "--config", cfg_path,
            "--checkpoint", ckpt_path, "--eval-only", "--device", "cpu",
        ]
        res_eval = subprocess.run(cmd_eval, capture_output=True, text=True, cwd=repo_root, env=env)
        assert res_eval.returncode == 0, f"Eval CLI failed:\n{res_eval.stderr}\n{res_eval.stdout}"
        assert "Evaluated Loss" in res_eval.stdout


def test_e2e_cli_synthesis_features():
    """End-to-end CLI integration test: synthesize audio from continuous features via K-Means and vocoder."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "synthesize.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Train and save a small K-Means model (5 clusters)
        quantizer = KMeansUnitQuantizer(num_clusters=5, random_state=42)
        dummy_feats = np.random.randn(50, 768).astype(np.float32)
        quantizer.fit(dummy_feats)
        km_path = os.path.join(temp_dir, "kmeans.bin")
        quantizer.save(km_path)

        # 2. Save synthetic vocoder weights (using CodeHiFiGANGenerator with num_units=5 to match K-Means)
        vocoder = CodeHiFiGANGenerator(num_units=5)
        vocoder_ckpt = os.path.join(temp_dir, "vocoder_weights.pt")
        torch.save({"generator": vocoder.state_dict()}, vocoder_ckpt)

        # 3. Save synthetic continuous speech features (20 frames)
        features = np.random.randn(20, 768).astype(np.float32)
        feat_path = os.path.join(temp_dir, "feats.npy")
        np.save(feat_path, features)

        out_wav = os.path.join(temp_dir, "output_feat.wav")

        # 4. Run CLI in features mode testing --kmeans-checkpoint
        cmd = [
            sys.executable,
            script_path,
            "--mode", "features",
            "--input-file", feat_path,
            "--output-wav", out_wav,
            "--kmeans-checkpoint", km_path,
            "--vocoder-checkpoint", vocoder_ckpt,
            "--device", "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)
        assert result.returncode == 0, f"Feature synthesis CLI failed:\n{result.stderr}\n{result.stdout}"
        assert os.path.exists(out_wav)

        data, sr = sf.read(out_wav)
        assert sr == 16000
        # 20 frames * 320 = 6400 samples
        assert len(data) == 6400


def test_convert_checkpoint_utility():
    """Verify scripts/convert_checkpoint.py sanitizes legacy checkpoints to weights_only=True format."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(repo_root, "scripts", "convert_checkpoint.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "legacy.pth")
        output_path = os.path.join(temp_dir, "converted.pth")

        # Create simulated checkpoint containing tensor parameters
        dummy_state = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }
        torch.save({"state_dict": dummy_state}, input_path)

        cmd = [sys.executable, script_path, "--input", input_path, "--output", output_path]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)
        assert res.returncode == 0
        assert os.path.exists(output_path)

        # Verify it can be loaded with weights_only=True
        loaded = torch.load(output_path, map_location="cpu", weights_only=True)
        assert "state_dict" in loaded
        assert "weight" in loaded["state_dict"]


def test_synthesizer_helpers():
    """Test discrete unit and feature loading helper functions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Valid units .npy
        valid_units_path = os.path.join(temp_dir, "units.npy")
        np.save(valid_units_path, np.array([10, 20, 30], dtype=np.int64))
        loaded_units = load_discrete_units(valid_units_path)
        assert len(loaded_units) == 3

        # Invalid units out of range
        invalid_units_path = os.path.join(temp_dir, "invalid_units.npy")
        np.save(invalid_units_path, np.array([10, 2000, 30], dtype=np.int64))
        with pytest.raises(ValueError, match="must be in"):
            load_discrete_units(invalid_units_path, num_units=1000)

        # Valid feature .npy
        feat_path = os.path.join(temp_dir, "features.npy")
        np.save(feat_path, np.random.randn(20, 768).astype(np.float32))
        loaded_feats = load_features(feat_path)
        assert loaded_feats.shape == (20, 768)

        # Feature dimension mismatch
        bad_feat_path = os.path.join(temp_dir, "bad_feat.npy")
        np.save(bad_feat_path, np.random.randn(20, 512).astype(np.float32))
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            load_features(bad_feat_path, expected_dim=768)


def test_installed_console_entry_points():
    """Verify installed console commands execute from outside repo working directory."""
    with tempfile.TemporaryDirectory() as external_dir:
        # Build environment including Python scripts directories in PATH
        env = dict(os.environ)
        script_dirs = [
            os.path.join(sys.prefix, "Scripts"),
            os.path.join(sys.prefix, "bin"),
            os.path.join(site.getuserbase(), "Scripts"),
            os.path.join(site.getuserbase(), "bin"),
        ]
        valid_script_dirs = [d for d in script_dirs if os.path.exists(d)]
        if valid_script_dirs:
            env["PATH"] = os.pathsep.join(valid_script_dirs) + os.pathsep + env.get("PATH", "")

        commands = [
            ("stethospeech-train", "Train StethoSpeech translation model"),
            ("stethospeech-synth", "StethoSpeech Neural Vocoding"),
            ("stethospeech-mfa-prep", "Prepare MFA corpus"),
        ]

        for cmd_name, expected_text in commands:
            cmd_path = shutil.which(cmd_name, path=env["PATH"])
            if not cmd_path:
                for d in valid_script_dirs:
                    for ext in [".exe", ""]:
                        candidate = os.path.join(d, f"{cmd_name}{ext}")
                        if os.path.exists(candidate):
                            cmd_path = candidate
                            break
                    if cmd_path:
                        break

            if cmd_path:
                res = subprocess.run(
                    [cmd_path, "--help"],
                    capture_output=True,
                    text=True,
                    cwd=external_dir,
                    env=env,
                )
                assert res.returncode == 0, f"{cmd_name} failed:\n{res.stderr}\n{res.stdout}"
                assert expected_text in res.stdout, f"Expected '{expected_text}' in {cmd_name} output."
                assert "No module named 'scripts'" not in res.stderr

        # Verify module execution from outside repo directory
        res_m = subprocess.run(
            [sys.executable, "-m", "stethospeech.training.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=external_dir,
            env=env,
        )
        assert res_m.returncode == 0
        assert "Train StethoSpeech translation model" in res_m.stdout
