# Reproducibility Guide & Verification Matrix

This guide provides exact instructions for reproducing the verified components of the StethoSpeech project and delineates the technical boundaries of what can be reproduced with current available assets.

---

## 1. Reproducibility Boundaries

To avoid ambiguity, the project explicitly separates reproducible components from those requiring original proprietary assets or lost cloud checkpoints:

| Component | Reproducibility Level | Requirements | Verified Output / Artifact |
|---|:---:|---|---|
| **Architectural Invariants & Parameter Counts** | **100% Fully Reproducible** | CPU only, Python 3.10–3.13, PyTorch | 41 passing automated tests: Baseline (19,702,016 params), FFT (29,263,360 params), Vocoder (13,375,809 params), CLI integration, installed console entry points, strict checkpoint validation, and conversion utility. |
| **Masked Loss & Temporal Alignment Math** | **100% Fully Reproducible** | CPU only | Linear temporal interpolation and zero-padding masking invariant tests pass. |
| **Early Stopping Mathematical Logic** | **100% Fully Reproducible** | CPU only | Exact reproduction of Epoch 35 best checkpoint selection and Epoch 45 halt. |
| **Vocoder Upsampling Rate ($320\times$)** | **100% Fully Reproducible** | CPU only | 50 frames @ 50 Hz correctly produce 16,000 waveform samples (1.00s @ 16kHz). |
| **Data Partitioning & Boundary Handling** | **100% Fully Reproducible** | CPU only | Flexible split ratios (including test_ratio=0.0), historical 80/10/10 split reproduction. |
| **CLI Pipelines & Installed Console Scripts** | **100% Fully Reproducible** | CPU only | `stethospeech-train`, `stethospeech-synth`, and `stethospeech-mfa-prep` console entry points, and hardened rejection of missing, unexpected, corrupt, and partially initialized checkpoints. |
| **MFA Corpus Organization Pipeline** | **100% Fully Reproducible** | Local or Kaggle raw audio directories | Automated pairing and structuring of multi-speaker corpus. |
| **Full Baseline Model Training (Phase 4A)** | **Reproducible with Raw Features** | GPU (NVIDIA P100/T4), ~2.5 hours, 11,323 matched pairs | Retrains baseline model across training epochs on matched pairs. Newly trained runs are subject to hardware, driver, and initialization variance and are not guaranteed to reproduce the exact historical 0.23171 MAE. |
| **Historical Baseline Weights (`best_stetho_model.pth`)** | **Non-Reproducible from Disk** | Original Kaggle ephemeral container | Checkpoint file was saved in ephemeral `/kaggle/working/` and expired. Must be retrained using `scripts/train.py`. |
| **Official Pretrained Unit HiFi-GAN Vocoder** | **Unverified in Release Bundle** | Authorized external download | Pure-PyTorch architecture and weight-norm reparameterization verified with synthetic weights; real pretrained audio quality unverified until authorized weights are staged. |
| **Phase 4B Full Convergence** | **Not Claimed / Incomplete** | Kaggle GPU | Historical run only trained ~2 preliminary epochs before session timeout. |
| **NAM-Predicted Audio Intelligibility** | **Unverified** | Human listening / ASR | No MOS listening scores or ASR transcriptions exist for predicted NAM speech. |

---

## 2. Environment Setup

### 2.1 Local Installation

```bash
# 1. Clone repository
git clone https://github.com/youcef-ach/stethospeech-nam-reproduction.git
cd stethospeech-nam-reproduction

# 2. Create virtual environment
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# 3. Install dependencies in editable mode
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

---

## 3. Running Automated Integrity Tests (CPU Smoke Tests)

The repository provides an automated CPU test suite that does not require GPU resources or proprietary speech datasets. It validates mathematical tensor operations, shapes, exact parameter counts, data split invariants, CLI subprocess workflows, and hardened checkpoint loading:

```bash
python -m pytest -v
```

Expected output:
```text
============================= test session starts =============================
collected 40 items

tests/test_cli.py::test_yaml_config_loading PASSED                       [  2%]
tests/test_cli.py::test_early_stopping_mathematical_resolution PASSED    [  5%]
tests/test_cli.py::test_trainer_cpu_smoke_step PASSED                    [  7%]
tests/test_cli.py::test_e2e_cli_training_baseline PASSED                 [ 10%]
tests/test_cli.py::test_e2e_cli_training_fft_phase_4b PASSED             [ 12%]
tests/test_cli.py::test_cli_train_failure_cases PASSED                   [ 15%]
tests/test_cli.py::test_e2e_cli_checkpoint_evaluation PASSED            [ 17%]
tests/test_cli.py::test_e2e_cli_synthesis_units PASSED                   [ 20%]
tests/test_cli.py::test_e2e_cli_synthesis_features PASSED                [ 22%]
tests/test_cli.py::test_cli_synthesis_failure_cases PASSED               [ 25%]
tests/test_cli.py::test_harden_checkpoint_loading PASSED                 [ 27%]
tests/test_cli.py::test_synthesizer_helpers PASSED                       [ 30%]
tests/test_cli.py::test_convert_checkpoint_utility PASSED                 [ 32%]
tests/test_data.py::test_find_matched_feature_pairs PASSED               [ 35%]
tests/test_data.py::test_missing_feature_directory_raises PASSED         [ 37%]
tests/test_data.py::test_speaker_dependent_splits_standard PASSED        [ 40%]
tests/test_data.py::test_speaker_dependent_splits_phase_4b_no_test PASSED [ 42%]
tests/test_data.py::test_speaker_dependent_splits_historical_fidelity PASSED [ 45%]
tests/test_data.py::test_speaker_dependent_splits_invalid_ratios_raise PASSED [ 47%]
tests/test_data.py::test_speaker_dependent_splits_dataset_too_small_raises PASSED [ 50%]
tests/test_data.py::test_stethospeech_pair_dataset PASSED                [ 52%]
tests/test_data.py::test_stethospeech_pair_dataset_missing_file_raises PASSED [ 55%]
tests/test_data.py::test_collate_stethospeech_batch PASSED               [ 57%]
tests/test_losses.py::test_masked_l1_loss_temporal_interpolation PASSED  [ 60%]
tests/test_losses.py::test_masked_l1_loss_ignores_padded_frames PASSED   [ 62%]
tests/test_losses.py::test_exact_masked_l1_loss_raises_on_shape_mismatch PASSED [ 65%]
tests/test_losses.py::test_auxiliary_ctc_loss PASSED                     [ 67%]
tests/test_models.py::test_phase_4a_parameter_count PASSED               [ 70%]
tests/test_models.py::test_phase_4a_forward_pass_dimensions PASSED       [ 72%]
tests/test_models.py::test_positional_encoding_shape_invariance PASSED   [ 75%]
tests/test_models.py::test_fft_block_temporal_invariance PASSED          [ 77%]
tests/test_models.py::test_phase_4b_executed_model_parameter_count PASSED [ 80%]
tests/test_models.py::test_phase_4b_forward_pass_dimensions PASSED       [ 82%]
tests/test_models.py::test_phase_4b_with_ctc_head PASSED                 [ 85%]
tests/test_models.py::test_fft_training_step_with_model_trainer PASSED   [ 87%]
tests/test_models.py::test_ctc_model_training_step_with_model_trainer PASSED [ 90%]
tests/test_vocoder.py::test_vocoder_parameter_count PASSED               [ 92%]
tests/test_vocoder.py::test_vocoder_temporal_upsampling_320x PASSED      [ 95%]
tests/test_vocoder.py::test_resblock1_receptive_fields PASSED            [ 97%]
tests/test_vocoder.py::test_kmeans_unit_quantizer_fit_predict PASSED     [100%]

============================== 40 passed in ~55s ===============================
```

---

## 4. Retraining the Baseline Model (GPU Required)

When feature directories are populated with matched `.npy` arrays (11,323 pairs):

```bash
# Train Phase 4A Transformer Baseline
python scripts/train.py --config configs/baseline_transformer.yaml --device cuda

# Train Phase 4B StethoSpeech FFT Architecture
python scripts/train.py --config configs/experimental_fft.yaml --device cuda
```

---

## 5. Offline Conversion for Legacy Pickled Checkpoints

In strict compliance with PyTorch security standards, all synthesis and evaluation routines enforce `weights_only=True` by default. If an older checkpoint contains pickled metadata and fails to load under `weights_only=True`, convert it in an isolated offline environment using:

```bash
python scripts/convert_checkpoint.py \
  --input path/to/legacy_checkpoint.pth \
  --output path/to/safe_checkpoint.pth
```

> [!CAUTION]
> **Isolation Requirement:** Legacy checkpoint conversion requires deserialization with `weights_only=False`. This must be performed strictly inside a **genuinely isolated execution environment** (e.g. an ephemeral air-gapped container or throwaway sandbox) and only on strictly trusted input files. The `convert_checkpoint.py` utility does **not** provide sandboxing or isolation itself.

---

## 6. Deserialization Security Guidelines

1. **PyTorch Checkpoints:** Model weights are loaded strictly with `torch.load(..., weights_only=True)` and strict layer key verification.
2. **K-Means Quantizers:** Deserialization of `.joblib` / `.bin` clustering models relies on `joblib.load()`, which uses Python's pickle protocol. Because joblib deserialization is **not inherently secure**, clustering models must strictly originate from trusted, verified sources.
