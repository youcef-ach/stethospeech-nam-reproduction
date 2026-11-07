# Reproducibility Guide & Verification Matrix

This guide provides exact instructions for reproducing the verified components of the StethoSpeech project and delineates the technical boundaries of what can be reproduced with current available assets.

---

## 1. Reproducibility Boundaries

To avoid ambiguity, the project explicitly separates reproducible components from those requiring original proprietary assets or lost cloud checkpoints:

| Component | Reproducibility Level | Requirements | Verified Output / Artifact |
|---|:---:|---|---|
| **Architectural Invariants & Parameter Counts** | **100% Fully Reproducible** | CPU only, Python 3.10–3.13, PyTorch | 24 passing automated tests: Baseline (19,702,016 params), FFT (29,263,360 params), Vocoder (13,375,809 params). |
| **Masked Loss & Temporal Alignment Math** | **100% Fully Reproducible** | CPU only | Linear temporal interpolation and zero-padding masking invariant tests pass. |
| **Early Stopping Mathematical Logic** | **100% Fully Reproducible** | CPU only | Exact reproduction of Epoch 35 best checkpoint selection and Epoch 45 halt. |
| **Vocoder Upsampling Rate ($320\times$)** | **100% Fully Reproducible** | CPU only | 50 frames @ 50 Hz correctly produce 16,000 waveform samples (1.00s @ 16kHz). |
| **MFA Corpus Organization Pipeline** | **100% Fully Reproducible** | Local or Kaggle raw audio directories | Automated pairing and structuring of multi-speaker corpus. |
| **Full Baseline Model Training (Phase 4A)** | **Reproducible with Raw Features** | GPU (NVIDIA P100/T4), ~2.5 hours, 11,323 matched pairs | Retrains baseline model to converge near ~0.23171 MAE test loss. |
| **Historical Baseline Weights (`best_stetho_model.pth`)** | **Non-Reproducible from Disk** | Original Kaggle ephemeral container | Checkpoint file was saved in ephemeral `/kaggle/working/` and expired. Must be retrained using `scripts/train.py`. |
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

The repository provides a fast CPU test suite that does not require GPU resources or proprietary speech datasets. It validates mathematical tensor operations, shapes, and exact parameter counts in ~7 seconds:

```bash
python -m pytest -v
```

Expected output:
```text
============================= test session starts =============================
collected 24 items

tests/test_cli.py::test_yaml_config_loading PASSED                       [  4%]
tests/test_cli.py::test_early_stopping_mathematical_resolution PASSED    [  8%]
tests/test_cli.py::test_trainer_cpu_smoke_step PASSED                    [ 12%]
tests/test_data.py::test_find_matched_feature_pairs PASSED               [ 16%]
tests/test_data.py::test_missing_feature_directory_raises PASSED         [ 20%]
tests/test_data.py::test_speaker_dependent_splits PASSED                 [ 25%]
tests/test_data.py::test_stethospeech_pair_dataset PASSED                [ 29%]
tests/test_data.py::test_stethospeech_pair_dataset_missing_file_raises PASSED [ 33%]
tests/test_data.py::test_collate_stethospeech_batch PASSED               [ 37%]
tests/test_losses.py::test_masked_l1_loss_temporal_interpolation PASSED  [ 41%]
tests/test_losses.py::test_masked_l1_loss_ignores_padded_frames PASSED   [ 45%]
tests/test_losses.py::test_exact_masked_l1_loss_raises_on_shape_mismatch PASSED [ 50%]
tests/test_losses.py::test_auxiliary_ctc_loss PASSED                     [ 54%]
tests/test_models.py::test_phase_4a_parameter_count PASSED               [ 58%]
tests/test_models.py::test_phase_4a_forward_pass_dimensions PASSED       [ 62%]
tests/test_models.py::test_positional_encoding_shape_invariance PASSED   [ 66%]
tests/test_models.py::test_fft_block_temporal_invariance PASSED          [ 70%]
tests/test_models.py::test_phase_4b_executed_model_parameter_count PASSED [ 75%]
tests/test_models.py::test_phase_4b_forward_pass_dimensions PASSED       [ 79%]
tests/test_models.py::test_phase_4b_with_ctc_head PASSED                 [ 83%]
tests/test_vocoder.py::test_vocoder_parameter_count PASSED               [ 87%]
tests/test_vocoder.py::test_vocoder_temporal_upsampling_320x PASSED      [ 91%]
tests/test_vocoder.py::test_resblock1_receptive_fields PASSED            [ 95%]
tests/test_vocoder.py::test_kmeans_unit_quantizer_fit_predict PASSED     [100%]

======================= 24 passed, 2 warnings in 6.96s ========================
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
