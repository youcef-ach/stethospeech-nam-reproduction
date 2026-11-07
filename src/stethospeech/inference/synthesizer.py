"""
Inference and End-to-End Audio Synthesis Pipeline.

Connects:
1. Neural Translation Model (Transformer baseline or StethoSpeech FFT)
2. K-Means Unit Quantizer (Continuous 768-dim features -> Discrete cluster IDs [0, 999])
3. Unit HiFi-GAN Vocoder (Discrete cluster IDs -> 16kHz audio waveform)

Scientific Boundary and Qualification:
The Unit HiFi-GAN vocoder generator has been experimentally verified on
ground-truth reference and synthetic speech tokens. However, speech audio
synthesized from model-predicted NAM representations has NOT undergone human
listening tests (MOS) or ASR decoding (WER/CER). Therefore, end-to-end
intelligibility of model-predicted audio is unverified.
"""

import argparse
import os
import sys
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

from stethospeech.models.vocoder import CodeHiFiGANGenerator
from stethospeech.features.quantization import KMeansUnitQuantizer


def remap_checkpoint_keys(state_dict: dict, target_model: nn.Module) -> dict:
    """
    Map official checkpoint key naming conventions to target model structure.
    Handles 'generator.' / 'model.' prefix stripping and official weight-norm reparameterization.
    """
    target_keys = set(target_model.state_dict().keys())
    if set(state_dict.keys()) == target_keys:
        return state_dict

    # Check for weight normalization keys (.weight_g, .weight_v)
    has_wn = any(k.endswith(".weight_g") for k in state_dict)
    if has_wn:
        converted = {}
        for k in state_dict:
            if k.endswith(".weight_g"):
                base = k[:-9]
                vg = state_dict[k]
                vv_key = base + ".weight_v"
                if vv_key in state_dict:
                    vv = state_dict[vv_key]
                    if "ups" in base:
                        # ConvTranspose1d: norm over dimensions (0, 2)
                        norm = torch.norm(vv, p=2, dim=[0, 2], keepdim=True)
                    else:
                        # Conv1d: norm over dimensions (1, 2)
                        norm = torch.norm(vv, p=2, dim=[1, 2], keepdim=True)
                    converted[base + ".weight"] = vg * (vv / (norm + 1e-12))
            elif k.endswith(".weight_v"):
                continue
            else:
                converted[k] = state_dict[k]
        state_dict = converted

    if set(state_dict.keys()) == target_keys:
        return state_dict

    # Check prefix stripping: generator.
    if any(k.startswith("generator.") for k in state_dict):
        stripped = {
            (k[10:] if k.startswith("generator.") else k): v
            for k, v in state_dict.items()
        }
        if set(stripped.keys()) == target_keys:
            return stripped
        state_dict = stripped

    # Check prefix stripping: model.
    if any(k.startswith("model.") for k in state_dict):
        stripped_model = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        if set(stripped_model.keys()) == target_keys:
            return stripped_model
        state_dict = stripped_model

    return state_dict


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """
    Safely load model weights from disk with structural validation.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Target PyTorch nn.Module.
        device: PyTorch device.
        strict: Whether to enforce strict key matching.

    Returns:
        Loaded model on the specified device in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    if os.path.getsize(checkpoint_path) == 0:
        raise ValueError(f"Checkpoint file is empty (0 bytes): {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load checkpoint '{checkpoint_path}' securely with weights_only=True: {exc}. "
            "Automatic fallback to unsafe weights_only=False is disabled. "
            "If this is a trusted legacy checkpoint containing pickled objects, convert it using "
            "scripts/convert_checkpoint.py in an isolated environment."
        ) from exc

    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(ckpt).__name__}")

    # Extract state dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "generator" in ckpt:
        state_dict = ckpt["generator"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Extracted state_dict must be a dictionary, got {type(state_dict).__name__}")

    state_dict = remap_checkpoint_keys(state_dict, model)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing {len(missing_keys)} required parameter keys for {type(model).__name__}: "
            f"{sorted(list(missing_keys))[:5]}... Partially populated checkpoints are rejected."
        )

    if unexpected_keys and strict:
        raise KeyError(
            f"Checkpoint contains {len(unexpected_keys)} unexpected keys for {type(model).__name__}: "
            f"{sorted(list(unexpected_keys))[:5]}... Checkpoint structure does not match target model."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        raise ValueError(
            f"Incompatible tensor shapes or weights in checkpoint for {type(model).__name__}: {err}"
        ) from err

    model.to(device).eval()
    return model


def load_discrete_units(units_path: str, num_units: int = 1000) -> np.ndarray:
    """
    Load and validate discrete unit sequence from a .npy or .txt file.

    Args:
        units_path: Path to unit tokens file.
        num_units: Maximum unit vocabulary size.

    Returns:
        1D or 2D integer numpy array of unit IDs.
    """
    if not os.path.exists(units_path):
        raise FileNotFoundError(f"Discrete units file not found: {units_path}")

    if units_path.endswith(".npy"):
        units = np.load(units_path)
    else:
        # Load from text file (whitespace or comma separated)
        try:
            units = np.loadtxt(units_path, dtype=np.int64)
        except Exception:
            with open(units_path, "r", encoding="utf-8") as f:
                content = f.read().replace(",", " ")
            units = np.array([int(x) for x in content.split()], dtype=np.int64)

    units = np.asarray(units)
    if units.size == 0:
        raise ValueError(f"Units file contains no tokens: {units_path}")

    # Validate integer values
    if not np.issubdtype(units.dtype, np.integer):
        raise ValueError(f"Discrete units must be integers, got dtype {units.dtype}")

    if np.any(units < 0) or np.any(units >= num_units):
        invalid = units[(units < 0) | (units >= num_units)]
        raise ValueError(
            f"Discrete units must be in [0, {num_units - 1}]. Found invalid values: {invalid[:5]}"
        )

    return units


def load_features(features_path: str, expected_dim: int = 768) -> np.ndarray:
    """
    Load continuous acoustic feature array (.npy).

    Args:
        features_path: Path to .npy feature array.
        expected_dim: Expected feature embedding dimension (default: 768).

    Returns:
        2D numpy array of shape (Time, expected_dim).
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    features = np.load(features_path)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array (Time, Dim), got shape {features.shape}")

    if features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}"
        )

    return features.astype(np.float32)


class StethoSpeechSynthesizer:
    """
    End-to-end inference synthesizer mapping input NAM features to 16kHz audio waveforms.

    Args:
        model: Trained feature translation model.
        quantizer: Fitted KMeansUnitQuantizer instance (1,000 units).
        vocoder: Instantiated CodeHiFiGANGenerator.
        device: PyTorch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        quantizer: Optional[KMeansUnitQuantizer] = None,
        vocoder: Optional[CodeHiFiGANGenerator] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval() if model is not None else None
        self.quantizer = quantizer
        self.vocoder = vocoder.to(self.device).eval() if vocoder is not None else None

    @classmethod
    def from_checkpoints(
        cls,
        model: Optional[nn.Module] = None,
        model_checkpoint_path: Optional[str] = None,
        kmeans_checkpoint_path: Optional[str] = None,
        vocoder_weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "StethoSpeechSynthesizer":
        """
        Instantiate pipeline with strict checkpoint validation.

        Fails immediately if mandatory checkpoints are missing or incompatible.
        Never falls back silently to uninitialized random weights.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load translation model weights if configured
        if model is not None:
            if model_checkpoint_path is None:
                raise ValueError("model_checkpoint_path must be provided when model is specified.")
            model = load_model_checkpoint(model_checkpoint_path, model, dev, strict=True)

        # Load K-Means quantizer if configured
        quantizer = None
        if kmeans_checkpoint_path is not None:
            quantizer = KMeansUnitQuantizer.load(kmeans_checkpoint_path)

        # Load Vocoder weights if configured
        vocoder = None
        if vocoder_weights_path is not None:
            num_units = 1000
            try:
                raw_sd = torch.load(vocoder_weights_path, map_location="cpu", weights_only=True)
                if isinstance(raw_sd, dict):
                    inner = raw_sd.get("generator", raw_sd.get("state_dict", raw_sd))
                    if isinstance(inner, dict) and "dict.weight" in inner:
                        num_units = int(inner["dict.weight"].shape[0])
            except Exception:
                pass
            raw_vocoder = CodeHiFiGANGenerator(num_units=num_units)
            vocoder = load_model_checkpoint(vocoder_weights_path, raw_vocoder, dev, strict=True)

        return cls(model=model, quantizer=quantizer, vocoder=vocoder, device=dev)

    def synthesize_from_units(
        self,
        discrete_units: Union[np.ndarray, torch.Tensor],
        output_wav_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate audio from a sequence of discrete acoustic units.

        Args:
            discrete_units: Array or tensor of unit IDs of shape (Time_frames,) or (1, Time_frames).
            output_wav_path: Optional file path to save 16kHz WAV file.

        Returns:
            1D numpy array containing audio samples normalized in [-1.0, 1.0].
        """
        if self.vocoder is None:
            raise RuntimeError("Vocoder model is not initialized or loaded.")

        if isinstance(discrete_units, np.ndarray):
            unit_tensor = torch.from_numpy(discrete_units).long()
        else:
            unit_tensor = discrete_units.long()

        if unit_tensor.ndim == 1:
            unit_tensor = unit_tensor.unsqueeze(0)

        unit_tensor = unit_tensor.to(self.device)

        with torch.no_grad():
            waveform = self.vocoder(unit_tensor)

        audio_samples = waveform.squeeze().cpu().numpy()

        if output_wav_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
            sf.write(output_wav_path, audio_samples, 16000)

        return audio_samples

    def synthesize_from_features(
        self,
        features: Union[np.ndarray, torch.Tensor],
        output_wav_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize continuous features and synthesize audio waveform.

        Args:
            features: Continuous feature array/tensor of shape (Time_frames, 768).
            output_wav_path: Optional WAV destination path.

        Returns:
            Tuple of (discrete_units, audio_samples).
        """
        if self.quantizer is None:
            raise RuntimeError("K-Means quantizer is not initialized or loaded.")

        units = self.quantizer.predict(features)
        audio = self.synthesize_from_units(units, output_wav_path=output_wav_path)
        return units, audio

    def synthesize_end_to_end(
        self,
        nam_features: Union[np.ndarray, torch.Tensor],
        output_wav_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        End-to-end inference from input NAM features to output speech waveform.

        Scientific Notice:
        Audio generated from model-predicted NAM representations has NOT undergone
        human listening tests (MOS) or ASR decoding (WER/CER). Perceptual intelligibility
        is an active research problem.
        """
        if self.model is None:
            raise RuntimeError("Translation model is not initialized or loaded.")

        if isinstance(nam_features, np.ndarray):
            nam_tensor = torch.from_numpy(nam_features).float()
        else:
            nam_tensor = nam_features.float()

        if nam_tensor.ndim == 2:
            nam_tensor = nam_tensor.unsqueeze(0)

        nam_tensor = nam_tensor.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "proj_in"):
                pred_speech = self.model(nam_tensor)
            else:
                out = self.model(nam_tensor)
                pred_speech = out[0] if isinstance(out, tuple) else out

        pred_features_np = pred_speech.squeeze(0).cpu().numpy()
        units, audio = self.synthesize_from_features(pred_features_np, output_wav_path=output_wav_path)
        return pred_features_np, units, audio


def run_synthesizer_cli():
    """Execute synthesis CLI workflow."""
    parser = argparse.ArgumentParser(description="StethoSpeech Neural Vocoding and Audio Synthesis CLI.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["units", "features", "end-to-end"],
        default="units",
        help=(
            "Synthesis mode: 'units' (from discrete tokens), "
            "'features' (from continuous 768d), 'end-to-end' (NAM -> speech)."
        ),
    )
    parser.add_argument("--input-file", type=str, required=True, help="Path to input .npy or .txt file")
    parser.add_argument("--output-wav", type=str, default="output/synthesis.wav", help="Destination path for 16kHz WAV file")
    parser.add_argument("--vocoder-checkpoint", type=str, default=None, help="Path to Unit HiFi-GAN vocoder weights (*.pt)")
    parser.add_argument(
        "--kmeans-checkpoint",
        "--kmeans-model",
        dest="kmeans_checkpoint",
        type=str,
        default=None,
        help="Path to K-Means model (*.bin or *.joblib)",
    )
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to translation model weights (*.pth)")
    parser.add_argument("--device", type=str, default=None, help="Execution device: cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file does not exist: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # 1. Mode: units (Discrete tokens directly to 16kHz audio)
    if args.mode == "units":
        if args.vocoder_checkpoint is None or not os.path.exists(args.vocoder_checkpoint):
            print(
                f"[ERROR] Vocoder checkpoint is required for unit synthesis. "
                f"Missing or invalid: {args.vocoder_checkpoint}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[Synthesizer] Loading discrete units from {args.input_file}...")
        units = load_discrete_units(args.input_file)
        print(f"  Loaded {units.size} tokens (at 50 Hz = {units.size / 50:.2f}s of audio).")

        synthesizer = StethoSpeechSynthesizer.from_checkpoints(
            vocoder_weights_path=args.vocoder_checkpoint,
            device=device,
        )

        print(f"[Synthesizer] Generating 16kHz audio waveform using Unit HiFi-GAN on {device}...")
        audio = synthesizer.synthesize_from_units(units, output_wav_path=args.output_wav)
        duration_s = len(audio) / 16000.0
        print(f"  ✅ Saved {len(audio):,} samples ({duration_s:.2f}s) to: {args.output_wav}")

    # 2. Mode: features (Continuous 768d HuBERT features -> K-Means -> Vocoder)
    elif args.mode == "features":
        if args.vocoder_checkpoint is None or not os.path.exists(args.vocoder_checkpoint):
            print(f"[ERROR] Missing vocoder checkpoint: {args.vocoder_checkpoint}", file=sys.stderr)
            sys.exit(1)
        if args.kmeans_checkpoint is None or not os.path.exists(args.kmeans_checkpoint):
            print(f"[ERROR] Missing K-Means checkpoint: {args.kmeans_checkpoint}", file=sys.stderr)
            sys.exit(1)

        print(f"[Synthesizer] Loading continuous features from {args.input_file}...")
        features = load_features(args.input_file)
        print(f"  Loaded feature array of shape {features.shape} ({features.shape[0] / 50:.2f}s).")

        synthesizer = StethoSpeechSynthesizer.from_checkpoints(
            kmeans_checkpoint_path=args.kmeans_checkpoint,
            vocoder_weights_path=args.vocoder_checkpoint,
            device=device,
        )

        print(f"[Synthesizer] Quantizing features and vocoding on {device}...")
        units, audio = synthesizer.synthesize_from_features(features, output_wav_path=args.output_wav)
        duration_s = len(audio) / 16000.0
        print(f"  ✅ Saved {len(audio):,} samples ({duration_s:.2f}s) to: {args.output_wav}")

    # 3. Mode: end-to-end (NAM features -> Translation Model -> K-Means -> Vocoder)
    elif args.mode == "end-to-end":
        print(
            "⚠️  [DISCLOSURE] Speech synthesis from model-predicted NAM representations has not "
            "been validated by human listening tests (MOS) or ASR decoding (WER/CER).",
            file=sys.stderr,
        )
        if args.model_checkpoint is None or not os.path.exists(args.model_checkpoint):
            print(f"[ERROR] Missing translation model checkpoint: {args.model_checkpoint}", file=sys.stderr)
            sys.exit(1)
        if args.vocoder_checkpoint is None or not os.path.exists(args.vocoder_checkpoint):
            print(f"[ERROR] Missing vocoder checkpoint: {args.vocoder_checkpoint}", file=sys.stderr)
            sys.exit(1)
        if args.kmeans_checkpoint is None or not os.path.exists(args.kmeans_checkpoint):
            print(f"[ERROR] Missing K-Means checkpoint: {args.kmeans_checkpoint}", file=sys.stderr)
            sys.exit(1)

        from stethospeech.models.transformer_baseline import TransformerSeq2Seq

        model = TransformerSeq2Seq()
        synthesizer = StethoSpeechSynthesizer.from_checkpoints(
            model=model,
            model_checkpoint_path=args.model_checkpoint,
            kmeans_checkpoint_path=args.kmeans_checkpoint,
            vocoder_weights_path=args.vocoder_checkpoint,
            device=device,
        )

        features = load_features(args.input_file)
        _, _, audio = synthesizer.synthesize_end_to_end(features, output_wav_path=args.output_wav)
        print(f"  ✅ Saved end-to-end prediction ({len(audio)/16000:.2f}s) to: {args.output_wav}")


def main():
    try:
        run_synthesizer_cli()
    except Exception as exc:
        print(f"[ERROR] Synthesis failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
