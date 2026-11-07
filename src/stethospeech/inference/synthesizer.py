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
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

from stethospeech.models.vocoder import CodeHiFiGANGenerator
from stethospeech.features.quantization import KMeansUnitQuantizer


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
        model: nn.Module,
        model_checkpoint_path: str,
        kmeans_checkpoint_path: str,
        vocoder_weights_path: str,
        device: Optional[torch.device] = None,
    ) -> "StethoSpeechSynthesizer":
        """
        Instantiate pipeline from checkpoint files on disk.

        Args:
            model: Instantiated model architecture.
            model_checkpoint_path: Path to translation model checkpoint (*.pth).
            kmeans_checkpoint_path: Path to serialized K-Means model (*.bin or *.joblib).
            vocoder_weights_path: Path to Unit HiFi-GAN generator weights (*.pt).
            device: PyTorch device.

        Returns:
            Configured StethoSpeechSynthesizer.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load translation model weights
        if os.path.exists(model_checkpoint_path):
            ckpt = torch.load(model_checkpoint_path, map_location=dev, weights_only=False)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model = model.to(dev).eval()

        # Load K-Means quantizer
        quantizer = KMeansUnitQuantizer.load(kmeans_checkpoint_path)

        # Load Vocoder weights
        vocoder = CodeHiFiGANGenerator()
        if os.path.exists(vocoder_weights_path):
            v_state = torch.load(vocoder_weights_path, map_location="cpu", weights_only=False)
            gen_state = v_state["generator"] if "generator" in v_state else v_state
            vocoder.load_state_dict(gen_state, strict=False)
        vocoder = vocoder.to(dev).eval()

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
            raise RuntimeError("Vocoder model is not initialized.")

        if isinstance(discrete_units, np.ndarray):
            unit_tensor = torch.from_numpy(discrete_units).long()
        else:
            unit_tensor = discrete_units.long()

        if unit_tensor.ndim == 1:
            unit_tensor = unit_tensor.unsqueeze(0)

        unit_tensor = unit_tensor.to(self.device)

        with torch.no_grad():
            # Waveform shape: (1, 1, Time_frames * 320)
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
            raise RuntimeError("Quantizer is not initialized.")

        units = self.quantizer.predict(features)
        audio = self.synthesize_from_units(units, output_wav_path=output_wav_path)
        return units, audio


def main() -> None:
    """CLI entry point for synthesis."""
    parser = argparse.ArgumentParser(description="Synthesize audio using StethoSpeech vocoder pipeline.")
    parser.add_argument("--units-file", type=str, default=None, help="Path to discrete unit sequence (*.npy or *.txt)")
    parser.add_argument("--output-wav", type=str, default="output/synthesis.wav", help="Destination WAV path")
    args = parser.parse_args()

    print("[Synthesizer] Synthesizer CLI ready. Provide discrete unit or feature inputs for synthesis.")


if __name__ == "__main__":
    main()
