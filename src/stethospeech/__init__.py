"""
StethoSpeech: Non-Audible Murmur (NAM) to Audible Speech Translation Research Package.

This package provides maintainable, typed, and reproducible implementations of:
- Phase 4A Baseline: 19.7M-parameter Transformer sequence regression model
- Phase 4B Experimental: 29.26M-parameter FastSpeech2-derived FFT architecture
- Phase 5 Vocoder: Native PyTorch Unit HiFi-GAN generator with 320x temporal upsampling
- Data processing and alignment utilities for the StethoSpeech stethoscopic acoustic corpus
"""

__version__ = "0.2.0"
__author__ = "Achouri Youcef Islem"
__license__ = "Research Use / Attribution Required"

from stethospeech.models.transformer_baseline import TransformerSeq2Seq
from stethospeech.models.stethospeech_fft import StethoSpeechModel, StethoSpeechWithCTC
from stethospeech.models.vocoder import CodeHiFiGANGenerator

__all__ = [
    "TransformerSeq2Seq",
    "StethoSpeechModel",
    "StethoSpeechWithCTC",
    "CodeHiFiGANGenerator",
]
