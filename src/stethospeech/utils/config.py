"""Configuration data classes and YAML loading utilities."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
import yaml


@dataclass
class ModelConfig:
    architecture: str = "transformer_baseline"  # "transformer_baseline" or "stethospeech_fft"
    input_dim: int = 768
    hidden_dim: int = 512
    output_dim: int = 768
    num_layers: int = 6
    nheads: int = 8
    dim_ff: int = 2048
    dropout: float = 0.1
    # FFT specific settings
    enc_layers: int = 6
    dec_layers: int = 4
    d_inner: int = 1024
    kernel_sizes: List[int] = field(default_factory=lambda: [9, 1])
    include_ctc_head: bool = False
    ctc_vocab: int = 32


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 45
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    lr_patience: int = 3
    lr_factor: float = 0.5
    early_stop_patience: int = 10
    early_stop_delta: float = 0.001
    device: str = "cuda"


@dataclass
class DataConfig:
    nam_dir: str = "data/features/nam"
    speech_dir: str = "data/features/speech"
    output_dir: str = "experiments/run"
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    num_workers: int = 2


@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline_reproduction"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        model_data = data.get("model", {})
        training_data = data.get("training", {})
        data_data = data.get("data", {})

        return cls(
            experiment_name=data.get("experiment_name", "unnamed_experiment"),
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            data=DataConfig(**data_data),
        )


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Populated ExperimentConfig instance.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return ExperimentConfig.from_dict(raw_dict)
