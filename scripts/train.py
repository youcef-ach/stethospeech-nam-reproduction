"""
Command-line interface to execute StethoSpeech model training and evaluation.

Loads configuration from YAML, matches paired NAM and Speech acoustic features,
partitions deterministic speaker-dependent splits, executes training with early stopping,
and evaluates the held-out test set with checkpoint validation.
"""

from stethospeech.training.cli import (
    build_model,
    main,
    parse_args,
    run_training_cli,
)

__all__ = ["parse_args", "build_model", "run_training_cli", "main"]

if __name__ == "__main__":
    main()
