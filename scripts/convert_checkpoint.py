"""
Trusted Offline Checkpoint Conversion Utility.

Converts legacy PyTorch checkpoints containing pickled Python objects into
pure tensor state-dictionaries that can be loaded securely with weights_only=True.

Usage:
    python scripts/convert_checkpoint.py --input legacy.pth --output secure.pth
"""

import argparse
import os
import sys
import torch


def convert_checkpoint(input_path: str, output_path: str) -> None:
    """
    Load a trusted legacy checkpoint offline and re-save pure state dict tensors.

    Args:
        input_path: Path to existing legacy checkpoint.
        output_path: Path where sanitized pure-tensor checkpoint will be saved.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input checkpoint does not exist: {input_path}")

    print(f"[Converter] Loading trusted checkpoint '{input_path}' (offline conversion)...")
    # Explicit offline load of trusted legacy file
    raw_ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"Expected dict-based checkpoint, got {type(raw_ckpt).__name__}")

    # Extract state dict
    if "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    elif "model" in raw_ckpt:
        state_dict = raw_ckpt["model"]
    elif "generator" in raw_ckpt:
        state_dict = raw_ckpt["generator"]
    else:
        state_dict = raw_ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Extracted state_dict must be a dictionary, got {type(state_dict).__name__}")

    # Validate all elements are pure PyTorch tensors
    sanitized = {}
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            sanitized[str(key)] = val.detach().cpu()
        else:
            print(f"[Warning] Skipping non-tensor metadata key: {key} ({type(val).__name__})")

    if not sanitized:
        raise ValueError("No tensor parameters found to save in converted checkpoint.")

    # Save pure state dict containing only tensors
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {"state_dict": sanitized}
    torch.save(payload, output_path)
    print(f"[Converter] Successfully wrote {len(sanitized)} tensors to '{output_path}'.")
    print("  ✅ Converted file can now be loaded securely with weights_only=True.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trusted legacy PyTorch checkpoints to secure weights_only=True format."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input legacy checkpoint")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output sanitized checkpoint")
    args = parser.parse_args()

    try:
        convert_checkpoint(args.input, args.output)
    except Exception as exc:
        print(f"[ERROR] Checkpoint conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

