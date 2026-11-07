"""
Montreal Forced Aligner (MFA) Corpus Preparation Utility.

Authorship & Provenance:
- Original Author: Achouri Youcef Islem (April 29, 2026)
- Original File: C:\\mini-projet-dl\\prepare_mfa_corpus.py (SHA-256: DC4BC371...)
- Evolution: Refactored into a typed, cross-platform library module for this release.

Purpose:
The raw StethoSpeech corpus distribution contains nested speaker directories
with separate WAV audio recordings and TXT transcripts across Seated and Mobility
recording sessions. This module extracts matching (audio, text) pairs and
restructures them into the canonical directory structure required by MFA:
  output_corpus/<speaker_id>/<base_name>.wav
  output_corpus/<speaker_id>/<base_name>.txt
"""

import argparse
import glob
import os
import shutil
from typing import Dict, List, Optional, Tuple


def prepare_mfa_corpus(
    mobility_dir: Optional[str] = None,
    seated_dir: Optional[str] = None,
    output_corpus: Optional[str] = None,
    copy_files: bool = True,
) -> int:
    """
    Search for audio (.wav) and transcript (.txt) pairs and structure them for MFA.

    Args:
        mobility_dir: Directory containing Mobility condition speaker folders.
        seated_dir: Directory containing Seated condition speaker folders.
        output_corpus: Destination directory for organized MFA corpus.
        copy_files: If True, copies files; if False, only scans and counts pairs.

    Returns:
        Total number of paired audio/transcript files successfully processed.
    """
    is_kaggle = os.path.exists("/kaggle/input")

    # Resolve default paths if not explicitly provided
    if output_corpus is None:
        output_corpus = "/kaggle/working/nam_corpus" if is_kaggle else "data/nam_corpus"

    if mobility_dir is None:
        if is_kaggle:
            mobility_dir = (
                "/kaggle/input/datasets/achouriyoucefislem/mobility/Mobility"
                if os.path.exists("/kaggle/input/datasets/achouriyoucefislem/mobility/Mobility")
                else "/kaggle/input/mobility/Mobility"
            )
        else:
            mobility_dir = "data/raw/Mobility"

    if seated_dir is None:
        if is_kaggle:
            seated_dir = (
                "/kaggle/input/datasets/achouriyoucefislem/seated/Seated"
                if os.path.exists("/kaggle/input/datasets/achouriyoucefislem/seated/Seated")
                else "/kaggle/input/seated/Seated"
            )
        else:
            seated_dir = "data/raw/Seated"

    if copy_files:
        os.makedirs(output_corpus, exist_ok=True)

    mobility_speakers = glob.glob(os.path.join(mobility_dir, "*")) if os.path.exists(mobility_dir) else []
    seated_speakers = glob.glob(os.path.join(seated_dir, "*")) if os.path.exists(seated_dir) else []
    all_speakers = mobility_speakers + seated_speakers

    total_pairs = 0
    print(f"[MFA Corpus] Found {len(all_speakers)} speaker candidate directories across paths.")

    for speaker_path in all_speakers:
        if not os.path.isdir(speaker_path):
            continue

        spk_name = os.path.basename(speaker_path)
        dest_spk_dir = os.path.join(output_corpus, spk_name)
        if copy_files:
            os.makedirs(dest_spk_dir, exist_ok=True)

        wav_files = glob.glob(os.path.join(speaker_path, "**", "*.wav"), recursive=True)
        txt_files = glob.glob(os.path.join(speaker_path, "**", "*.txt"), recursive=True)

        txt_map: Dict[str, str] = {
            os.path.splitext(os.path.basename(f))[0].lower(): f for f in txt_files
        }

        speaker_pair_count = 0
        for wav_path in wav_files:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            base_key = base_name.lower()

            if base_key in txt_map:
                txt_path = txt_map[base_key]
                if copy_files:
                    dest_wav = os.path.join(dest_spk_dir, f"{base_name}.wav")
                    dest_txt = os.path.join(dest_spk_dir, f"{base_name}.txt")

                    if not os.path.exists(dest_wav):
                        shutil.copy2(wav_path, dest_wav)
                    if not os.path.exists(dest_txt):
                        shutil.copy2(txt_path, dest_txt)

                speaker_pair_count += 1
                total_pairs += 1

        print(f"  -> Speaker '{spk_name}': {speaker_pair_count} pairs paired.")

    print(f"[MFA Corpus] Total dataset prepared: {total_pairs} paired files in '{output_corpus}'.")
    return total_pairs


def main() -> None:
    """CLI entry point for MFA corpus preparation."""
    parser = argparse.ArgumentParser(description="Prepare MFA corpus from raw NAM recordings.")
    parser.add_argument("--mobility", type=str, default=None, help="Path to raw Mobility dataset directory")
    parser.add_argument("--seated", type=str, default=None, help="Path to raw Seated dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Path to destination MFA corpus folder")
    parser.add_argument("--dry-run", action="store_true", help="Scan and count pairs without copying files")

    args = parser.parse_args()
    prepare_mfa_corpus(
        mobility_dir=args.mobility,
        seated_dir=args.seated,
        output_corpus=args.output,
        copy_files=not args.dry_run,
    )


if __name__ == "__main__":
    main()
