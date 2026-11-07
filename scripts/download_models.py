"""
Download pretrained checkpoints and quantizer models from official Meta AI CDN.

Downloads:
1. mhubert_base.pt (~378 MB): Multilingual HuBERT Base model
2. mhubert_km1000.bin (~3.1 MB): 1,000-centroid K-Means model on Layer 11
3. matched_vocoder.pt (~53 MB): Unit HiFi-GAN generator checkpoint (g_00500000)
4. matched_config.json: Vocoder architecture configuration
"""

import argparse
import os
import urllib.request

FILES = {
    "mhubert_base.pt": "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt",
    "mhubert_km1000.bin": "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
    "matched_vocoder.pt": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000",
    "matched_config.json": "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json",
}


def download_checkpoints(output_dir: str = "checkpoints") -> None:
    """Download Meta AI model checkpoints into output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(output_dir, name)
        if os.path.exists(dest):
            print(f"[Skip] {name} already exists ({os.path.getsize(dest)/1024/1024:.1f} MB)")
            continue
        print(f"[Download] Fetching {name} from Meta CDN...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✅ Saved: {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained model weights.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Destination folder for weights")
    args = parser.parse_args()
    download_checkpoints(args.output_dir)
