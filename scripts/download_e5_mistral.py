"""Download E5-Mistral-7B-Instruct model to the local cache.

Tries multiple sources in order:
  1. ModelScope (best for China mainland)
  2. HuggingFace Hub (official)
  3. HuggingFace Mirror (hf-mirror.com)

Usage::

    python scripts/download_e5_mistral.py

The model is ~14 GB.  Make sure you have enough disk space (~15 GB free)
and a stable internet connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET_DIR = Path("cache/models/e5-mistral-7b-instruct")
MODEL_ID = "intfloat/e5-mistral-7b-instruct"


def _check_existing() -> bool:
    """Return True if the model dir already looks complete."""
    if not TARGET_DIR.exists():
        return False
    # Check for essential files
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not (TARGET_DIR / f).exists()]
    if missing:
        print(f"  Existing dir is incomplete (missing: {missing})")
        return False
    # Check for model weights (at least one .safetensors file)
    safetensors = list(TARGET_DIR.glob("*.safetensors"))
    if not safetensors:
        print("  No .safetensors files found — model weights not downloaded")
        return False
    total_gb = sum(f.stat().st_size for f in safetensors) / (1024**3)
    print(f"  Found {len(safetensors)} weight files ({total_gb:.1f} GB)")
    return True


def download_modelscope():
    """ModelScope (iic/e5-mistral-7b-instruct)."""
    print("\n── Trying ModelScope ──")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  modelscope not installed. Run: pip install modelscope")
        return False
    try:
        snapshot_download(
            "iic/e5-mistral-7b-instruct",
            local_dir=str(TARGET_DIR),
        )
        return True
    except Exception as exc:
        print(f"  ModelScope failed: {exc}")
        return False


def download_huggingface():
    """Official HuggingFace Hub."""
    print("\n── Trying HuggingFace Hub ──")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL_ID,
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True
    except Exception as exc:
        print(f"  HuggingFace failed: {exc}")
        return False


def download_hf_mirror():
    """hf-mirror.com (commonly used in China)."""
    import os
    print("\n── Trying HF Mirror ──")
    try:
        from huggingface_hub import snapshot_download
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        snapshot_download(
            MODEL_ID,
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True
    except Exception as exc:
        print(f"  HF Mirror failed: {exc}")
        return False


def main():
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if _check_existing():
        print(f"\nModel already downloaded: {TARGET_DIR.resolve()}")
        return

    print(f"Downloading {MODEL_ID} → {TARGET_DIR.resolve()}")
    print("This is a ~14 GB download. Please be patient.\n")

    for download_fn in (download_modelscope, download_huggingface, download_hf_mirror):
        if download_fn():
            if _check_existing():
                print(f"\n✓ Download complete: {TARGET_DIR.resolve()}")
                return
            else:
                print("  Download reported success but files are incomplete — trying next source")

    print("\n✗ All download sources failed.")
    print("\nManual download options:")
    print(f"  1. pip install modelscope && python -c \"from modelscope import snapshot_download; snapshot_download('iic/e5-mistral-7b-instruct', local_dir='{TARGET_DIR}')\"")
    print(f"  2. git lfs install && git clone https://huggingface.co/{MODEL_ID} {TARGET_DIR}")
    print(f"  3. Download from https://hf-mirror.com/{MODEL_ID} manually")
    sys.exit(1)


if __name__ == "__main__":
    main()
