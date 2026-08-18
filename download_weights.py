"""
Fetch the trained X-VQA checkpoint.

The checkpoint (~1.1 GB) is not tracked in git. It is published as a
GitHub Release asset. Run this once before starting the app:

    python download_weights.py
"""

import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# EDIT THIS after you create the GitHub Release (see SETUP.md, step 5).
# Format: https://github.com/<user>/<repo>/releases/download/<tag>/best_model.pth
# ---------------------------------------------------------------------------
WEIGHTS_URL = os.environ.get(
    "XVQA_WEIGHTS_URL",
    "https://github.com/YOUR_USERNAME/xvqa-explainable-vqa/releases/download/v1.0/best_model.pth",
)
DEFAULT_DEST = "best_model.pth"


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    done = count * block_size
    pct = min(100.0, done * 100.0 / total_size)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.0f} / {total_size / 1e6:.0f} MB)")
    sys.stdout.flush()


def download(dest: str = DEFAULT_DEST, url: str = WEIGHTS_URL) -> str:
    if os.path.exists(dest):
        print(f"{dest} already present — skipping download.")
        return dest
    if "YOUR_USERNAME" in url:
        raise RuntimeError(
            "WEIGHTS_URL has not been configured. Edit download_weights.py "
            "(or set the XVQA_WEIGHTS_URL env var) to point at your release asset."
        )
    print(f"Downloading checkpoint from {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    os.replace(tmp, dest)
    print(f"\nSaved to {dest}")
    return dest


if __name__ == "__main__":
    download()
