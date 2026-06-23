"""Model file management — download the MediaPipe hand model on demand.

The .task model is a ~7.5MB binary artifact, so it's downloaded rather than
committed to git. `ensure_hand_model()` fetches it once and caches it under
models/; HandTracker calls this automatically, and scripts/download_model.py
exposes it on the command line.
"""

from __future__ import annotations

import urllib.request

from . import config


def ensure_hand_model() -> str:
    """Return the local path to the hand model, downloading it if it's missing."""
    path = config.HAND_MODEL_PATH
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[models] downloading hand model -> {path} ...")
    urllib.request.urlretrieve(config.HAND_MODEL_URL, path)
    print(f"[models] done ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)
