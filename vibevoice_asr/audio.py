"""Audio decoding helper."""

from __future__ import annotations


def decode_audio(audio_bytes: bytes, target_sr: int):
    """Decode any container to mono float32 at `target_sr` via librosa/ffmpeg."""
    import io

    import librosa
    import numpy as np

    arr, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    return arr.astype(np.float32)
