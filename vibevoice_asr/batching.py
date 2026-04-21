"""Auto-batch-size heuristic for long-form transcription.

Empirically calibrated VRAM model; see scripts/plot_vram.py.
"""

from __future__ import annotations

_WEIGHTS_GB = 18.0
_HEADROOM_GB = 5.0
_KV_GB_PER_CHUNK_MIN = 0.22


def auto_batch_size(
    gpu_total_mem_bytes: int,
    chunk_minutes: float,
    cap: int = 16,
) -> int:
    """Choose a safe batch_size for the current GPU/chunk-size combination."""
    free_gb = max(0.0, gpu_total_mem_bytes / 1024**3 - _WEIGHTS_GB - _HEADROOM_GB)
    if chunk_minutes <= 0:
        return 1
    raw = int(free_gb // (_KV_GB_PER_CHUNK_MIN * chunk_minutes))
    return max(1, min(cap, raw))
