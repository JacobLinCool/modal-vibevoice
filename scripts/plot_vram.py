"""Render the chunk-minute × VRAM × safe-batch curves for VibeVoice-ASR.

VRAM model (empirically calibrated against the multi-GPU benchmark):
    peak_vram(GB) = weights + headroom + KV_per_min × chunk_minutes × batch_size
                  = 18    +   5      + 0.15           × chunk_minutes × batch_size

Solving for batch:
    safe_batch = floor((vram - 23) / (0.15 × chunk_minutes))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WEIGHTS_GB = 18.0
HEADROOM_GB = 5.0
KV_PER_MIN_GB = 0.15  # measured: 28 GB peak − 18 GB weights = 10 GB / (3 chunks × 20 min) ≈ 0.167; round down

# Modal GPUs we tested + L40S as a workstation reference.
GPUS = [
    ("RTX PRO 6000 Blackwell 96 GB", 96, "tab:purple"),
    ("H100 80GB", 80, "tab:red"),
    ("A100-80GB", 80, "tab:orange"),
    ("L40S 48GB", 48, "tab:green"),
    ("A100-40GB", 40, "tab:blue"),
]


def safe_batch(vram_gb: float, chunk_minutes: np.ndarray) -> np.ndarray:
    free = max(0.0, vram_gb - WEIGHTS_GB - HEADROOM_GB)
    return np.maximum(1, np.floor(free / (KV_PER_MIN_GB * chunk_minutes))).astype(int)


def main() -> None:
    chunk_min = np.linspace(5, 60, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Plot 1: chunk_min vs safe_batch, one line per GPU ----------------
    ax = axes[0]
    for label, vram, color in GPUS:
        # Skip duplicate H100/A100-80 line (same VRAM → same curve)
        if label.startswith("A100-80GB"):
            continue
        ax.plot(
            chunk_min,
            safe_batch(vram, chunk_min),
            label=label,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("chunk size (minutes)")
    ax.set_ylabel("safe max batch_size (chunks in flight)")
    ax.set_title("Safe max batch_size vs chunk size, by GPU VRAM")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xticks([5, 10, 15, 20, 30, 45, 60])
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.axvline(20, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(45, color="gray", linestyle="--", alpha=0.4)
    ax.text(20, 30, "20-min", ha="center", fontsize=8, color="gray")
    ax.text(45, 30, "45-min", ha="center", fontsize=8, color="gray")
    ax.legend(loc="upper right", fontsize=9)

    # --- Plot 2: chunk_min vs peak VRAM at fixed batch sizes --------------
    ax = axes[1]
    for B in [1, 2, 4, 8, 16]:
        peak = WEIGHTS_GB + HEADROOM_GB + KV_PER_MIN_GB * chunk_min * B
        ax.plot(chunk_min, peak, label=f"batch={B}", linewidth=2)

    for _, vram, color in GPUS:
        ax.axhline(vram, linestyle="--", color=color, alpha=0.55, linewidth=1)
        ax.text(60.5, vram, f"{vram} GB", color=color, va="center", fontsize=9)

    ax.set_xlabel("chunk size (minutes)")
    ax.set_ylabel("predicted peak VRAM (GB)")
    ax.set_title("Predicted peak VRAM vs chunk size at fixed batch_size")
    ax.set_xticks([5, 10, 15, 20, 30, 45, 60])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(5, 70)
    ax.set_ylim(0, 130)
    ax.legend(loc="upper left", fontsize=9, title="batch")

    fig.suptitle(
        f"VibeVoice-ASR VRAM: {WEIGHTS_GB:.0f} GB weights "
        f"+ {HEADROOM_GB:.0f} GB headroom + {KV_PER_MIN_GB} GB/(chunk·min) × batch",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).resolve().parent.parent / "assets" / "vram_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    # Pretty table: max batch per GPU at common chunk sizes
    print("\nMax safe batch_size table:")
    chunks = [10, 20, 30, 45, 55]
    header = "GPU                              " + "".join(
        f"{c:>5} min" for c in chunks
    )
    print(header)
    print("-" * len(header))
    for label, vram, _ in GPUS:
        cells = "".join(
            f"{int(safe_batch(vram, np.array([c])).item()):>9}" for c in chunks
        )
        print(f"{label:<33}{cells}")


if __name__ == "__main__":
    main()
