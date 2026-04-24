"""Plot per-key x-vectors returned in `unify.speaker_embeddings`.

Produces a two-panel figure:
  * Left  : PCA 2D scatter of the embeddings, coloured by global_speaker_id
            and annotated with the (chunk, local) key.
  * Right : heatmap of pairwise cosine distance between the same keys,
            matching `unify.distance_matrix`.

Usage:
    uv run --with numpy --with scikit-learn --with matplotlib \
        python scripts/plot_embeddings.py \
        --json benchmarks/test_cannot_link.json \
        --out benchmarks/test_cannot_link_embeddings.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    doc = json.loads(args.json_path.read_text())
    u = doc.get("unify", {})
    spk_embs = u.get("speaker_embeddings")
    if not spk_embs:
        print(
            "no speaker_embeddings in JSON (run with return_speaker_embeddings=True)",
            file=sys.stderr,
        )
        return 1

    rows: list[tuple[str, int, int, list[float]]] = []
    for gid, entries in spk_embs.items():
        for e in entries:
            rows.append((gid, e["chunk_id"], e["local_speaker_id"], e["embedding"]))

    gids = [r[0] for r in rows]
    labels = [f"c{r[1]}s{r[2]}" for r in rows]
    X = np.asarray([r[3] for r in rows], dtype=np.float32)

    Xp = PCA(n_components=2).fit_transform(X) if len(X) >= 2 else np.zeros((len(X), 2))

    unique_gids = sorted(set(gids))
    cmap = plt.get_cmap("tab10")
    color_of = {g: cmap(i % 10) for i, g in enumerate(unique_gids)}

    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(13, 5))

    for gid in unique_gids:
        mask = [g == gid for g in gids]
        pts = Xp[mask]
        ax_s.scatter(
            pts[:, 0],
            pts[:, 1],
            color=color_of[gid],
            label=gid,
            s=140,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.7,
        )
    for (x, y), lab in zip(Xp, labels):
        ax_s.annotate(
            lab, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9
        )
    ax_s.set_title(f"per-key x-vectors (PCA 2D, N={len(X)})")
    ax_s.set_xlabel("PC1")
    ax_s.set_ylabel("PC2")
    ax_s.grid(alpha=0.25)
    ax_s.legend(title="global speaker", loc="best", frameon=True)

    dist = u.get("distance_matrix")
    if dist:
        dm = np.asarray(dist, dtype=np.float32)
        im = ax_h.imshow(dm, cmap="viridis_r", vmin=0.0, vmax=max(0.5, float(dm.max())))
        ax_h.set_xticks(range(len(labels)))
        ax_h.set_yticks(range(len(labels)))
        ax_h.set_xticklabels(labels, rotation=45, ha="right")
        ax_h.set_yticklabels(labels)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                ax_h.text(
                    j,
                    i,
                    f"{dm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if dm[i, j] > 0.25 else "black",
                    fontsize=8,
                )
        thr = u.get("distance_threshold")
        title = "pairwise cosine distance"
        if thr is not None:
            title += f"  (threshold={thr})"
        ax_h.set_title(title)
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
