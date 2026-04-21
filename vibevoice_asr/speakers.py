"""Speaker x-vector extraction (CAM++ via sherpa-onnx) and cross-chunk unification."""

from __future__ import annotations


class SpeakerEmbedder:
    """WeSpeaker CAM++ via sherpa-onnx → 192-dim x-vector.

    Runs purely through onnxruntime on CPU; language-agnostic in practice.
    """

    def __init__(self, model_path: str, num_threads: int = 2) -> None:
        import sherpa_onnx

        self.model = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=model_path,
                num_threads=num_threads,
                debug=False,
                provider="cpu",
            )
        )

    def embed(self, audio_16k):
        """Run CAM++ on one mono 16 kHz waveform → 192-d x-vector np.ndarray."""
        import numpy as np

        stream = self.model.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=audio_16k.astype(np.float32))
        stream.input_finished()
        emb = self.model.compute(stream)
        return np.asarray(emb, dtype=np.float32)


def unify_speakers(
    audio,
    sr: int,
    segments: list[dict],
    embedder: SpeakerEmbedder,
    distance_threshold: float = 0.3,
    min_audio_per_speaker_s: float = 1.0,
    max_audio_per_speaker_s: float = 30.0,
) -> dict:
    """Cluster (chunk_id, local_speaker_id) keys across chunks via x-vectors.

    Mutates `segments` in place: rewrites `global_speaker_id` to `S{k}` for
    the global cluster id, or leaves the per-chunk fallback when there's
    not enough audio for a reliable embedding.
    """
    import time as _time
    from collections import Counter

    import librosa
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    t0 = _time.perf_counter()

    # Collect audio slices for each (chunk_id, local_speaker_id)
    per_speaker: dict[tuple[int, int], list] = {}
    for seg in segments:
        spk = seg.get("speaker_id")
        ck = seg.get("chunk_id")
        if spk is None or ck is None:
            continue
        try:
            s = int(float(seg["start_time"]) * sr)
            e = int(float(seg["end_time"]) * sr)
        except (KeyError, TypeError, ValueError):
            continue
        s = max(0, min(s, len(audio)))
        e = max(s, min(e, len(audio)))
        if e - s <= 0:
            continue
        per_speaker.setdefault((ck, int(spk)), []).append((s, e))

    keys: list[tuple[int, int]] = []
    embeddings: list = []
    skipped: list[dict] = []
    min_samples = int(min_audio_per_speaker_s * sr)
    max_samples = int(max_audio_per_speaker_s * sr)

    for key, ranges in per_speaker.items():
        slices = [audio[s:e] for s, e in ranges]
        if not slices:
            continue
        joined = np.concatenate(slices)
        if len(joined) < min_samples:
            skipped.append(
                {
                    "chunk_id": key[0],
                    "local_speaker_id": key[1],
                    "audio_s": round(len(joined) / sr, 3),
                }
            )
            continue
        if len(joined) > max_samples:
            joined = joined[:max_samples]
        joined_16k = librosa.resample(
            joined.astype(np.float32), orig_sr=sr, target_sr=16000
        )
        embeddings.append(embedder.embed(joined_16k))
        keys.append(key)

    if len(keys) == 0:
        return {
            "elapsed_s": round(_time.perf_counter() - t0, 3),
            "num_global_speakers": 0,
            "num_keys_embedded": 0,
            "num_skipped": len(skipped),
            "skipped": skipped,
        }

    embs = np.stack(embeddings, axis=0)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    # Pairwise cosine distance matrix (for diagnostics)
    sim = embs @ embs.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist_matrix = (1.0 - sim).round(4).tolist()

    if len(keys) == 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(embs)

    mapping = {k: int(lab) for k, lab in zip(keys, labels)}

    # Rewrite segments
    for seg in segments:
        ck = seg.get("chunk_id")
        sp = seg.get("speaker_id")
        if ck is None or sp is None:
            continue
        key = (ck, int(sp))
        if key in mapping:
            seg["global_speaker_id"] = f"S{mapping[key]}"

    sizes = Counter(int(label) for label in labels)
    return {
        "elapsed_s": round(_time.perf_counter() - t0, 3),
        "num_global_speakers": int(len(set(labels))),
        "num_keys_embedded": int(len(keys)),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "distance_threshold": distance_threshold,
        "cluster_sizes": {f"S{k}": v for k, v in sorted(sizes.items())},
        "keys": [{"chunk_id": k[0], "local_speaker_id": k[1]} for k in keys],
        "distance_matrix": dist_matrix,
        "mapping": [
            {
                "chunk_id": k[0],
                "local_speaker_id": k[1],
                "global_speaker_id": f"S{mapping[k]}",
            }
            for k in keys
        ],
    }
