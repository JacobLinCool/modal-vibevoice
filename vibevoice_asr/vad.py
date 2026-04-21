"""Silero VAD + greedy VAD-aware chunking for long-form audio."""

from __future__ import annotations

_VAD_WINDOW = 512  # 32 ms at 16 kHz, silero v5 native window size


class SileroVAD:
    """Silero VAD v5 run directly through onnxruntime (no torchaudio dep)."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def speech_probs(self, audio_16k):
        import numpy as np

        n_windows = len(audio_16k) // _VAD_WINDOW
        if n_windows == 0:
            return np.zeros(0, dtype=np.float32)
        probs = np.empty(n_windows, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(16000, dtype=np.int64)
        audio_f = audio_16k.astype(np.float32, copy=False)
        for i in range(n_windows):
            x = audio_f[i * _VAD_WINDOW : (i + 1) * _VAD_WINDOW][None, :]
            out, state = self.sess.run(None, {"input": x, "state": state, "sr": sr})
            probs[i] = float(out[0, 0])
        return probs


def vad_speech_ranges(
    audio,
    sr: int,
    vad: SileroVAD,
    threshold: float = 0.5,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.4,
    speech_pad_s: float = 0.1,
) -> list[tuple[int, int]]:
    """ONNX silero VAD → list of (start_sample, end_sample) at `sr`."""
    import librosa
    import numpy as np

    if sr != 16000:
        audio_16k = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=16000
        )
    else:
        audio_16k = audio.astype(np.float32)

    probs = vad.speech_probs(audio_16k)
    if len(probs) == 0:
        return []
    is_speech = probs > threshold

    # Contiguous speech windows → 16k sample ranges
    ranges_16k: list[list[int]] = []
    in_run = False
    run_start = 0
    for i, sp in enumerate(is_speech):
        if sp and not in_run:
            run_start = i
            in_run = True
        elif not sp and in_run:
            ranges_16k.append([run_start * _VAD_WINDOW, i * _VAD_WINDOW])
            in_run = False
    if in_run:
        ranges_16k.append([run_start * _VAD_WINDOW, len(is_speech) * _VAD_WINDOW])

    # Merge runs separated by < min_silence_s of silence
    min_silence_16k = int(min_silence_s * 16000)
    merged: list[list[int]] = []
    for s, e in ranges_16k:
        if merged and s - merged[-1][1] < min_silence_16k:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Drop too-short runs, pad, scale back to caller sr
    min_speech_16k = int(min_speech_s * 16000)
    pad_16k = int(speech_pad_s * 16000)
    n_16k = len(audio_16k)
    scale = sr / 16000
    out: list[tuple[int, int]] = []
    for s, e in merged:
        if e - s < min_speech_16k:
            continue
        s = max(0, s - pad_16k)
        e = min(n_16k, e + pad_16k)
        out.append((int(s * scale), int(e * scale)))
    return out


def chunk_by_vad(
    total_samples: int,
    sr: int,
    vad_ranges: list[tuple[int, int]],
    target_s: float,
    max_s: float,
    min_gap_s: float = 0.3,
) -> list[tuple[int, int]]:
    """Greedy split into chunks ≈ `target_s`, ending at silence gaps ≥ `min_gap_s`.

    Hard cap at `max_s`. Falls back to forced cuts when no silence is found
    within the allowed window (rare).
    """
    target = int(target_s * sr)
    max_samples = int(max_s * sr)
    min_gap = int(min_gap_s * sr)
    chunks: list[tuple[int, int]] = []
    cursor = 0

    while cursor < total_samples:
        ideal_end = cursor + target
        if ideal_end >= total_samples:
            chunks.append((cursor, total_samples))
            break

        max_end = min(cursor + max_samples, total_samples)
        if max_end >= total_samples:
            chunks.append((cursor, total_samples))
            break

        # Search silence gaps with midpoint in [cursor + 0.5*target, max_end].
        min_split = cursor + target // 2
        best_split = None
        best_dist = float("inf")
        for i in range(len(vad_ranges) - 1):
            gap_start = vad_ranges[i][1]
            gap_end = vad_ranges[i + 1][0]
            if gap_end - gap_start < min_gap:
                continue
            mid = (gap_start + gap_end) // 2
            if mid < min_split or mid > max_end:
                continue
            dist = abs(mid - ideal_end)
            if dist < best_dist:
                best_dist = dist
                best_split = mid

        if best_split is None or best_split <= cursor:
            best_split = max_end
        chunks.append((cursor, best_split))
        cursor = best_split

    return chunks
