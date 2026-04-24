"""Split an audio file into per-speaker clips from a transcribe_long JSON.

Reads `segments[*]` (start_time, end_time, global_speaker_id, text) from the
unify JSON and writes one WAV per segment under
`<out_dir>/<global_speaker_id>/<NNN>_<start>-<end>.wav`. Also drops an
`index.tsv` per speaker with timing + text so you can browse by content.

Usage:
    uv run --with soundfile python scripts/split_by_speaker.py \
        --audio test.mp3 \
        --json benchmarks/test_cannot_link.json \
        --out benchmarks/test_cannot_link_segments
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _decode(audio_path: Path, sr: int = 24000) -> tuple:
    import soundfile as sf

    if audio_path.suffix.lower() == ".wav":
        data, file_sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            raise SystemExit(
                f"WAV sample rate {file_sr} != {sr}; please pre-resample or pass mp3"
            )
        return data, sr

    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                str(tmp),
            ],
            check=True,
        )
        data, _ = sf.read(str(tmp), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--json", dest="json_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    doc = json.loads(args.json_path.read_text())
    segments = doc.get("segments", [])
    if not segments:
        print("no segments in JSON", file=sys.stderr)
        return 1

    print(f"decoding {args.audio} → {args.sr} Hz mono …")
    audio, sr = _decode(args.audio, args.sr)
    n = len(audio)

    args.out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    indexes: dict[str, list[str]] = {}

    for i, seg in enumerate(segments):
        gid = seg.get("global_speaker_id")
        if not gid:
            cid = seg.get("chunk_id")
            spk = seg.get("speaker_id")
            gid = (
                f"c{cid}_s{spk}" if cid is not None and spk is not None else "UNLABELED"
            )
        try:
            s = float(seg["start_time"])
            e = float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue

        s_idx = max(0, int(s * sr))
        e_idx = min(n, int(e * sr))
        if e_idx <= s_idx:
            continue

        sdir = args.out / gid
        sdir.mkdir(exist_ok=True)
        counts[gid] = counts.get(gid, 0) + 1
        idx = counts[gid]

        fname = f"{idx:03d}_{s:09.2f}-{e:09.2f}.wav"
        sf.write(str(sdir / fname), audio[s_idx:e_idx], sr)

        text = (seg.get("text") or "").replace("\t", " ").replace("\n", " ").strip()
        indexes.setdefault(gid, []).append(
            f"{idx:03d}\t{s:9.2f}\t{e:9.2f}\t{e - s:6.2f}\tc{seg.get('chunk_id')}s{seg.get('speaker_id')}\t{text}"
        )

    for gid, lines in indexes.items():
        idx_path = args.out / gid / "index.tsv"
        idx_path.write_text(
            "idx\tstart_s\tend_s\tdur_s\tlocal\ttext\n" + "\n".join(lines) + "\n"
        )

    print("\nper-speaker counts:")
    for gid in sorted(counts):
        total_s = sum(float(line.split("\t")[3]) for line in indexes[gid])
        print(f"  {gid}: {counts[gid]:3d} segments, {total_s:7.1f} s total")
    print(f"\nwrote to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
