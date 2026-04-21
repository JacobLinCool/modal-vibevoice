"""Segment post-processing helpers (chunk offsetting, tail priming)."""

from __future__ import annotations


def shift_segments(segments: list[dict], offset_s: float, chunk_id: int) -> list[dict]:
    """Apply chunk offset to timestamps and tag a fallback `global_speaker_id`."""
    out = []
    for seg in segments:
        s = dict(seg)
        for k in ("start_time", "end_time"):
            v = s.get(k)
            if v is None:
                continue
            try:
                s[k] = round(float(v) + offset_s, 3)
            except (TypeError, ValueError):
                pass
        s["chunk_id"] = chunk_id
        spk = s.get("speaker_id")
        if spk is not None:
            s["global_speaker_id"] = f"c{chunk_id}_s{spk}"
        out.append(s)
    return out


def tail_text(segments: list[dict], tail_seconds: float, max_chars: int = 600) -> str:
    """Return concatenated text of trailing segments covering ~`tail_seconds`."""
    if not segments:
        return ""
    try:
        last_end = max(float(s.get("end_time", 0) or 0) for s in segments)
    except (TypeError, ValueError):
        return ""
    cutoff = last_end - tail_seconds
    chosen = [
        s.get("text", "")
        for s in segments
        if (
            s.get("end_time") is not None and float(s.get("end_time", 0) or 0) >= cutoff
        )
    ]
    text = " ".join(t for t in chosen if t).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text
