"""Minimal client that hits the deployed VibeVoice-ASR web endpoint.

Usage:
    uv run python client.py <audio_file> [endpoint_url]

Find the endpoint URL in the `modal deploy` output (the `/web` URL of
the `VibeVoiceASR` class).
"""

from __future__ import annotations

import sys
from pathlib import Path

import urllib.request


def transcribe(audio_path: str, endpoint_url: str) -> None:

    boundary = "----modalvibevoiceboundary"
    audio_bytes = Path(audio_path).read_bytes()
    filename = Path(audio_path).name

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        endpoint_url.rstrip("/") + "/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        print(resp.read().decode())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python client.py <audio_file> <endpoint_url>")
        sys.exit(1)
    transcribe(sys.argv[1], sys.argv[2])
