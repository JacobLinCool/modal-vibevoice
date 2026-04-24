"""Modal provider for VibeVoice-ASR.

Optimizations applied for fastest inference:
  * NVIDIA NGC PyTorch 25.12 base image (PyTorch + CUDA + flash-attn prebuilt)
  * bf16 weights + flash_attention_2 kernels on H100
  * TF32 matmul, cuDNN benchmark, inference_mode
  * Persistent HuggingFace cache Volume (weights downloaded once)
  * hf_transfer for faster HF downloads
  * Model weights baked into the image (snapshot_download at build time)
  * Warm-up forward pass inside @modal.enter so the first real request skips
    CUDA kernel / autotuner compilation

Inference logic lives in `vibevoice_asr/`. This file is the Modal-specific
shim: image build, @app.cls with the three-stage lifecycle (build → enter →
request), and local entrypoints. Other providers should wrap the same
`VibeVoiceASRRunner` and replicate these stages in their own way.
"""

from __future__ import annotations

import os

import modal

from vibevoice_asr.config import (
    HF_CACHE_DIR,
    SILERO_VAD_PATH,
    SILERO_VAD_URL,
    SPEAKER_MODEL_PATH,
    SPEAKER_MODEL_URL,
    RunnerConfig,
)

APP_NAME = "vibevoice-asr"

# Override at deploy/run time to bench other accelerators:
#   MODAL_GPU=H100 uv run modal run app.py::bench --audio-path ...
#   MODAL_GPU=A100-80GB uv run modal run app.py::long ...
#   MODAL_GPU=A100-40GB uv run modal run app.py::long ...
GPU_TYPE = os.environ.get("MODAL_GPU", "RTX-PRO-6000")

# Override to swap in a quantized variant (e.g. scerz/VibeVoice-ASR-4bit).
# 4-bit bnb checkpoints carry their own `quantization_config` in config.json,
# so the loader auto-applies it — only the `bitsandbytes` dependency is new.
MODEL_NAME = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR")


def _runner_config() -> RunnerConfig:
    return RunnerConfig(model_name=MODEL_NAME)


hf_cache = modal.Volume.from_name("vibevoice-hf-cache", create_if_missing=True)


def _prefetch_weights() -> None:
    # Stage 1 (BUILD): run inside the image build, baking HF weights into the
    # image layer so `@modal.enter` never hits the network.
    from vibevoice_asr.runner import VibeVoiceASRRunner

    VibeVoiceASRRunner.prefetch_weights(_runner_config())


image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.12-py3",
        add_python=None,
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONUNBUFFERED": "1",
            # Bake MODEL_NAME into the image so remote build (prefetch) and
            # runtime containers both see it — otherwise they re-import app.py
            # without the local shell's env and silently fall back to default.
            # Becomes part of the image cache key, so swapping models rebuilds.
            "VIBEVOICE_MODEL": MODEL_NAME,
            # NGC 25.12 links PyTorch against CUDA 13.1, but the host driver
            # advertises only CUDA 13.0 (see the compat-mode warning at
            # container start). bitsandbytes ships pre-compiled .so files up
            # through cuda130 — pin it so bnb loads a binary the driver runs.
            "BNB_CUDA_VERSION": "130",
        }
    )
    .run_commands(
        "pip install --upgrade pip uv",
        (
            "uv pip install --system --no-cache-dir "
            "'transformers>=4.51.3,<5.0.0' accelerate hf_transfer huggingface_hub "
            "bitsandbytes "
            "librosa soundfile scipy diffusers ml-collections absl-py "
            "tqdm pydub 'numba>=0.57.0' 'llvmlite>=0.40.0' fastapi python-multipart "
            "pynvml onnxruntime scikit-learn sherpa-onnx"
        ),
        (
            "uv pip install --system --no-cache-dir --no-deps "
            "git+https://github.com/microsoft/VibeVoice.git"
        ),
        # ONNX assets fetched in the image layer so they layer-cache and are
        # ready by the time any request arrives.
        f"curl -fsSL {SILERO_VAD_URL} -o {SILERO_VAD_PATH}",
        f"curl -fsSL {SPEAKER_MODEL_URL} -o {SPEAKER_MODEL_PATH}",
    )
    # Copy the runner into the image so `run_function` can import it.
    .add_local_python_source("vibevoice_asr", copy=True)
    .run_function(_prefetch_weights, volumes={HF_CACHE_DIR: hf_cache})
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=3600,
    scaledown_window=60,
)
class VibeVoiceASR:
    @modal.enter()
    def _enter(self) -> None:
        # Stage 2 (ENTER): once per container.
        from vibevoice_asr.runner import VibeVoiceASRRunner

        self.runner = VibeVoiceASRRunner(_runner_config())
        self.runner.load()

    # Stage 3 (REQUEST): every method below is a thin wrapper over the
    # corresponding runner method so every provider can offer the same API.

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_info: str | None = None,
    ) -> dict:
        return self.runner.transcribe(
            audio_bytes,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            context_info=context_info,
        )

    @modal.method()
    def transcribe_batch(
        self,
        audio_bytes_list: list[bytes],
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        context_info: str | None = None,
    ) -> list[dict]:
        return self.runner.transcribe_batch(
            audio_bytes_list,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            context_info=context_info,
        )

    @modal.method()
    def benchmark(
        self,
        audio_bytes: bytes,
        max_new_tokens: int = 32768,
        sample_hz: float = 2.0,
        context_info: str | None = None,
    ) -> dict:
        return self.runner.benchmark(
            audio_bytes,
            max_new_tokens=max_new_tokens,
            sample_hz=sample_hz,
            context_info=context_info,
        )

    @modal.method()
    def transcribe_long(
        self,
        audio_bytes: bytes,
        context_info: str | None = None,
        chunk_target_s: float = 2700.0,
        chunk_max_s: float = 3300.0,
        prime_with_prev_tail: bool = True,
        prev_tail_seconds: float = 30.0,
        max_new_tokens: int = 32768,
        batch_size: int = 0,
        unify_speakers: bool = True,
        unify_distance_threshold: float = 0.3,
        return_speaker_embeddings: bool = True,
    ) -> dict:
        return self.runner.transcribe_long(
            audio_bytes,
            context_info=context_info,
            chunk_target_s=chunk_target_s,
            chunk_max_s=chunk_max_s,
            prime_with_prev_tail=prime_with_prev_tail,
            prev_tail_seconds=prev_tail_seconds,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            unify_speakers=unify_speakers,
            unify_distance_threshold=unify_distance_threshold,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile

        api = FastAPI(title="VibeVoice-ASR")

        @api.get("/healthz")
        async def healthz():
            return {"ok": True}

        @api.post("/transcribe")
        async def transcribe_endpoint(
            audio: UploadFile = File(...),
            max_new_tokens: int = Form(32768),
            num_beams: int = Form(1),
            temperature: float = Form(0.0),
            top_p: float = Form(1.0),
            context_info: str | None = Form(None),
        ):
            data = await audio.read()
            if not data:
                raise HTTPException(400, "empty upload")
            return self.runner.transcribe(
                data,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                context_info=context_info,
            )

        @api.post("/transcribe_long")
        async def transcribe_long_endpoint(
            audio: UploadFile = File(...),
            context_info: str | None = Form(None),
            chunk_target_s: float = Form(2700.0),
            chunk_max_s: float = Form(3300.0),
            prime_with_prev_tail: bool = Form(True),
            prev_tail_seconds: float = Form(30.0),
            max_new_tokens: int = Form(32768),
            batch_size: int = Form(0),
            unify_speakers: bool = Form(True),
            unify_distance_threshold: float = Form(0.3),
            return_speaker_embeddings: bool = Form(True),
        ):
            data = await audio.read()
            if not data:
                raise HTTPException(400, "empty upload")
            return self.runner.transcribe_long(
                data,
                context_info=context_info,
                chunk_target_s=chunk_target_s,
                chunk_max_s=chunk_max_s,
                prime_with_prev_tail=prime_with_prev_tail,
                prev_tail_seconds=prev_tail_seconds,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                unify_speakers=unify_speakers,
                unify_distance_threshold=unify_distance_threshold,
                return_speaker_embeddings=return_speaker_embeddings,
            )

        return api


@app.local_entrypoint()
def main(
    audio_path: str,
    max_new_tokens: int = 32768,
    num_beams: int = 1,
    context_info: str | None = None,
):
    from pathlib import Path

    data = Path(audio_path).read_bytes()
    result = VibeVoiceASR().transcribe.remote(
        data,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        context_info=context_info,
    )
    print("\n=== Raw ===")
    print(result["text"])
    if result["segments"]:
        print(f"\n=== Segments ({len(result['segments'])}) ===")
        for seg in result["segments"][:100]:
            print(
                f"[{seg.get('start_time', '?')} - {seg.get('end_time', '?')}] "
                f"spk={seg.get('speaker_id', '?')}: {seg.get('text', '')}"
            )


@app.local_entrypoint()
def bench(
    audio_path: str,
    max_new_tokens: int = 32768,
    context_info: str | None = None,
):
    """Run the benchmark path and print a formatted report."""
    import time
    from pathlib import Path

    from vibevoice_asr.reporting import print_bench_report

    data = Path(audio_path).read_bytes()
    print(f"Uploading {len(data) / 1024**2:.1f} MB from {audio_path}")

    t0 = time.perf_counter()
    result = VibeVoiceASR().benchmark.remote(
        data, max_new_tokens=max_new_tokens, context_info=context_info
    )
    wall = time.perf_counter() - t0

    print_bench_report(result, wall)


@app.local_entrypoint()
def long(
    audio_path: str,
    context_info: str | None = None,
    chunk_target_min: float = 45.0,
    chunk_max_min: float = 55.0,
    no_prime: bool = False,
    prev_tail_seconds: float = 30.0,
    max_new_tokens: int = 32768,
    batch_size: int = 0,
    no_unify: bool = False,
    unify_distance_threshold: float = 0.3,
    no_speaker_embeddings: bool = False,
    out_json: str | None = None,
):
    """Long-form (multi-hour) transcription via VAD-aware chunking."""
    import time
    from pathlib import Path

    from vibevoice_asr.reporting import print_long_report

    data = Path(audio_path).read_bytes()
    print(f"Uploading {len(data) / 1024**2:.1f} MB from {audio_path}")

    t0 = time.perf_counter()
    result = VibeVoiceASR().transcribe_long.remote(
        data,
        context_info=context_info,
        chunk_target_s=chunk_target_min * 60,
        chunk_max_s=chunk_max_min * 60,
        prime_with_prev_tail=not no_prime,
        prev_tail_seconds=prev_tail_seconds,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        unify_speakers=not no_unify,
        unify_distance_threshold=unify_distance_threshold,
        return_speaker_embeddings=not no_speaker_embeddings,
    )
    wall = time.perf_counter() - t0

    print_long_report(result, wall, out_json)
