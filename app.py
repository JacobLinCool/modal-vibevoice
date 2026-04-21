"""Modal app serving Microsoft VibeVoice-ASR.

Optimizations applied for fastest inference:
  * NVIDIA NGC PyTorch 25.12 base image (PyTorch + CUDA + flash-attn prebuilt)
  * bf16 weights + flash_attention_2 kernels on H100
  * TF32 matmul, cuDNN benchmark, inference_mode
  * Persistent HuggingFace cache Volume (weights downloaded once)
  * hf_transfer for faster HF downloads
  * Model weights baked into the image (snapshot_download at build time)
  * Warm-up forward pass inside @modal.enter so the first real request skips
    CUDA kernel / autotuner compilation
"""

from __future__ import annotations

import os

import modal

APP_NAME = "vibevoice-asr"
MODEL_NAME = "microsoft/VibeVoice-ASR"
QWEN_TOKENIZER = "Qwen/Qwen2.5-7B"
# WeSpeaker CAM++ ONNX (VoxCeleb-trained, 192-dim x-vector). Language-
# agnostic in practice; for pure Mandarin workloads swap in the 3D-Speaker
# zh-cn model:
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/
#     speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
# Exported by sherpa-onnx; runs purely through onnxruntime and avoids the
# torchaudio dependency that blocks SpeechBrain in NVIDIA's NGC container.
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "wespeaker_en_voxceleb_CAM++.onnx"
)
SPEAKER_MODEL_PATH = "/opt/spk_wespeaker_en_campplus.onnx"

# Override at deploy/run time to bench other accelerators:
#   MODAL_GPU=H100 uv run modal run app.py::bench --audio-path ...
#   MODAL_GPU=A100-80GB uv run modal run app.py::long ...
#   MODAL_GPU=A100-40GB uv run modal run app.py::long ...
GPU_TYPE = os.environ.get("MODAL_GPU", "RTX-PRO-6000")
HF_CACHE_DIR = "/root/.cache/huggingface"
TARGET_SR = 24_000

hf_cache = modal.Volume.from_name("vibevoice-hf-cache", create_if_missing=True)


def _prefetch_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    snapshot_download(
        QWEN_TOKENIZER,
        cache_dir=HF_CACHE_DIR,
        allow_patterns=[
            "tokenizer*",
            "*.json",
            "merges.txt",
            "vocab.txt",
            "special_tokens_map.json",
        ],
    )
    # speaker model fetched at image-build time via curl, nothing to do here


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
        }
    )
    .run_commands(
        "pip install --upgrade pip uv",
        (
            "uv pip install --system --no-cache-dir "
            "'transformers>=4.51.3,<5.0.0' accelerate hf_transfer huggingface_hub "
            "librosa soundfile scipy diffusers ml-collections absl-py "
            "tqdm pydub 'numba>=0.57.0' 'llvmlite>=0.40.0' fastapi python-multipart "
            "pynvml onnxruntime scikit-learn sherpa-onnx"
        ),
        (
            "uv pip install --system --no-cache-dir --no-deps "
            "git+https://github.com/microsoft/VibeVoice.git"
        ),
        # silero-vad ONNX (run via onnxruntime; avoids the torchaudio wheel
        # that ABI-conflicts with NGC's PyTorch nightly).
        (
            "curl -fsSL "
            "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx "
            "-o /opt/silero_vad.onnx"
        ),
        # 3D-Speaker CAM++ Mandarin x-vector ONNX (~7 MB).
        f"curl -fsSL {SPEAKER_MODEL_URL} -o {SPEAKER_MODEL_PATH}",
    )
    .run_function(_prefetch_weights, volumes={HF_CACHE_DIR: hf_cache})
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=3600,
    scaledown_window=600,
)
class VibeVoiceASR:
    @modal.enter()
    def load(self) -> None:
        import numpy as np
        import sherpa_onnx
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            MODEL_NAME,
            language_model_pretrained_name=QWEN_TOKENIZER,
        )
        self.model = (
            VibeVoiceASRForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            .to("cuda")
            .eval()
        )
        self.vad = _SileroVAD()

        # Speaker x-vector extractor: WeSpeaker CAM++ (VoxCeleb). Loaded via
        # sherpa-onnx (pure C++/onnxruntime, no torchaudio dep), includes
        # Kaldi-style fbank front-end. Language-agnostic in practice.
        self.spk_model = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=SPEAKER_MODEL_PATH,
                num_threads=2,
                debug=False,
                provider="cpu",
            )
        )

        # 1s of silence: triggers kernel autotune before the first real call
        self._generate([np.zeros(TARGET_SR, dtype=np.float32)], max_new_tokens=8)

    def _generate(
        self,
        audios: list,
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_info: str | None = None,
    ) -> list[dict]:
        import torch

        proc_kwargs = dict(
            audio=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        if context_info:
            proc_kwargs["context_info"] = context_info
        inputs = self.processor(**proc_kwargs)
        inputs = {
            k: (v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
        elif temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        eos_id = self.processor.tokenizer.eos_token_id
        results = []
        for i in range(len(audios)):
            gen = output_ids[i, input_length:]
            eos_positions = (gen == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen = gen[: eos_positions[0] + 1]
            text = self.processor.decode(gen, skip_special_tokens=True)
            try:
                segments = self.processor.post_process_transcription(text)
            except Exception:
                segments = []
            results.append({"text": text, "segments": segments})
        return results

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
        audio = _decode_audio(audio_bytes)
        return self._generate(
            [audio],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            context_info=context_info,
        )[0]

    @modal.method()
    def benchmark(
        self,
        audio_bytes: bytes,
        max_new_tokens: int = 32768,
        sample_hz: float = 2.0,
        context_info: str | None = None,
    ) -> dict:
        """Transcribe once while measuring timings, tokens/s, and GPU stats."""
        import threading
        import time as _time

        import numpy as np
        import pynvml
        import torch

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total

        t0 = _time.perf_counter()
        audio = _decode_audio(audio_bytes)
        decode_t = _time.perf_counter() - t0
        audio_duration = float(len(audio) / TARGET_SR)

        t0 = _time.perf_counter()
        proc_kwargs = dict(
            audio=[audio],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        if context_info:
            proc_kwargs["context_info"] = context_info
        inputs = self.processor(**proc_kwargs)
        inputs = {
            k: (v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        torch.cuda.synchronize()
        preprocess_t = _time.perf_counter() - t0
        input_len = int(inputs["input_ids"].shape[1])

        # Sample GPU util + mem in background during generate()
        stop = threading.Event()
        samples: list[tuple[float, float, int]] = []

        def sampler():
            interval = 1.0 / sample_hz
            while not stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                samples.append((util.gpu, util.memory, mem.used))
                stop.wait(interval)

        torch.cuda.reset_peak_memory_stats()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        t0 = _time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generate_t = _time.perf_counter() - t0

        stop.set()
        th.join(timeout=2)

        gen_ids = output_ids[0, input_len:]
        eos_id = self.processor.tokenizer.eos_token_id
        eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[: eos_positions[0] + 1]
        n_generated = int(gen_ids.shape[0])

        text = self.processor.decode(gen_ids, skip_special_tokens=True)
        try:
            segments = self.processor.post_process_transcription(text)
        except Exception:
            segments = []

        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        if samples:
            gpu_utils = [s[0] for s in samples]
            mem_utils = [s[1] for s in samples]
            mem_used = [s[2] for s in samples]
            gpu_stats = {
                "samples": len(samples),
                "gpu_util_mean": float(np.mean(gpu_utils)),
                "gpu_util_p50": float(np.percentile(gpu_utils, 50)),
                "gpu_util_p95": float(np.percentile(gpu_utils, 95)),
                "gpu_util_max": float(np.max(gpu_utils)),
                "mem_bw_util_mean": float(np.mean(mem_utils)),
                "mem_used_peak_mb": float(np.max(mem_used) / 1024**2),
            }
        else:
            gpu_stats = {}

        total_t = decode_t + preprocess_t + generate_t
        return {
            "gpu_name": gpu_name,
            "gpu_total_mem_mb": round(total_mem / 1024**2, 1),
            "audio_duration_s": round(audio_duration, 3),
            "decode_s": round(decode_t, 3),
            "preprocess_s": round(preprocess_t, 3),
            "generate_s": round(generate_t, 3),
            "total_s": round(total_t, 3),
            "rtf_generate": round(generate_t / audio_duration, 5),
            "rtf_total": round(total_t / audio_duration, 5),
            "speedup_vs_realtime": round(audio_duration / generate_t, 2),
            "input_tokens": input_len,
            "generated_tokens": n_generated,
            "tokens_per_sec": round(n_generated / generate_t, 2),
            "peak_alloc_mb": round(peak_alloc / 1024**2, 1),
            "peak_reserved_mb": round(peak_reserved / 1024**2, 1),
            "gpu": gpu_stats,
            "num_segments": len(segments),
            "text_chars": len(text),
            "text_preview": text[:400],
        }

    @modal.method()
    def transcribe_batch(
        self,
        audio_bytes_list: list[bytes],
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        context_info: str | None = None,
    ) -> list[dict]:
        audios = [_decode_audio(b) for b in audio_bytes_list]
        return self._generate(
            audios,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            context_info=context_info,
        )

    @modal.method()
    def transcribe_long(
        self,
        audio_bytes: bytes,
        context_info: str | None = None,
        chunk_target_s: float = 2700.0,  # 45 min
        chunk_max_s: float = 3300.0,  # 55 min hard cap (model handles 60)
        prime_with_prev_tail: bool = True,
        prev_tail_seconds: float = 30.0,
        max_new_tokens: int = 32768,
        batch_size: int = 0,
        unify_speakers: bool = True,
        unify_distance_threshold: float = 0.3,
    ) -> dict:
        """Transcribe arbitrarily long audio via VAD-aware chunking.

        `batch_size=0` (default) picks a value automatically from the live
        GPU's VRAM and the configured chunk size, using the empirically
        calibrated model:
            peak_vram ≈ 18 GB (weights) + 5 GB (headroom)
                       + 0.15 GB/(chunk·min) × chunk_minutes × batch.
        Pass any positive int to override.


        Pipeline:
          1. Decode → 24 kHz mono float32.
          2. silero-vad → speech regions, gaps become candidate cut points.
          3. Greedy chunker targets `chunk_target_s` and always cuts at a real
             silence (≥0.3 s gap) below `chunk_max_s`.
          4. Chunks are generated in micro-batches of `batch_size`:
             • `batch_size=1` → strict sequential, every chunk gets a primed
               `context_info` containing the previous chunk's last
               `prev_tail_seconds` of transcription.
             • `batch_size>1` → all chunks in a batch share the same priming
               (taken from the previous batch's last chunk). Faster, slightly
               weaker continuity at intra-batch boundaries.
          5. Per-chunk timestamps are offset; per-chunk speaker IDs are tagged
             with chunk index.
          6. If `unify_speakers`, run WavLMForXVector on the per-(chunk,
             speaker) audio to produce x-vectors, then agglomeratively cluster
             across chunks (cosine distance, average linkage,
             `distance_threshold=unify_distance_threshold`). Final
             `global_speaker_id` is `S0`, `S1`, ... unified across chunks.
        """
        import time as _time

        import pynvml
        import torch

        t_total = _time.perf_counter()

        torch.cuda.reset_peak_memory_stats()
        pynvml.nvmlInit()
        nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(nvml_h)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        gpu_total_mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_h).total

        t0 = _time.perf_counter()
        audio = _decode_audio(audio_bytes)
        decode_t = _time.perf_counter() - t0
        audio_duration = float(len(audio) / TARGET_SR)

        # Short-circuit single-pass case
        if audio_duration <= chunk_max_s:
            res = self._generate(
                [audio],
                max_new_tokens=max_new_tokens,
                context_info=context_info,
            )[0]
            shifted = _shift_segments(res["segments"], 0.0, 0)
            unify_info = {}
            if unify_speakers:
                unify_info = self._unify_speakers(
                    audio, TARGET_SR, shifted, unify_distance_threshold
                )
            peak_alloc = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            return {
                "gpu_name": gpu_name,
                "gpu_total_mem_mb": round(gpu_total_mem / 1024**2, 1),
                "peak_alloc_mb": round(peak_alloc / 1024**2, 1),
                "peak_reserved_mb": round(peak_reserved / 1024**2, 1),
                "audio_duration_s": round(audio_duration, 3),
                "decode_s": round(decode_t, 3),
                "vad_s": 0.0,
                "generate_s": round(_time.perf_counter() - t_total - decode_t, 3),
                "unify_s": round(unify_info.get("elapsed_s", 0.0), 3),
                "total_s": round(_time.perf_counter() - t_total, 3),
                "num_chunks": 1,
                "batch_size": 1,
                "chunks": [
                    {
                        "index": 0,
                        "start_s": 0.0,
                        "end_s": round(audio_duration, 3),
                        "duration_s": round(audio_duration, 3),
                        "num_segments": len(res["segments"]),
                    }
                ],
                "segments": shifted,
                "text": res["text"],
                "unify": unify_info,
            }

        t0 = _time.perf_counter()
        vad_ranges = _vad_speech_ranges(audio, TARGET_SR, self.vad)
        vad_t = _time.perf_counter() - t0

        chunk_ranges = _chunk_by_vad(
            len(audio),
            TARGET_SR,
            vad_ranges,
            target_s=chunk_target_s,
            max_s=chunk_max_s,
        )

        all_segments: list[dict] = []
        chunk_summaries: list[dict] = []
        text_parts: list[str] = []
        prev_tail = ""
        gen_total = 0.0

        # pynvml sampler: collect GPU util + memory BW util across all
        # generate() calls so the final report covers the full chunked run.
        import threading

        util_samples: list[tuple[int, int]] = []
        util_stop = threading.Event()

        def _sampler():
            while not util_stop.is_set():
                u = pynvml.nvmlDeviceGetUtilizationRates(nvml_h)
                util_samples.append((u.gpu, u.memory))
                util_stop.wait(0.5)

        sampler_th = threading.Thread(target=_sampler, daemon=True)
        sampler_th.start()

        # Process in micro-batches of `batch_size` (auto-pick if ≤ 0)
        if batch_size and batch_size > 0:
            bs = int(batch_size)
            print(f"batch_size={bs} (user-set)")
        else:
            bs = _auto_batch_size(
                gpu_total_mem_bytes=gpu_total_mem,
                chunk_minutes=chunk_target_s / 60.0,
            )
            print(
                f"batch_size={bs} (auto from "
                f"{gpu_total_mem / 1024**3:.0f} GB VRAM, "
                f"{chunk_target_s / 60:.0f} min chunks)"
            )
        for batch_start in range(0, len(chunk_ranges), bs):
            batch = chunk_ranges[batch_start : batch_start + bs]
            batch_audios = [audio[s:e] for (s, e) in batch]

            # Build context_info for this whole batch
            ctx_parts: list[str] = []
            if context_info:
                ctx_parts.append(context_info.strip())
            if prime_with_prev_tail and prev_tail:
                ctx_parts.append(f"Continued from previous segment: {prev_tail}")
            ctx = " | ".join(ctx_parts) if ctx_parts else None

            t0 = _time.perf_counter()
            results = self._generate(
                batch_audios,
                max_new_tokens=max_new_tokens,
                context_info=ctx,
            )
            batch_gen_dt = _time.perf_counter() - t0
            gen_total += batch_gen_dt

            for offset_in_batch, (res, (s, e)) in enumerate(zip(results, batch)):
                ci = batch_start + offset_in_batch
                chunk_offset = s / TARGET_SR
                chunk_dur = (e - s) / TARGET_SR
                shifted = _shift_segments(res["segments"], chunk_offset, ci)
                all_segments.extend(shifted)
                text_parts.append(res["text"])
                chunk_summaries.append(
                    {
                        "index": ci,
                        "start_s": round(chunk_offset, 3),
                        "end_s": round(chunk_offset + chunk_dur, 3),
                        "duration_s": round(chunk_dur, 3),
                        "generate_s": round(batch_gen_dt / len(batch), 3),
                        "batch_index": batch_start // bs,
                        "num_segments": len(res["segments"]),
                        "context_info_used": ctx,
                    }
                )

            # Update prev_tail from the LAST chunk's transcription in this batch
            if prime_with_prev_tail:
                prev_tail = _tail_text(results[-1]["segments"], prev_tail_seconds)

        util_stop.set()
        sampler_th.join(timeout=2)

        unify_info: dict = {}
        if unify_speakers:
            unify_info = self._unify_speakers(
                audio, TARGET_SR, all_segments, unify_distance_threshold
            )

        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        nvml_used_after = pynvml.nvmlDeviceGetMemoryInfo(nvml_h).used

        import numpy as _np

        if util_samples:
            sm = [s[0] for s in util_samples]
            mb = [s[1] for s in util_samples]
            gpu_util = {
                "samples": len(util_samples),
                "sm_util_mean": float(_np.mean(sm)),
                "sm_util_p50": float(_np.percentile(sm, 50)),
                "sm_util_p95": float(_np.percentile(sm, 95)),
                "sm_util_max": float(_np.max(sm)),
                "mem_bw_util_mean": float(_np.mean(mb)),
                "mem_bw_util_p95": float(_np.percentile(mb, 95)),
            }
        else:
            gpu_util = {}

        return {
            "gpu_name": gpu_name,
            "gpu_total_mem_mb": round(gpu_total_mem / 1024**2, 1),
            "peak_alloc_mb": round(peak_alloc / 1024**2, 1),
            "peak_reserved_mb": round(peak_reserved / 1024**2, 1),
            "nvml_mem_used_after_mb": round(nvml_used_after / 1024**2, 1),
            "gpu_util": gpu_util,
            "audio_duration_s": round(audio_duration, 3),
            "decode_s": round(decode_t, 3),
            "vad_s": round(vad_t, 3),
            "generate_s": round(gen_total, 3),
            "unify_s": round(unify_info.get("elapsed_s", 0.0), 3),
            "total_s": round(_time.perf_counter() - t_total, 3),
            "num_chunks": len(chunk_ranges),
            "batch_size": bs,
            "chunks": chunk_summaries,
            "segments": all_segments,
            "text": "\n\n".join(text_parts),
            "unify": unify_info,
        }

    def _embed_speaker(self, audio_16k):
        """Run CAM++ on one mono 16 kHz waveform → 192-d x-vector np.ndarray."""
        import numpy as np

        stream = self.spk_model.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=audio_16k.astype(np.float32))
        stream.input_finished()
        emb = self.spk_model.compute(stream)
        return np.asarray(emb, dtype=np.float32)

    def _unify_speakers(
        self,
        audio,
        sr: int,
        segments: list[dict],
        distance_threshold: float = 0.3,
        min_audio_per_speaker_s: float = 1.0,
        max_audio_per_speaker_s: float = 30.0,
    ) -> dict:
        """Cluster (chunk_id, local_speaker_id) keys across chunks via WavLM x-vectors.

        Mutates `segments` in place: rewrites `global_speaker_id` to `S{k}` for
        the global cluster id, or leaves the per-chunk fallback when there's
        not enough audio for a reliable embedding.
        """
        import time as _time

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
        n_audio_s = sr  # samples per second
        min_samples = int(min_audio_per_speaker_s * n_audio_s)
        max_samples = int(max_audio_per_speaker_s * n_audio_s)

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
            embeddings.append(self._embed_speaker(joined_16k))
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
            # else: leave the c{ck}_s{sp} fallback set by _shift_segments

        # Per-cluster size for diagnostics
        from collections import Counter

        sizes = Counter(int(l) for l in labels)
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
            arr = _decode_audio(data)
            return self._generate(
                [arr],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                context_info=context_info,
            )[0]

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
        ):
            data = await audio.read()
            if not data:
                raise HTTPException(400, "empty upload")
            return self.transcribe_long.local(
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
            )

        return api


# Empirically calibrated VRAM model; see scripts/plot_vram.py.
_WEIGHTS_GB = 18.0
_HEADROOM_GB = 5.0
_KV_GB_PER_CHUNK_MIN = 0.22


def _auto_batch_size(
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


def _decode_audio(audio_bytes: bytes):
    """Decode any container to mono float32 at TARGET_SR via librosa/ffmpeg."""
    import io

    import librosa
    import numpy as np

    arr, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    return arr.astype(np.float32)


SILERO_VAD_PATH = "/opt/silero_vad.onnx"
_VAD_WINDOW = 512  # 32 ms at 16 kHz, silero v5 native window size


class _SileroVAD:
    """Silero VAD v5 run directly through onnxruntime (no torchaudio dep)."""

    def __init__(self, model_path: str = SILERO_VAD_PATH) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def speech_probs(self, audio_16k) -> "np.ndarray":  # type: ignore[name-defined]
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


def _vad_speech_ranges(
    audio,
    sr: int,
    vad: "_SileroVAD",
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


def _chunk_by_vad(
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


def _shift_segments(segments: list[dict], offset_s: float, chunk_id: int) -> list[dict]:
    """Apply chunk offset to timestamps and tag global speaker id."""
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


def _tail_text(segments: list[dict], tail_seconds: float, max_chars: int = 600) -> str:
    """Return concatenated text of trailing segments covering ~tail_seconds."""
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
    import json
    import time
    from pathlib import Path

    data = Path(audio_path).read_bytes()
    print(f"Uploading {len(data) / 1024**2:.1f} MB from {audio_path}")

    t0 = time.perf_counter()
    result = VibeVoiceASR().benchmark.remote(
        data, max_new_tokens=max_new_tokens, context_info=context_info
    )
    wall = time.perf_counter() - t0

    g = result.get("gpu", {})
    print("\n" + "=" * 72)
    print("VibeVoice-ASR Benchmark Report")
    print("=" * 72)
    print(
        f"GPU                 : {result['gpu_name']} ({result['gpu_total_mem_mb']:.0f} MiB)"
    )
    print(
        f"Audio duration      : {result['audio_duration_s']:.2f} s "
        f"({result['audio_duration_s'] / 60:.2f} min)"
    )
    print("-" * 72)
    print(f"Decode              : {result['decode_s']:.2f} s")
    print(f"Preprocess          : {result['preprocess_s']:.2f} s")
    print(f"Generate            : {result['generate_s']:.2f} s")
    print(f"Total (server-side) : {result['total_s']:.2f} s")
    print(f"Wall (incl. upload) : {wall:.2f} s")
    print("-" * 72)
    print(f"RTF (generate only) : {result['rtf_generate']:.5f}")
    print(f"RTF (end-to-end)    : {result['rtf_total']:.5f}")
    print(f"Speedup vs realtime : {result['speedup_vs_realtime']:.2f}x")
    print("-" * 72)
    print(f"Input tokens        : {result['input_tokens']}")
    print(f"Generated tokens    : {result['generated_tokens']}")
    print(f"Tokens / sec        : {result['tokens_per_sec']:.2f}")
    print("-" * 72)
    print(f"Peak alloc (torch)  : {result['peak_alloc_mb']:.0f} MiB")
    print(f"Peak reserved       : {result['peak_reserved_mb']:.0f} MiB")
    if g:
        print(
            f"GPU util  mean/p50/p95/max : "
            f"{g['gpu_util_mean']:.1f}% / {g['gpu_util_p50']:.1f}% / "
            f"{g['gpu_util_p95']:.1f}% / {g['gpu_util_max']:.1f}%"
        )
        print(f"Mem-BW util (mean)  : {g['mem_bw_util_mean']:.1f}%")
        print(
            f"Peak NVML mem used  : {g['mem_used_peak_mb']:.0f} MiB "
            f"({g['mem_used_peak_mb'] / result['gpu_total_mem_mb'] * 100:.1f}% of total)"
        )
        print(f"Samples collected   : {g['samples']}")
    print("-" * 72)
    print(f"Segments            : {result['num_segments']}")
    print(f"Output chars        : {result['text_chars']}")
    print("=" * 72)
    print("\nPreview:\n" + result["text_preview"])
    print("\nJSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


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
    out_json: str | None = None,
):
    """Long-form (multi-hour) transcription via VAD-aware chunking."""
    import json
    import time
    from pathlib import Path

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
    )
    wall = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("VibeVoice-ASR Long-Form Report")
    print("=" * 72)
    print(
        f"GPU             : {result.get('gpu_name', '?')} "
        f"({result.get('gpu_total_mem_mb', 0):.0f} MiB)"
    )
    print(
        f"Peak alloc      : {result.get('peak_alloc_mb', 0):.0f} MiB "
        f"(reserved {result.get('peak_reserved_mb', 0):.0f} MiB)"
    )
    print(
        f"Audio duration  : {result['audio_duration_s']:.1f}s "
        f"({result['audio_duration_s'] / 60:.1f} min, "
        f"{result['audio_duration_s'] / 3600:.2f} h)"
    )
    print(f"Decode          : {result['decode_s']:.2f}s")
    print(f"VAD             : {result['vad_s']:.2f}s")
    print(f"Generate (sum)  : {result['generate_s']:.2f}s")
    print(f"Unify spk       : {result['unify_s']:.2f}s")
    print(f"Server total    : {result['total_s']:.2f}s")
    print(f"Wall            : {wall:.2f}s")
    print(f"RTF (e2e)       : {result['total_s'] / result['audio_duration_s']:.4f}")
    print(f"Speedup         : {result['audio_duration_s'] / result['total_s']:.2f}x")
    print(
        f"Chunks          : {result['num_chunks']} (batch_size={result['batch_size']})"
    )
    for c in result["chunks"]:
        print(
            f"  [{c['index']:>2}] {c['start_s'] / 60:>6.1f}–{c['end_s'] / 60:>6.1f} min "
            f"(dur {c['duration_s']:>5.0f}s, gen {c.get('generate_s', 0):>5.1f}s, "
            f"{c['num_segments']} segs)"
        )
    print(f"Total segments  : {len(result['segments'])}")
    print(f"Output chars    : {len(result['text'])}")
    gu = result.get("gpu_util") or {}
    if gu:
        print(
            f"SM util mean/p50/p95/max : "
            f"{gu['sm_util_mean']:.1f}% / {gu['sm_util_p50']:.1f}% / "
            f"{gu['sm_util_p95']:.1f}% / {gu['sm_util_max']:.1f}%"
        )
        print(
            f"Mem-BW util mean/p95     : "
            f"{gu['mem_bw_util_mean']:.1f}% / {gu['mem_bw_util_p95']:.1f}% "
            f"(n={gu['samples']})"
        )
    u = result.get("unify") or {}
    if u:
        print(
            f"Global speakers : {u.get('num_global_speakers', '?')} "
            f"(embedded {u.get('num_keys_embedded', 0)}, "
            f"skipped {u.get('num_skipped', 0)})"
        )
        if u.get("cluster_sizes"):
            sizes = ", ".join(f"{k}:{v}" for k, v in u["cluster_sizes"].items())
            print(f"  cluster sizes : {sizes}")
        if u.get("mapping"):
            print("  mapping (first 20):")
            for m in u["mapping"][:20]:
                print(
                    f"    chunk {m['chunk_id']:>2} local s{m['local_speaker_id']} "
                    f"→ {m['global_speaker_id']}"
                )
        if u.get("distance_matrix"):
            keys = u.get("keys", [])
            mat = u["distance_matrix"]
            labels = [f"c{k['chunk_id']}s{k['local_speaker_id']}" for k in keys]
            print(
                f"  pairwise cosine distance (threshold={u.get('distance_threshold')}):"
            )
            print("       " + "  ".join(f"{l:>5}" for l in labels))
            for i, row in enumerate(mat):
                print(f"  {labels[i]:>4} " + "  ".join(f"{v:>5.2f}" for v in row))
    print("=" * 72)

    print("\n--- First 30 segments ---")
    for seg in result["segments"][:30]:
        spk = seg.get("global_speaker_id", seg.get("speaker_id", "?"))
        print(
            f"[{seg.get('start_time', '?'):>8} - {seg.get('end_time', '?'):>8}] "
            f"{spk}: {seg.get('text', '')}"
        )
    if len(result["segments"]) > 30:
        print(f"... and {len(result['segments']) - 30} more")

    if out_json:
        Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nFull result written to {out_json}")
