"""Provider-agnostic VibeVoice-ASR runner with a three-stage lifecycle."""

from __future__ import annotations

from .audio import decode_audio
from .batching import auto_batch_size
from .config import RunnerConfig
from .postproc import shift_segments, tail_text
from .speakers import SpeakerEmbedder
from .speakers import unify_speakers as _unify_speakers_fn
from .vad import SileroVAD, chunk_by_vad, vad_speech_ranges


class VibeVoiceASRRunner:
    """Three-stage ASR runner; wraps Microsoft VibeVoice-ASR + VAD + speaker unify.

    Lifecycle (mirrors Modal's image→enter→request split so any provider —
    Modal, RunPod, Docker, local — can amortize work the same way):

      Stage 1 — BUILD  : `VibeVoiceASRRunner.prefetch_weights(cfg)`
                         Downloads HF snapshot into `cfg.hf_cache_dir`.
                         Call at image build time. ONNX assets (silero
                         VAD + CAM++) are the provider's responsibility —
                         `curl` during image build is usually fastest and
                         gets layer-cached.
      Stage 2 — ENTER  : `runner = VibeVoiceASRRunner(cfg); runner.load()`
                         Builds processor/model on GPU, constructs VAD +
                         speaker embedder, and runs a 1-second warmup so
                         the first real request skips CUDA autotune.
      Stage 3 — REQUEST: `runner.transcribe(bytes)` / `transcribe_batch`
                         / `transcribe_long` / `benchmark`.
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()
        self.model = None
        self.processor = None
        self.vad: SileroVAD | None = None
        self.spk_embedder: SpeakerEmbedder | None = None

    # ------------------------------------------------------------ build

    @classmethod
    def prefetch_weights(cls, config: RunnerConfig | None = None) -> None:
        """Snapshot_download HF repos into `config.hf_cache_dir`."""
        from huggingface_hub import snapshot_download

        cfg = config or RunnerConfig()
        snapshot_download(cfg.model_name, cache_dir=cfg.hf_cache_dir)
        snapshot_download(
            cfg.tokenizer_name,
            cache_dir=cfg.hf_cache_dir,
            allow_patterns=[
                "tokenizer*",
                "*.json",
                "merges.txt",
                "vocab.txt",
                "special_tokens_map.json",
            ],
        )

    # ------------------------------------------------------------ enter

    def load(self) -> None:
        """Build processor/model/VAD/speaker and warm up CUDA kernels."""
        import numpy as np
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.config.model_name,
            language_model_pretrained_name=self.config.tokenizer_name,
        )
        # `device_map="cuda"` works for both full-precision and bnb-quantized
        # checkpoints. bnb 4-bit models self-describe via `quantization_config`
        # in config.json, so transformers auto-applies it — no explicit
        # BitsAndBytesConfig needed here. Calling `.to("cuda")` on a bnb model
        # would raise, so we rely on accelerate's dispatch instead.
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        self.vad = SileroVAD(self.config.silero_vad_path)
        self.spk_embedder = SpeakerEmbedder(self.config.speaker_model_path)

        # 1s of silence: triggers kernel autotune before the first real call
        self.generate(
            [np.zeros(self.config.target_sr, dtype=np.float32)], max_new_tokens=8
        )

    # ----------------------------------------------------------- request

    def generate(
        self,
        audios: list,
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_info: str | None = None,
    ) -> list[dict]:
        """Low-level batched generation; callers pass already-decoded float32 arrays."""
        import torch

        proc_kwargs = dict(
            audio=audios,
            sampling_rate=self.config.target_sr,
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

    def transcribe(
        self,
        audio_bytes: bytes,
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_info: str | None = None,
    ) -> dict:
        audio = decode_audio(audio_bytes, self.config.target_sr)
        return self.generate(
            [audio],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            context_info=context_info,
        )[0]

    def transcribe_batch(
        self,
        audio_bytes_list: list[bytes],
        max_new_tokens: int = 32768,
        num_beams: int = 1,
        context_info: str | None = None,
    ) -> list[dict]:
        audios = [decode_audio(b, self.config.target_sr) for b in audio_bytes_list]
        return self.generate(
            audios,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            context_info=context_info,
        )

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

        sr = self.config.target_sr

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total

        t0 = _time.perf_counter()
        audio = decode_audio(audio_bytes, sr)
        decode_t = _time.perf_counter() - t0
        audio_duration = float(len(audio) / sr)

        t0 = _time.perf_counter()
        proc_kwargs = dict(
            audio=[audio],
            sampling_rate=sr,
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
                       + 0.22 GB/(chunk·min) × chunk_minutes × batch.
        Pass any positive int to override.

        Pipeline:
          1. Decode → 24 kHz mono float32.
          2. silero-vad → speech regions, gaps become candidate cut points.
          3. Greedy chunker targets `chunk_target_s` and always cuts at a
             real silence (≥0.3 s gap) below `chunk_max_s`.
          4. Chunks are generated in micro-batches of `batch_size`:
             • `batch_size=1` → strict sequential, every chunk gets a
               primed `context_info` containing the previous chunk's last
               `prev_tail_seconds` of transcription.
             • `batch_size>1` → all chunks in a batch share the same
               priming (taken from the previous batch's last chunk).
               Faster, slightly weaker continuity at intra-batch
               boundaries.
          5. Per-chunk timestamps are offset; per-chunk speaker IDs are
             tagged with chunk index.
          6. If `unify_speakers`, run CAM++ on the per-(chunk, speaker)
             audio to produce x-vectors, then agglomeratively cluster
             across chunks (cosine distance, average linkage,
             `distance_threshold=unify_distance_threshold`). Final
             `global_speaker_id` is `S0`, `S1`, ... unified across chunks.
        """
        import threading
        import time as _time

        import numpy as _np
        import pynvml
        import torch

        sr = self.config.target_sr
        t_total = _time.perf_counter()

        torch.cuda.reset_peak_memory_stats()
        pynvml.nvmlInit()
        nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(nvml_h)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        gpu_total_mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_h).total

        t0 = _time.perf_counter()
        audio = decode_audio(audio_bytes, sr)
        decode_t = _time.perf_counter() - t0
        audio_duration = float(len(audio) / sr)

        # Short-circuit single-pass case
        if audio_duration <= chunk_max_s:
            res = self.generate(
                [audio],
                max_new_tokens=max_new_tokens,
                context_info=context_info,
            )[0]
            shifted = shift_segments(res["segments"], 0.0, 0)
            unify_info: dict = {}
            if unify_speakers:
                unify_info = _unify_speakers_fn(
                    audio, sr, shifted, self.spk_embedder, unify_distance_threshold
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
        vad_ranges = vad_speech_ranges(audio, sr, self.vad)
        vad_t = _time.perf_counter() - t0

        chunk_ranges = chunk_by_vad(
            len(audio),
            sr,
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
            bs = auto_batch_size(
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
            results = self.generate(
                batch_audios,
                max_new_tokens=max_new_tokens,
                context_info=ctx,
            )
            batch_gen_dt = _time.perf_counter() - t0
            gen_total += batch_gen_dt

            for offset_in_batch, (res, (s, e)) in enumerate(zip(results, batch)):
                ci = batch_start + offset_in_batch
                chunk_offset = s / sr
                chunk_dur = (e - s) / sr
                shifted = shift_segments(res["segments"], chunk_offset, ci)
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
                prev_tail = tail_text(results[-1]["segments"], prev_tail_seconds)

        util_stop.set()
        sampler_th.join(timeout=2)

        unify_info = {}
        if unify_speakers:
            unify_info = _unify_speakers_fn(
                audio, sr, all_segments, self.spk_embedder, unify_distance_threshold
            )

        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        nvml_used_after = pynvml.nvmlDeviceGetMemoryInfo(nvml_h).used

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
