"""Pretty-print formatters for the benchmark and long-form CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path


def print_bench_report(result: dict, wall: float) -> None:
    g = result.get("gpu", {})
    print("\n" + "=" * 72)
    print("VibeVoice-ASR Benchmark Report")
    print("=" * 72)
    print(
        f"GPU                 : {result['gpu_name']} "
        f"({result['gpu_total_mem_mb']:.0f} MiB)"
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


def print_long_report(result: dict, wall: float, out_json: str | None = None) -> None:
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
            print("       " + "  ".join(f"{label:>5}" for label in labels))
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
