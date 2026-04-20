"""Multi-GPU benchmark harness for VibeVoice-ASR on Modal.

Runs `modal run app.py::long` on each listed GPU, collects the JSON output
(timing + GPU util + peak VRAM + speaker unification stats), then writes:

  benchmarks/<timestamp>/
    <GPU>.json         per-GPU raw result (as returned by transcribe_long)
    <GPU>.log          full stdout/stderr from that run
    summary.json       combined results + run parameters
    report.md          markdown comparison table

Usage:
    uv run python scripts/benchmark.py test.mp3

    uv run python scripts/benchmark.py test.mp3 \\
        --gpus RTX-PRO-6000 H100 A100-80GB \\
        --chunk-target-min 45 --chunk-max-min 55 \\
        --context-info "Domain: ...; Speakers: ..." \\
        --parallel
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Modal published pricing (USD per GPU-hour), as of 2026-04. Verify on
# https://modal.com/pricing and update when rates change.
PRICING_USD_PER_HOUR: dict[str, float] = {
    "RTX-PRO-6000": 3.03,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 1.95,
    "A10G": 1.10,
    "L4": 0.80,
    "T4": 0.59,
}

DEFAULT_GPUS = ["RTX-PRO-6000", "H100", "A100-80GB", "A100-40GB"]


def run_on_gpu(
    gpu: str,
    audio_path: Path,
    out_dir: Path,
    chunk_target_min: float,
    chunk_max_min: float,
    context_info: str | None,
    batch_size: int,
    unify_distance_threshold: float,
    repo_root: Path,
) -> dict:
    """Run `modal run app.py::long` on one GPU; return its parsed result dict."""
    out_json = out_dir / f"{gpu}.json"
    log_path = out_dir / f"{gpu}.log"

    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "app.py::long",
        "--audio-path",
        str(audio_path),
        "--chunk-target-min",
        str(chunk_target_min),
        "--chunk-max-min",
        str(chunk_max_min),
        "--batch-size",
        str(batch_size),
        "--unify-distance-threshold",
        str(unify_distance_threshold),
        "--out-json",
        str(out_json),
    ]
    if context_info:
        cmd.extend(["--context-info", context_info])

    env = os.environ.copy()
    env["MODAL_GPU"] = gpu

    print(f"[{gpu}] launching (log: {log_path})")
    t0 = time.perf_counter()
    with open(log_path, "wb") as fh:
        proc = subprocess.run(
            cmd, cwd=repo_root, env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    wall = time.perf_counter() - t0

    if proc.returncode != 0 or not out_json.exists():
        print(f"[{gpu}] FAILED (exit {proc.returncode}) — see {log_path}")
        return {
            "modal_gpu": gpu,
            "error": f"exit {proc.returncode}",
            "log_path": str(log_path),
            "client_wall_s": round(wall, 2),
        }

    data = json.loads(out_json.read_text())
    # Drop the giant 'segments' list from the combined summary, keep it in
    # the per-GPU JSON for those who want it.
    data.pop("segments", None)
    data.pop("text", None)
    data["modal_gpu"] = gpu
    data["client_wall_s"] = round(wall, 2)
    rtf = data["total_s"] / data["audio_duration_s"]
    print(
        f"[{gpu}] OK — RTF {rtf:.4f}, server {data['total_s']:.0f}s, wall {wall:.0f}s"
    )
    return data


def _fmt_row(d: dict) -> str:
    """Pretty one-line row for the console summary."""
    if "error" in d:
        return f"{d['modal_gpu']:<28} ERROR: {d['error']}"
    gpu_short = d["modal_gpu"]
    vram_gb = d["gpu_total_mem_mb"] / 1024
    batch = d["batch_size"]
    peak_gb = d["peak_alloc_mb"] / 1024
    gen = d["generate_s"]
    wall = d["client_wall_s"]
    rtf = d["total_s"] / d["audio_duration_s"]
    xrt = d["audio_duration_s"] / d["total_s"]
    gu = d.get("gpu_util") or {}
    sm = gu.get("sm_util_mean", 0)
    mbw = gu.get("mem_bw_util_mean", 0)
    price = PRICING_USD_PER_HOUR.get(gpu_short, 0.0)
    cost = price * wall / 3600
    return (
        f"{gpu_short:<28} {vram_gb:>3.0f}GB  b{batch:<2d}  peak {peak_gb:>4.1f}GB  "
        f"gen {gen:>6.1f}s  wall {wall:>6.1f}s  "
        f"RTF {rtf:>6.4f}  {xrt:>5.1f}x  "
        f"SM {sm:>4.1f}%  memBW {mbw:>4.1f}%  "
        f"${price:>4.2f}/h  ${cost:>5.3f}/job"
    )


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 140)
    print("VibeVoice-ASR multi-GPU benchmark")
    print("=" * 140)
    for d in results:
        print(_fmt_row(d))
    print("=" * 140)


def write_markdown_report(results: list[dict], out_path: Path, run_meta: dict) -> None:
    lines = [
        "# VibeVoice-ASR Multi-GPU Benchmark",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Run parameters",
        "",
        f"- Audio: `{run_meta['audio_path']}` ({run_meta.get('audio_duration_s', '?')} s)",
        f"- Chunk target: {run_meta['chunk_target_min']} min (max {run_meta['chunk_max_min']} min)",
        f"- Batch size: `{run_meta['batch_size']}` (0 = auto per GPU)",
        f"- Unify threshold: {run_meta['unify_distance_threshold']}",
        f"- Context info: `{run_meta.get('context_info') or '(none)'}`",
        "",
        "## Results",
        "",
        "| GPU | VRAM | batch | Peak alloc | Generate | Wall | RTF (e2e) | ×RT | SM% | MemBW% | $/h | $/job |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in results:
        if "error" in d:
            lines.append(
                f"| {d['modal_gpu']} | ERROR: {d['error']} |  |  |  |  |  |  |  |  |  |  |"
            )
            continue
        gpu_short = d["modal_gpu"]
        vram_gb = d["gpu_total_mem_mb"] / 1024
        gu = d.get("gpu_util") or {}
        price = PRICING_USD_PER_HOUR.get(gpu_short, 0.0)
        cost = price * d["client_wall_s"] / 3600
        rtf = d["total_s"] / d["audio_duration_s"]
        xrt = d["audio_duration_s"] / d["total_s"]
        lines.append(
            f"| {gpu_short} | {vram_gb:.0f} GB | {d['batch_size']} | "
            f"{d['peak_alloc_mb'] / 1024:.1f} GB | {d['generate_s']:.1f} s | "
            f"{d['client_wall_s']:.1f} s | {rtf:.4f} | {xrt:.1f}× | "
            f"{gu.get('sm_util_mean', 0):.1f}% | {gu.get('mem_bw_util_mean', 0):.1f}% | "
            f"${price:.2f} | ${cost:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-GPU Modal benchmark for VibeVoice-ASR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", type=Path, help="Audio file to transcribe")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=DEFAULT_GPUS,
        help="Modal GPU identifiers (as accepted by MODAL_GPU)",
    )
    parser.add_argument("--chunk-target-min", type=float, default=20.0)
    parser.add_argument("--chunk-max-min", type=float, default=30.0)
    parser.add_argument(
        "--context-info",
        default=None,
        help="Optional hotword / speaker / domain hint for VibeVoice",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = auto-pick from live VRAM; positive int to override",
    )
    parser.add_argument("--unify-distance-threshold", type=float, default=0.3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: benchmarks/<timestamp>)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all GPUs in parallel (default: sequential). Much faster "
        "for a 4-way sweep; uses one subprocess per GPU.",
    )
    args = parser.parse_args()

    audio_path: Path = args.audio_path.resolve()
    if not audio_path.exists():
        print(f"error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = (args.out_dir or Path("benchmarks") / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"GPUs: {', '.join(args.gpus)}")
    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")

    repo_root = Path(__file__).resolve().parent.parent

    call_kwargs = dict(
        audio_path=audio_path,
        out_dir=out_dir,
        chunk_target_min=args.chunk_target_min,
        chunk_max_min=args.chunk_max_min,
        context_info=args.context_info,
        batch_size=args.batch_size,
        unify_distance_threshold=args.unify_distance_threshold,
        repo_root=repo_root,
    )

    results: list[dict] = []
    if args.parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
            futures = {
                ex.submit(run_on_gpu, gpu, **call_kwargs): gpu for gpu in args.gpus
            }
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
        order = {g: i for i, g in enumerate(args.gpus)}
        results.sort(key=lambda r: order.get(r.get("modal_gpu", ""), 999))
    else:
        for gpu in args.gpus:
            results.append(run_on_gpu(gpu, **call_kwargs))

    print_summary(results)

    run_meta = {
        "timestamp": ts,
        "audio_path": str(audio_path),
        "audio_duration_s": next(
            (r.get("audio_duration_s") for r in results if "audio_duration_s" in r),
            None,
        ),
        "chunk_target_min": args.chunk_target_min,
        "chunk_max_min": args.chunk_max_min,
        "batch_size": args.batch_size,
        "unify_distance_threshold": args.unify_distance_threshold,
        "context_info": args.context_info,
        "gpus": args.gpus,
        "mode": "parallel" if args.parallel else "sequential",
    }

    (out_dir / "summary.json").write_text(
        json.dumps({"run": run_meta, "results": results}, ensure_ascii=False, indent=2)
    )
    write_markdown_report(results, out_dir / "report.md", run_meta)

    # Non-zero exit if any GPU failed — easier to plug into CI / smoke tests.
    return 0 if all("error" not in r for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
