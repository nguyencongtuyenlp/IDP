#!/usr/bin/env python3
"""
Benchmark Script — Measure OCR speed, RAM, and quality.

Compares GPU vs CPU performance across quality modes.

Usage:
    python benchmark.py --input data/input/
    python benchmark.py --input test.jpg --device cpu
    python benchmark.py --compare   # GPU vs CPU comparison
"""

import argparse
import json
import sys
import time
from pathlib import Path

import psutil

from src.utils import setup_logging, get_logger, format_time, format_size, get_image_files
from src.ocr_engine import OCREngine

logger = get_logger(__name__)


def benchmark_single(engine: OCREngine, image_path: str) -> dict:
    """Benchmark a single image."""
    mem_before = psutil.virtual_memory().used

    start = time.perf_counter()
    results = engine.process(image_path)
    elapsed = time.perf_counter() - start

    mem_after = psutil.virtual_memory().used
    mem_delta = (mem_after - mem_before) / 1024 / 1024

    return {
        "file": Path(image_path).name,
        "time_seconds": round(elapsed, 3),
        "regions": len(results),
        "avg_confidence": round(
            sum(r.confidence for r in results) / max(len(results), 1), 4
        ),
        "ram_delta_mb": round(mem_delta, 1),
    }


def run_benchmark(
    input_path: str,
    device: str = "auto",
    mode: str = "balanced",
    runs: int = 1,
) -> dict:
    """Run benchmark on input files."""
    path = Path(input_path)

    if path.is_file():
        files = [path]
    else:
        files = get_image_files(path)

    if not files:
        print(f"❌ No images found in: {input_path}")
        sys.exit(1)

    print(f"\n🏁 Benchmark: {len(files)} file(s) | device={device} | mode={mode} | runs={runs}")
    print("=" * 70)

    engine = OCREngine(device=device, mode=mode)

    all_results = []
    for run in range(1, runs + 1):
        if runs > 1:
            print(f"\n--- Run {run}/{runs} ---")

        for f in files:
            result = benchmark_single(engine, str(f))
            all_results.append(result)
            print(
                f"  ✅ {result['file']:30s} | "
                f"{format_time(result['time_seconds']):>6s} | "
                f"{result['regions']:3d} regions | "
                f"conf={result['avg_confidence']:.0%} | "
                f"RAM Δ={result['ram_delta_mb']:+.0f}MB"
            )

    engine.cleanup()

    # Summary
    avg_time = sum(r["time_seconds"] for r in all_results) / len(all_results)
    total_regions = sum(r["regions"] for r in all_results)
    avg_conf = sum(r["avg_confidence"] for r in all_results) / len(all_results)

    summary = {
        "device": engine.device_mgr.device_info.device,
        "gpu_name": engine.device_mgr.device_info.gpu_name,
        "mode": mode,
        "files": len(files),
        "runs": runs,
        "avg_time_per_image": round(avg_time, 3),
        "total_regions": total_regions,
        "avg_confidence": round(avg_conf, 4),
        "details": all_results,
    }

    print("\n" + "=" * 70)
    print(f"📊 SUMMARY")
    print(f"   Device: {summary['device'].upper()}" +
          (f" ({summary['gpu_name']})" if summary['gpu_name'] else ""))
    print(f"   Mode: {mode}")
    print(f"   Avg time/image: {format_time(avg_time)}")
    print(f"   Total regions: {total_regions}")
    print(f"   Avg confidence: {avg_conf:.0%}")

    return summary


def compare_devices(input_path: str) -> None:
    """Compare GPU vs CPU performance."""
    print("\n🔥 GPU vs CPU Comparison")
    print("=" * 70)

    results = {}

    for device in ["cuda", "cpu"]:
        try:
            print(f"\n--- Testing {device.upper()} ---")
            results[device] = run_benchmark(input_path, device=device, mode="balanced")
        except Exception as e:
            print(f"⚠️  {device.upper()} failed: {e}")
            results[device] = None

    # Comparison table
    if all(results.values()):
        gpu_time = results["cuda"]["avg_time_per_image"]
        cpu_time = results["cpu"]["avg_time_per_image"]
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print("\n" + "=" * 70)
        print("📊 COMPARISON")
        print(f"{'Metric':<25s} | {'GPU':>10s} | {'CPU':>10s} | {'Speedup':>10s}")
        print("-" * 60)
        print(f"{'Avg time/image':<25s} | {format_time(gpu_time):>10s} | {format_time(cpu_time):>10s} | {speedup:.1f}x")
        print(f"{'Regions (total)':<25s} | {results['cuda']['total_regions']:>10d} | {results['cpu']['total_regions']:>10d} | -")
        print(f"{'Avg confidence':<25s} | {results['cuda']['avg_confidence']:>9.0%} | {results['cpu']['avg_confidence']:>9.0%} | -")


def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark")
    parser.add_argument("--input", "-i", default="data/input", help="Image file or directory")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--mode", "-m", default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument("--compare", action="store_true", help="Compare GPU vs CPU")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()
    setup_logging("WARNING")

    if args.compare:
        compare_devices(args.input)
    else:
        results = run_benchmark(args.input, args.device, args.mode, args.runs)
        if args.output:
            Path(args.output).write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n💾 Results saved: {args.output}")


if __name__ == "__main__":
    main()
