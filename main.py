#!/usr/bin/env python3
"""
CLI Entry Point — Offline Document OCR Extractor.

Usage:
    python main.py --input document.jpg
    python main.py --input folder/ --output results/ --device cuda --mode fast
    python main.py --input scan.pdf --output results/
"""

import argparse
import sys
import time
from pathlib import Path

from src.utils import setup_logging, get_logger, format_time
from src.ocr_engine import OCREngine
from src.postprocessor import Postprocessor
from src.pdf_handler import PDFHandler

logger = get_logger(__name__)


def process_image(engine: OCREngine, post: Postprocessor, image_path: Path) -> dict:
    """Process a single image through OCR pipeline."""
    import cv2

    start = time.perf_counter()

    # OCR
    results = engine.process(str(image_path))
    elapsed = time.perf_counter() - start

    # Load original image for annotation
    original = cv2.imread(str(image_path))

    # Export all formats
    exports = post.export_all(
        results=results,
        image=original,
        filename=image_path.name,
        processing_time=elapsed,
        device=engine.device_mgr.device_info.device,
        mode=engine.device_mgr.preset.name,
        draw_func=engine.draw_boxes,
    )

    return {
        "file": image_path.name,
        "regions": len(results),
        "time": elapsed,
        "exports": {k: str(v) for k, v in exports.items()},
    }


def process_pdf(
    engine: OCREngine,
    post: Postprocessor,
    pdf_path: Path,
) -> list:
    """Process a PDF file (multi-page)."""
    handler = PDFHandler(dpi=200)
    images = handler.pdf_to_images(str(pdf_path))

    all_results = []
    for i, page_img in enumerate(images):
        logger.info("📄 Page %d/%d", i + 1, len(images))
        start = time.perf_counter()

        results = engine.process(page_img)
        elapsed = time.perf_counter() - start

        page_name = f"{pdf_path.stem}_page{i + 1}"
        exports = post.export_all(
            results=results,
            image=page_img,
            filename=f"{page_name}.png",
            processing_time=elapsed,
            device=engine.device_mgr.device_info.device,
            mode=engine.device_mgr.preset.name,
            draw_func=engine.draw_boxes,
        )

        all_results.append({
            "page": i + 1,
            "regions": len(results),
            "time": elapsed,
        })

    return all_results


def main():
    parser = argparse.ArgumentParser(
        prog="DocOCR",
        description="Offline Document OCR Extractor — PaddleOCR + GPU/CPU Fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input data/input/invoice.jpg\n"
            "  python main.py --input data/input/ --device cuda --mode fast\n"
            "  python main.py --input document.pdf --output results/\n"
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="Image file, PDF, or directory")
    parser.add_argument("--output", "-o", default="data/output", help="Output directory")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device: auto/cuda/cpu (default: auto)")
    parser.add_argument("--mode", "-m", default="balanced", choices=["fast", "balanced", "accurate"],
                        help="Quality mode (default: balanced)")
    parser.add_argument("--lang", default="vi", help="OCR language (default: vi)")
    parser.add_argument("--denoise", action="store_true", help="Enable noise reduction")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)

    # Initialize engine
    logger.info("=" * 60)
    logger.info("🚀 Offline Document OCR Extractor")
    logger.info("=" * 60)

    engine = OCREngine(device=args.device, mode=args.mode, lang=args.lang, denoise=args.denoise)
    post = Postprocessor(output_dir=args.output)

    # Collect files
    from src.utils import IMAGE_EXTENSIONS, get_supported_files
    from src.pdf_handler import PDFHandler

    total_start = time.perf_counter()

    if input_path.is_file():
        files = [input_path]
    else:
        files = get_supported_files(input_path)

    if not files:
        print(f"❌ No supported files found in: {input_path}")
        sys.exit(1)

    logger.info("📂 Processing %d file(s)", len(files))

    results_summary = []
    for f in files:
        logger.info("━" * 40)
        if PDFHandler.is_pdf(str(f)):
            pages = process_pdf(engine, post, f)
            results_summary.append({"file": f.name, "pages": pages})
        else:
            result = process_image(engine, post, f)
            results_summary.append(result)

    total_time = time.perf_counter() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    for r in results_summary:
        if "pages" in r:
            total_regions = sum(p["regions"] for p in r["pages"])
            total_page_time = sum(p["time"] for p in r["pages"])
            print(f"\n📄 {r['file']} ({len(r['pages'])} pages)")
            print(f"   Regions: {total_regions} | Time: {format_time(total_page_time)}")
        else:
            print(f"\n✅ {r['file']}")
            print(f"   Regions: {r['regions']} | Time: {format_time(r['time'])}")
            for fmt, path in r.get("exports", {}).items():
                print(f"   📁 {fmt}: {path}")

    print(f"\n⏱️  Total: {format_time(total_time)}")
    print(f"🖥️  Device: {engine.device_mgr.device_info.device.upper()}")
    print(f"🎯 Mode: {engine.device_mgr.preset.name}")

    engine.cleanup()


if __name__ == "__main__":
    main()
