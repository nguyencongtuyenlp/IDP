#!/usr/bin/env python3
"""
Gradio UI — Offline Document OCR Extractor.

Upload image/PDF → OCR → preview bbox → download results.

Usage:
    python app.py
    python app.py --device cuda --mode balanced --port 7860
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.utils import setup_logging, get_logger, format_time
from src.ocr_engine import OCREngine
from src.postprocessor import Postprocessor
from src.pdf_handler import PDFHandler

logger = get_logger(__name__)

# Global engine (initialized on startup)
ENGINE: OCREngine = None
POST: Postprocessor = None


def init_engine(device: str = "auto", mode: str = "balanced", lang: str = "vi",
                use_vietocr: bool = True):
    """Initialize OCR engine globally."""
    global ENGINE, POST
    ENGINE = OCREngine(device=device, mode=mode, lang=lang, use_vietocr=use_vietocr)
    POST = Postprocessor(output_dir=tempfile.mkdtemp())


def process_upload(
    file,
    device_choice: str,
    mode_choice: str,
    confidence: float,
    show_boxes: bool,
    use_vietocr: bool,
):
    """Main processing function for Gradio UI."""
    global ENGINE, POST

    if file is None:
        return None, "❌ No file uploaded", None, None, None

    file_path = file if isinstance(file, str) else file.name
    filename = Path(file_path).name

    # Reinitialize engine if settings changed
    if (ENGINE is None
            or ENGINE.device_mgr.requested_device != device_choice
            or ENGINE.device_mgr.mode != mode_choice
            or ENGINE.use_vietocr != use_vietocr):
        init_engine(device=device_choice, mode=mode_choice, use_vietocr=use_vietocr)

    start = time.perf_counter()

    try:
        # Handle PDF
        if PDFHandler.is_pdf(file_path):
            handler = PDFHandler(dpi=200)
            images = handler.pdf_to_images(file_path)
            if not images:
                return None, "❌ Could not convert PDF", None, None, None

            # Process first page for preview
            image = images[0]
            results = ENGINE.process(image, confidence_threshold=confidence)

            status = f"📄 PDF: {len(images)} pages | Showing page 1\n"
            status += f"📝 {len(results)} text regions detected"
        else:
            # Process image
            image = cv2.imread(file_path)
            if image is None:
                return None, "❌ Cannot read image", None, None, None

            results = ENGINE.process(file_path, confidence_threshold=confidence)

        elapsed = time.perf_counter() - start

        # Annotated preview
        if show_boxes and results:
            annotated = ENGINE.draw_boxes(
                image, results,
                color=(0, 200, 0), thickness=2,
                show_text=True, show_confidence=True,
            )
            preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        else:
            preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate text output
        lines = POST.merge_lines(results)
        text_output = "\n".join(lines)

        # Generate JSON output
        json_output = json.dumps({
            "file": filename,
            "device": ENGINE.device_mgr.device_info.device,
            "mode": ENGINE.device_mgr.preset.name,
            "num_regions": len(results),
            "text": text_output,
            "regions": [r.to_dict() for r in results],
            "processing_time_seconds": round(elapsed, 3),
        }, ensure_ascii=False, indent=2)

        # Status info
        status = (
            f"✅ **{filename}** — {len(results)} regions | "
            f"{format_time(elapsed)} | "
            f"{ENGINE.device_mgr.device_info.device.upper()} mode"
        )
        if results:
            avg_conf = sum(r.confidence for r in results) / len(results)
            status += f" | Avg conf: {avg_conf:.0%}"

        # Save files for download
        text_path = POST.export_text(results, filename)
        json_path = POST.export_json(
            results, filename, elapsed,
            ENGINE.device_mgr.device_info.device,
            ENGINE.device_mgr.preset.name,
        )

        return preview, status, text_output, json_output, [str(text_path), str(json_path)]

    except Exception as e:
        logger.error("❌ Processing failed: %s", e)
        return None, f"❌ Error: {str(e)}", None, None, None


def build_ui():
    """Build Gradio interface."""
    with gr.Blocks(
        title="📄 Offline Document OCR Extractor",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 10px; }
        .status-bar { font-size: 14px; }
        """,
    ) as app:
        gr.Markdown(
            "# 📄 Offline Document OCR Extractor\n"
            "**PaddleOCR** | GPU/CPU Fallback | Vietnamese + English\n\n"
            "Upload an image or PDF → Extract text with bounding boxes",
            elem_classes="main-title",
        )

        with gr.Row():
            # Left: Input
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📁 Upload Image / PDF",
                    file_types=["image", ".pdf"],
                    type="filepath",
                )

                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="🖥️ Device",
                        scale=1,
                    )
                    mode_dropdown = gr.Dropdown(
                        choices=["fast", "balanced", "accurate"],
                        value="balanced",
                        label="🎯 Mode",
                        scale=1,
                    )

                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="🔍 Confidence Threshold",
                )
                show_boxes = gr.Checkbox(value=True, label="📦 Show Bounding Boxes")
                use_vietocr = gr.Checkbox(value=True, label="🇻🇳 VietOCR (tiếng Việt có dấu)")

                process_btn = gr.Button("🚀 Extract Text", variant="primary", size="lg")

            # Right: Output
            with gr.Column(scale=1):
                preview_image = gr.Image(label="📷 Preview", type="numpy")
                status_bar = gr.Markdown("*Ready*", elem_classes="status-bar")

        with gr.Row():
            with gr.Column():
                text_output = gr.Textbox(
                    label="📝 Extracted Text",
                    lines=12, max_lines=30,
                )
            with gr.Column():
                json_output = gr.Textbox(
                    label="📊 Structured JSON",
                    lines=12, max_lines=30,
                )

        download_files = gr.File(label="📥 Download Results", visible=True)

        # Event handler
        process_btn.click(
            fn=process_upload,
            inputs=[file_input, device_dropdown, mode_dropdown, confidence_slider, show_boxes, use_vietocr],
            outputs=[preview_image, status_bar, text_output, json_output, download_files],
        )

        # Examples
        gr.Markdown("---\n### 📌 Tips")
        gr.Markdown(
            "- **Fast** mode: smaller images → quicker but may miss small text\n"
            "- **Accurate** mode: full resolution → best quality, slower\n"
            "- **CPU** mode: works without GPU, ~3-5x slower\n"
            "- Supports: JPG, PNG, BMP, TIFF, WebP, PDF"
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio OCR UI")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--mode", default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("🚀 Offline Document OCR Extractor — Gradio UI")
    logger.info("=" * 60)

    # Pre-initialize engine
    init_engine(device=args.device, mode=args.mode, lang=args.lang)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
