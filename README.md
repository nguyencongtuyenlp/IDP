# 📄 Offline Document OCR Extractor

**PaddleOCR** | GPU/CPU Fallback | Gradio UI | Docker | Vietnamese + English

> Built an offline OCR application for documents (images/PDF) with GPU acceleration and CPU fallback.
> Exports extracted text + structured JSON with bounding boxes. Containerized for deployment.

## ✨ Features

- 🇻🇳 **Vietnamese OCR** — VietOCR Transformer for full diacritical marks (dấu)
- 🚀 **Hybrid Pipeline**: PaddleOCR detection + VietOCR recognition
- ⚡ **GPU acceleration** with automatic CPU fallback
- 🎯 **3 quality modes**: fast / balanced / accurate
- 📄 **PDF support**: Multi-page, auto-convert to images
- 📦 **Bounding box** visualization + annotated image export
- 🖥️ **Gradio Web UI**: Upload → OCR → preview → download
- 🐳 **Docker**: CPU + GPU images ready
- 📊 **Benchmark**: GPU vs CPU comparison

## 🏗️ Architecture

```
Image/PDF → Preprocessor → PaddleOCR (Detection) → VietOCR (Recognition) → Postprocessor
              ├─ EXIF rotate   └─ bbox detection       └─ Việt text + dấu    ├─ Text (.txt)
              ├─ Deskew                                                      ├─ JSON (.json)
              ├─ Resize        Fallback: PaddleOCR-only for non-Vietnamese   └─ Annotated (.png)
              └─ Denoise
```

## 🚀 Quick Start

### Install

```bash
# CPU only
pip install -r requirements.txt

# With GPU (CUDA 11.8+)
pip install -r requirements-gpu.txt
```

### CLI

```bash
# Single image
python main.py --input photo.jpg

# Batch processing
python main.py --input data/input/ --output data/output/

# Force CPU + fast mode
python main.py --input doc.jpg --device cpu --mode fast

# PDF
python main.py --input document.pdf

# Disable VietOCR (PaddleOCR recognition only)
python main.py --input photo.jpg --no-vietocr

# Fast VietOCR model (seq2seq)
python main.py --input photo.jpg --vietocr-model vgg_seq2seq
```

### Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

### Docker

```bash
# CPU
docker compose --profile cpu up

# GPU (requires nvidia-docker)
docker compose --profile gpu up
```

## 🎯 Quality Modes

| Mode | Max Size | Angle Detect | Speed | Use Case |
|------|----------|-------------|-------|----------|
| `fast` | 960px | ❌ | ⚡⚡⚡ | Quick previews |
| `balanced` | 1280px | ✅ | ⚡⚡ | Default (recommended) |
| `accurate` | 1920px | ✅ | ⚡ | High-quality scans |

## 📊 Benchmark

### CPU Results (Intel i5, balanced mode)

| Image | Time | Regions | Avg Confidence |
|-------|------|---------|----------------|
| tho.jpg (Vietnamese poem) | 726ms | 9 | 96.6% |
| hihi.jpg (Rotated signboard) | 1.1s | 13 | 95.5% |
| **Average** | **1.1s** | **11** | **96%** |

### Run Benchmark

```bash
# Single device
python benchmark.py --input data/input/ --device cpu

# GPU vs CPU comparison
python benchmark.py --input data/input/ --compare
```

### Tests

```
pytest tests/ -v
# 93 passed in 19.14s
```

## 📁 Project Structure

```
├── app.py                # Gradio Web UI
├── main.py               # CLI interface
├── benchmark.py          # Performance benchmark
├── Dockerfile.cpu/gpu    # Docker images
├── docker-compose.yml
├── configs/default.yaml  # Configuration
├── src/
│   ├── device_manager.py # GPU/CPU detection + fallback
│   ├── preprocessor.py   # Image preprocessing pipeline
│   ├── ocr_engine.py     # PaddleOCR wrapper
│   ├── postprocessor.py  # Output formatting + export
│   ├── pdf_handler.py    # PDF conversion
│   └── utils.py          # Logging + helpers
└── data/
    ├── input/            # Input documents
    └── output/           # OCR results
```

## 🛠️ Technology Stack

- **OCR**: PaddleOCR (detection + recognition)
- **Framework**: PaddlePaddle (GPU/CPU)
- **UI**: Gradio
- **PDF**: PyMuPDF
- **Image**: OpenCV, Pillow
- **Container**: Docker, Docker Compose

## 📝 Output Formats

### Text (`_extracted.txt`)
Plain text, reading order sorted, line-merged.

### JSON (`_result.json`)
Structured data with bounding boxes, confidence scores, metadata.

### Annotated (`_annotated.png`)
Original image with drawn bounding boxes and text labels.

## ⚙️ Environment

- **Works on CPU-only environments; GPU recommended for speed.**
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- Tested: NVIDIA T4, GTX 1650, CPU-only
- 93 unit tests (pytest)

## 📄 License

MIT
