# 📊 Performance Benchmarks

Detailed benchmark results for **Offline Document OCR Extractor**.

## Test Environment

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA T4 (16GB VRAM, Lightning.ai) |
| **CPU** | Intel Core i5-8250U @ 1.60GHz (4 cores) |
| **RAM** | 16GB |
| **OS** | Ubuntu 20.04 LTS |
| **CUDA** | 11.8 |
| **Python** | 3.11 |

## GPU vs CPU Performance

### Test Dataset
- **tho.jpg**: Vietnamese poem (classical serif font, 9 text regions)
- **hihi.jpg**: Rotated signboard (modern font, 13 text regions)
- **cap.png**: Modern document (sans-serif, 15 text regions)

### Results

| Device | Mode | Image | Time | Regions | Confidence | VRAM |
|--------|------|-------|------|---------|------------|------|
| **NVIDIA T4** | Balanced | tho.jpg | **180ms** | 9 | 96.6% | 1.8GB |
| **NVIDIA T4** | Balanced | hihi.jpg | **245ms** | 13 | 95.5% | 1.9GB |
| **NVIDIA T4** | Balanced | cap.png | **252ms** | 15 | 95.0% | 2.1GB |
| **Intel i5** | Balanced | tho.jpg | 726ms | 9 | 96.6% | N/A |
| **Intel i5** | Balanced | hihi.jpg | 1.1s | 13 | 95.5% | N/A |
| **Intel i5** | Balanced | cap.png | 1.2s | 15 | 96.0% | N/A |

**GPU Speedup: 4.4x faster on average**

## VietOCR Accuracy Analysis

### Modern Sans-serif Fonts (Arial, Roboto, Helvetica)

| Metric | PaddleOCR Only | VietOCR Hybrid | Improvement |
|--------|---------------|----------------|-------------|
| **Character Accuracy** | 85% ❌ | **95%** ✅ | +10% |
| **Diacritical Marks** | 0% missing | **100%** preserved | +100% |
| **Word Error Rate** | 15% | **5%** | -10% |

**Example:**
- Input text: "Tiếng suối trong như tiếng hát xa"
- PaddleOCR: "Tieng suoi trong nhu tieng hat xa" ❌ (missing dấu)
- VietOCR: "Tiếng suối trong như tiếng hát xa" ✅

### Classical Serif Fonts (Times New Roman, Garamond)

| Metric | PaddleOCR Only | VietOCR Hybrid | Recommendation |
|--------|---------------|----------------|----------------|
| **Character Accuracy** | 80% | 73% ⚠️ | Use `--no-vietocr` |
| **Diacritical Marks** | 0% missing | 80% preserved | - |
| **Word Error Rate** | 20% | 27% | - |

**Note:** VietOCR model not optimized for classical serif fonts. PaddleOCR performs better in this case.

## Quality Mode Comparison

| Mode | Image Size | Processing Time (GPU) | Accuracy | Use Case |
|------|-----------|----------------------|----------|----------|
| **Fast** | 960px max | **120ms** ⚡⚡⚡ | 90% | Quick previews |
| **Balanced** | 1280px max | 250ms ⚡⚡ | 95% | **Default** |
| **Accurate** | 1920px max | 450ms ⚡ | 97% | High-quality scans |

## Memory Usage

### GPU (NVIDIA T4)

| Component | VRAM Usage |
|-----------|-----------|
| PaddleOCR (detection) | 800MB |
| VietOCR (transformer) | 1.2GB |
| **Total** | **~2GB** |

### CPU (Intel i5)

| Component | RAM Usage |
|-----------|----------|
| PaddleOCR (detection + recognition) | 1.5GB |
| VietOCR (transformer) | 2.0GB |
| **Total** | **~3.5GB** |

## Throughput (Batch Processing)

### GPU (T4)

| Batch Size | Images/sec | Avg Latency |
|-----------|-----------|-------------|
| 1 | 4.0 img/s | 250ms |
| 4 | 12.0 img/s | 333ms |
| 8 | 18.0 img/s | 444ms |

### CPU (i5)

| Batch Size | Images/sec | Avg Latency |
|-----------|-----------|-------------|
| 1 | 0.9 img/s | 1.1s |
| 4 | 2.8 img/s | 1.4s |
| 8 | 4.5 img/s | 1.8s |

## Run Your Own Benchmarks

```bash
# Single device test
python benchmark.py --input data/input/ --device cpu --mode balanced

# GPU vs CPU comparison
python benchmark.py --input data/input/ --compare --mode balanced

# Custom iterations
python benchmark.py --input data/input/ --device cuda --iterations 100
```

## Conclusion

✅ **GPU recommended** for production use (4.4x faster)  
✅ **VietOCR hybrid** for modern Vietnamese fonts (+10% accuracy)  
⚠️ **PaddleOCR-only** (`--no-vietocr`) for classical serif fonts  
✅ **Balanced mode** optimal for most use cases  
