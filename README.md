# C4PS - Captioning, Enhancement, and Multilingual Processing

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-ff69b4.svg)](https://huggingface.co/docs/transformers/index)
[![Real-ESRGAN](https://img.shields.io/badge/RealESRGAN-Image%20Upscaling-4b8bbe.svg)](https://github.com/xinntao/Real-ESRGAN)
[![GFPGAN](https://img.shields.io/badge/GFPGAN-Face%20Restoration-d4a574.svg)](https://github.com/TencentARC/GFPGAN)
[![Discord](https://img.shields.io/badge/Discord-Webhooks-5865F2.svg)](https://discord.com/)
[![Translation](https://img.shields.io/badge/Translation-Multilingual%20NLP-2ea44f.svg)](https://huggingface.co/docs/transformers/en/tasks/translation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> A production-grade, modular pipeline for image enhancement, caption generation, and offline multilingual translation. Fully local, GPU-optimized, with optional Discord integration.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
  - [Main Pipeline](#main-pipeline)
  - [Enhancement Modes](#enhancement-modes)
  - [Test & Utility Scripts](#test--utility-scripts)
- [Module Documentation](#module-documentation)
  - [Captioning Module](#captioning-module)
  - [Enhancement Module](#enhancement-module)
  - [Translation Module](#translation-module)
  - [Utilities](#utilities)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

C4PS is a comprehensive, **offline-capable** pipeline that intelligently chains:

1. **Image Enhancement** – Upscaling (2x–8x) with optional face restoration via Real-ESRGAN & GFPGAN
2. **Caption Generation** – Transformer-based image-to-text (GIT model from HuggingFace)
3. **Multilingual Translation** – Offline translation to 40+ languages via IndicTrans2, NLLB, and MarianMT
4. **Output Reporting** – Discord webhook integration or local saved reports

All models are cached locally; after initial download, the pipeline runs **completely offline**.

---

## Key Features

### **Smart Enhancement**
- **4 Enhancement Modes**: Fast (2×), Vehicle Auto-Detect, Sharp (Anime/Vehicles), General (4×+)
- **Chaining**: Combine 2× then 4× for effective 8× upscaling
- **Optional Face Restoration**: GFPGAN integration for portrait photos
- **Adaptive Tiling**: Memory-aware tile sizing to prevent OOM on lower-VRAM GPUs

### **Intelligent Captioning**
- GIT (Generative Image-to-Text) transformer model
- ~350M parameters, lightweight, suitable for local inference
- Generates descriptive captions from original or enhanced images

### **Comprehensive Multilingual Support**
- **45+ target languages** including:
  - **Indian languages**: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu
  - **Major world languages**: English, French, Spanish, German, Chinese (simplified), Japanese, Russian, Arabic, Portuguese, Italian, Dutch, Korean, Turkish, Vietnamese, Indonesian, and more
- **3 Translation Backends**:
  - NLLB-200 (Indian + East Asian languages)
  - MarianMT (European & diverse languages)
  - IndicTrans2 via NLLB (optimized for Indian languages)

### **Discord Integration**
- Post captions and translations directly to Discord webhooks
- Automatic image compression if file size exceeds upload limits
- Rich embed formatting with translations

### **Hardware Optimization**
- Auto-detection: CUDA > MPS > CPU
- Tile-size selection for GPU memory constraints (400–1200px)
- Thread limiting for stability
- Benchmarking and metric computation (PSNR, SSIM)

### **Utility Scripts**
- Model weight conversion (state_dict ↔ checkpoint)
- Enhancement benchmark sweeps with quality metrics
- Side-by-side model comparison (x4 direct vs. x4plus tiled)
- Interactive translation testing and batch translation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch 2.x with CUDA 13.0 support |
| **Image Enhancement** | Real-ESRGAN, GFPGAN, BasicSR |
| **Captioning** | Microsoft GIT (HuggingFace Transformers) |
| **Translation** | IndicTrans2, NLLB-200, MarianMT, SentencePiece |
| **Terminal UI** | inquirer (interactive menus), tqdm (progress bars) |
| **Discord Integration** | discord.py (webhook support) |
| **Image Processing** | Pillow, OpenCV, scikit-image, NumPy |
| **Python** | 3.12+ (Python 3.14+ compatible) |

---

## Project Structure

```
C4PS/
├── main.py                           # Main orchestration & user interface
├── requirements.txt                  # Dependencies (CUDA 13.0 optimized)
├── LICENSE
├── README.md
│
├── captioning/                       # Image-to-text captioning
│   ├── __init__.py
│   ├── generator.py                  # CaptionGenerator class
│   └── model.py                      # GIT model loading & initialization
│
├── enhancement/                      # Image upscaling & face restoration
│   ├── __init__.py
│   ├── enhancer.py                   # enhance_image() high-level API
│   └── model.py                      # EnhancementModel (RealESRGAN + GFPGAN)
│
├── translation/                      # Multilingual translation
│   ├── __init__.py
│   ├── translator.py                 # translate_caption() high-level API
│   ├── router.py                     # Backend routing logic
│   ├── indic.py                      # IndicTransWrapper (NLLB for Indian langs)
│   ├── nllb.py                       # NLLBWrapper (NLLB-200 distilled)
│   └── marian.py                     # MarianMTWrapper (European/diverse langs)
│
├── utils/                            # Configuration & utilities
│   ├── __init__.py
│   ├── config.py                     # Configuration constants & env loading
│   ├── downloader.py                 # Asset & model weight downloading
│   ├── terminal_ui.py                # Terminal UI functions, report display
│   └── output_handler.py             # Discord webhook integration
│
├── tests/                            # Test & benchmark scripts
│   ├── __init__.py
│   ├── translations.py               # Interactive translation testing
│   ├── choose_and_run.py             # Interactive mode selection helper
│   ├── sweep_enhancement.py          # Benchmark multiple models
│   ├── compare_x4_x4plus.py          # Compare model variants
│   ├── convert_weights.py            # Weight format conversion
│   └── sweep_results.csv             # Benchmark results
│
├── assets/                           # Test images & sample data
│
├── outputs/                          # Enhanced images & reports (auto-created)
│
├── weights/                          # Model weights (auto-downloaded)
│   ├── RealESRGAN_x2.pth
│   ├── RealESRGAN_x4.pth
│   ├── RealESRGAN_x4plus.pth
│   ├── GFPGANv1.4.pth
│   ├── hf/                           # HuggingFace model cache
│   ├── transformer_model_cache/      # GIT captioning model
│   └── argostranslate/packages/      # Offline translation models
│
├── gfpgan/                           # GFPGAN dependencies
│   └── weights/                      # Face detection & parsing models
│
└── temp_basicsr/                     # BasicSR source (local build)
```

---

## Installation & Setup

### Prerequisites
- **Python 3.12+** (3.14+ supported)
- **GPU with CUDA 13.0** (optional, CPU also works)
- **~30GB disk space** (for all models and weights)
- **4GB+ VRAM** minimum (8GB+ recommended for best performance)

### Step 1: Clone & Create Virtual Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/meghmavani/C4PS.git
cd C4PS
python -m venv venv
.\venv\Scripts\Activate
```

**macOS/Linux (Bash):**
```bash
git clone https://github.com/meghmavani/C4PS.git
cd C4PS
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: `requirements.txt` includes PyTorch with CUDA 13.0 support. To use a different CUDA version or CPU-only, modify the `--index-url` directive before installing.

### Step 3: Create `.env` File

In the project root, create a `.env` file:

```env
# Optional: URL for test image downloads
IMAGE_URL="https://example.com/sample.jpg"

# Optional: Discord webhook (for online reporting)
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
DISCORD_SERVER_INVITE="https://discord.gg/..."

# Auto-set by config.py; can override if needed
# ARGOS_PACKAGES_DIR=weights/argostranslate/packages
```

### Step 4: Download Models & Weights (One-Time)

```powershell
python -m utils.downloader
```

This script will:
- Create `assets/`, `weights/`, and cache directories
- Download Real-ESRGAN weights (if missing)
- Download sample image (if `IMAGE_URL` is set)
- Initialize HuggingFace cache for GIT model
- Pre-download translation models

**Duration**: 5–15 minutes depending on internet speed.

You only need to run this **once** per machine. Subsequent runs are fully offline.

---

## How to Run

### Main Pipeline

**Interactive Full Pipeline:**
```powershell
python main.py
```

**Execution Flow:**

1. **Select Image Source** – Local path or URL
2. **Choose Output Mode** – Online (Discord) or Offline (local save)
3. **Select Enhancement Mode** – See [Enhancement Modes](#enhancement-modes)
4. **Set Tile Size** – Memory optimization (400, 800, 1200 pixels)
5. **Optional Prompts** – Face enhancement, chaining, language selection
6. **Assets Setup** – Download models if needed
7. **Enhancement** – Upscale image
8. **Captioning** – Generate caption
9. **Translation** – Translate to selected languages
10. **Output** – Save locally or post to Discord

---

### Enhancement Modes

When running `main.py`, you'll be prompted to choose an enhancement mode:

| Mode | Scale | Speed | Best For | Details |
|------|-------|-------|----------|---------|
| **Fast (x2)** | 2× | ~2 sec | Quick processing, memory-constrained | Lightweight, fast inference. Can be chained with 4× for 8× total. |
| **Vehicle Optimized** | 4× | ~10 sec | Mixed-content posts, auto-detect | Specialized for vehicles; smooth transitions on varied content. |
| **Sharp (Anime/Vehicles)** | 4× | ~10 sec | Anime, sharp subjects | Forced sharp mode; preserves edges on illustrated or geometric content. |
| **General (x4plus)** | 4×+ | ~150 sec | Portrait, landscape, general photos | Most versatile; best quality for natural images. Supports face enhancement (GFPGAN). |

#### **Memory Considerations:**
- **Small (400px tile)** – 2–4GB VRAM suitable
- **Medium (800px tile)** – 4–6GB VRAM (recommended)
- **Large (1200px tile)** – 8GB+ VRAM

If you encounter OOM errors, the system automatically falls back to smaller tiles.

#### **Chaining Enhancements:**
After choosing **Fast (x2)**, you have the option to upscale the result again with the 4× model, resulting in effective **8× total upscaling** (though with potential quality trade-offs).

---

### Test & Utility Scripts

#### 1. **Interactive Translation Testing** (`tests/translations.py`)

Test the translation pipeline with custom input:

```powershell
python tests/translations.py
```

**What it does:**
- Prompts for a sentence
- Translates to all supported languages
- Saves results to `outputs/translations_output.txt`
- Useful for verifying translation backends are working

---

#### 2. **Enhancement Benchmark Sweep** (`tests/sweep_enhancement.py`)

Run comprehensive benchmarks on different RealESRGAN models and tile sizes:

```powershell
python tests/sweep_enhancement.py
```

**What it does:**
- Tests models: x2, x4, x4plus, x8 variants
- Tests tile sizes: 400, 800, 1200 pixels
- Measures: execution time, PSNR, SSIM
- Outputs results to `sweep_results.csv`
- Helps identify optimal configs for your hardware

**Example output (sweep_results.csv):**
```csv
weight,tile_size,status,elapsed,psnr,ssim
RealESRGAN_x4.pth,400,ok,12.34,28.45,0.92
RealESRGAN_x4.pth,800,ok,18.56,28.67,0.93
RealESRGAN_x4_ema.pth,1200,oom,0,0,0
```

---

#### 3. **Model Comparison** (`tests/compare_x4_x4plus.py`)

Direct comparison of x4 direct vs. x4plus tiled enhancement:

```powershell
python tests/compare_x4_x4plus.py
```

**What it does:**
- Takes your image input
- Applies direct x4 upscaling (full image, no tiling)
- Applies x4plus upscaling (with adaptive tiling)
- Computes PSNR & SSIM metrics
- Saves side-by-side comparison to `outputs/`

**Outputs:**
- `outputs/out_x4.png` – Direct x4 result
- `outputs/out_x4plus.png` – Tiled x4plus result
- `outputs/input.png` – Original
- `outputs/reference.png` – Reference (if available)

---

#### 4. **Weight Conversion Utility** (`tests/convert_weights.py`)

Convert RealESRGAN weight files between formats:

```powershell
python tests/convert_weights.py
```

**What it does:**
- Converts state_dict format to checkpoint format (wrapped under `params_ema` key)
- Updates: `RealESRGAN_x2.pth` → `RealESRGAN_x2_chk.pth`
- Updates: `RealESRGAN_x4.pth` → `RealESRGAN_x4_chk.pth`

**When to use:** Only needed if you add custom or incompatible weight files.

---

#### 5. **Interactive Mode Helper** (`tests/choose_and_run.py`)

Choose and run enhancement with statistical selection:

```powershell
python tests/choose_and_run.py
```

**What it does:**
- Reads `sweep_results.csv`
- Offers options: fastest, most accurate, best for tile size
- Selects weight based on benchmarks
- Runs enhancement with optimized config
- Outputs to `outputs/`

**Useful for:** Automating enhancement selection based on prior benchmarks.

---

## Module Documentation

### Captioning Module

Located in `captioning/`

#### **generator.py** – `CaptionGenerator` Class

```python
from captioning.generator import CaptionGenerator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_gen = CaptionGenerator(device)

from PIL import Image
image = Image.open("photo.jpg")
caption = caption_gen.generate_caption(image)
print(caption)  # "A person standing in front of a sunset..."
```

**Key Methods:**
- `__init__(device)` – Initialize GIT model
- `generate_caption(image)` – Returns caption string (max 50 tokens)

**Model Details:**
- Model: `microsoft/git-base-coco` (~350M params)
- Tokenizer: GIT processor from HuggingFace
- Tokens: Up to 50 new tokens generated per caption
- Cache: `weights/transformer_model_cache/`

---

#### **model.py** – `create_transformer_model(device)`

```python
from captioning.model import create_transformer_model

model, processor = create_transformer_model(device)
```

Downloads and initializes the GIT model from HuggingFace.

---

### Enhancement Module

Located in `enhancement/`

#### **enhancer.py** – `enhance_image()` Function

High-level enhancement API:

```python
from enhancement.enhancer import enhance_image
from PIL import Image

enhanced = enhance_image(
    image_path="input.jpg",
    mode="general",           # "fast", "general", "sharp_anime", "auto_vehicle"
    tile_size=800,            # 400, 800, or 1200
    face_enhance=True,        # Enable GFPGAN
    enhance_faces=False       # Additional face boost
)

enhanced.save("output_enhanced.jpg")
```

**Parameters:**
- `image_path` (str) – Local file path
- `mode` (str) – Enhancement strategy
- `tile_size` (int) – Tile size for memory management
- `face_enhance` (bool) – Enable GFPGAN
- `enhance_faces` (bool) – Additional face processing

**Returns:** PIL Image

---

#### **model.py** – `EnhancementModel` Class

Core upscaling engine:

```python
from enhancement.model import EnhancementModel

model = EnhancementModel(
    model_params={...},
    outscale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    tile_size=800,
    face_enhance=True
)

import numpy as np
image_array = np.array(Image.open("input.jpg"))
upscaled = model.upscale(image_array)
```

**Features:**
- Auto device detection (CUDA > MPS > CPU)
- Memory-adaptive tiling (auto-reduce on OOM)
- GFPGAN face restoration (optional)
- Benchmarking & metric computation (PSNR, SSIM)
- Thread limiting for stability

---

### Translation Module

Located in `translation/`

#### **translator.py** – `translate_caption()` Function

High-level translation API:

```python
from translation.translator import translate_caption

caption = "This is a beautiful sunset."
target_langs = ["fr", "hi", "es", "ja"]

translations = translate_caption(caption, target_langs)
print(translations)
# {
#   "en": "This is a beautiful sunset.",
#   "fr": "C'est un beau coucher de soleil.",
#   "hi": "यह एक सुंदर सूर्यास्त है।",
#   ...
# }
```

**Returns:** Dict mapping language codes to translated strings

---

#### **router.py** – Backend Selection Logic

Routes translations to the appropriate backend:

```
Language Code → Backend
hi, ta, te, kn, ml, bn, mr, gu, pa, or, as, ur → IndicTrans (NLLB)
ko, tr, ja, pt, pb → NLLB
fr, es, de, zh, zt, ru, ar, it, nl, vi, ... → MarianMT
```

---

#### **indic.py** – `IndicTransWrapper` (Indian Languages)

NLLB-based translation for Indian languages:

```python
from translation.indic import IndicTransWrapper

wrapper = IndicTransWrapper.get_instance()
result = wrapper.translate("Hello", target_lang="hi", source_lang="en")
# "नमस्ते"
```

**Supported Languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese

**Model:** `facebook/nllb-200-distilled-600M`

---

#### **nllb.py** – `NLLBWrapper` (Multilingual)

NLLB-200 for diverse languages:

```python
from translation.nllb import NLLBWrapper

wrapper = NLLBWrapper.get_instance()
result = wrapper.translate("Hello", target_lang="ja", source_lang="en")
# "こんにちは"
```

**Supported:** Korean, Turkish, Japanese, Portuguese, and 30+ others via FLORES-200 codes.

---

#### **marian.py** – `MarianMTWrapper` (European & Diverse)

Helsinki-NLP MarianMT for broad language support:

```python
from translation.marian import MarianMTWrapper

wrapper = MarianMarian.get_instance("fr", "en")  # target, source
result = wrapper.translate("Good morning")
# "Bonjour"
```

**Memory Optimization:** Only one model loaded at a time; automatic unloading prevents OOM on 4GB VRAM.

**Supported:** French, Spanish, German, Chinese, Russian, Arabic, Italian, Dutch, Vietnamese, Indonesian, Turkish, Portuguese, and 20+ others.

---

### Utilities

Located in `utils/`

#### **config.py** – Central Configuration

```python
from utils import config

# Automatically loads .env file
print(config.BASE_DIR)                    # Projects root
print(config.CAPTIONING_MODEL_PATH)       # HF model cache
print(config.DISCORD_WEBHOOK_URL)         # From .env
print(config.TARGET_LANGUAGES)            # List of 45+ langs
```

**Key Variables:**
- `BASE_DIR`, `ASSETS_DIR`, `WEIGHTS_DIR` – Paths
- `CAPTIONING_MODEL_PATH` – GIT model cache
- `TARGET_LANGUAGES` – Supported language codes
- `DISCORD_*` – Webhook credentials (from `.env`)

---

#### **downloader.py** – Asset Downloading

```python
from utils.downloader import download_assets

download_assets()  # Downloads all necessary weights & images
```

**Downloads:**
- Sample image (if `IMAGE_URL` set)
- RealESRGAN x4plus weights
- Translation models (lazy-loaded on first use via HuggingFace)

---

#### **terminal_ui.py** – UI Utilities

```python
from utils.terminal_ui import (
    clear_screen, print_header, print_step,
    display_offline_report, suppress_warnings
)

clear_screen()
print_header()
print_step(1, "Processing image...")
display_offline_report(image_path, caption, translations)
```

**Available Functions:**
- `clear_screen()` – Clear terminal (cross-platform)
- `print_header()` – Display application header
- `print_step(num, msg)` – Formatted step output
- `display_offline_report()` – Show results, open image
- `suppress_warnings()` – Suppress library warnings

---

#### **output_handler.py** – Discord Integration

```python
from utils.output_handler import send_to_discord, is_discord_upload_too_large

if is_discord_upload_too_large(image_path):
    print("Image size OK for Discord")

send_to_discord(
    webhook_url="https://discord.com/api/webhooks/...",
    server_invite="https://discord.gg/...",
    enhanced_path="output.jpg",
    english_caption="Beautiful sunset.",
    multilingual_captions={"fr": "Beau coucher de soleil.", ...},
    allow_compression=True  # Auto-compress if needed
)
```

**Features:**
- File size validation (7.9 MB Discord limit)
- Automatic progressive compression (scales: 1.0, 0.85, 0.7, ...; quality: 90, 80, 70, ...)
- Rich Discord embeds with caption & translations
- Temp file cleanup

---

## Configuration

### Central Config File (`utils/config.py`)

```python
# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# --- MODELS ---
CAPTIONING_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'transformer_model_cache')
ENHANCER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'RealESRGAN_x4plus.pth')

# --- LANGUAGES (45+ supported) ---
TARGET_LANGUAGES = [
    "sq", "ar", "az", "eu", "bg", "ca", "zh", "zt", "cs", "da",
    "nl", "eo", "et", "fi", "fr", "gl", "de", "el", "he", "hu",
    "id", "ga", "it", "ja", "ko", "ky", "lv", "lt", "ms", "nb",
    "fa", "pl", "pt", "pb", "ro", "ru", "sk", "sl", "es", "sv",
    "tl", "th", "tr", "uk", "vi"
]

# --- DISCORD (from .env) ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_SERVER_INVITE = os.getenv("DISCORD_SERVER_INVITE")
```

### Environment Variables (`.env`)

```env
# Sample image for testing
IMAGE_URL="https://example.com/image.jpg"

# Discord webhook credentials
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
DISCORD_SERVER_INVITE="https://discord.gg/YOUR_CODE"

# Translation model directory (auto-configured)
ARGOS_PACKAGES_DIR=weights/argostranslate/packages
```

---

## Advanced Usage

### Batch Processing

Process multiple images with custom enhancement settings:

```python
from enhancement.enhancer import enhance_image
from captioning.generator import CaptionGenerator
from utils import config
import torch
import os
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_gen = CaptionGenerator(device)

for image_path in glob("batch_images/*.jpg"):
    print(f"Processing {image_path}...")
    
    # Enhance
    enhanced = enhance_image(image_path, mode="general", tile_size=800)
    
    # Caption
    caption = caption_gen.generate_caption(enhanced)
    
    # Save
    base = os.path.splitext(os.path.basename(image_path))[0]
    enhanced.save(f"outputs/{base}_enhanced.jpg")
    
    # Log
    with open(f"outputs/{base}_caption.txt", "w") as f:
        f.write(caption)
```

### Custom Enhancement Workflow

```python
from enhancement.enhancer import enhance_image
from utils.output_handler import send_to_discord
from utils import config

# 1. Enhance image
enhanced = enhance_image("input.jpg", mode="fast", tile_size=400)

# 2. Save locally
enhanced.save("output.jpg")

# 3. Send to Discord (if configured)
send_to_discord(
    config.DISCORD_WEBHOOK_URL,
    config.DISCORD_SERVER_INVITE,
    "output.jpg",
    "My enhanced image",
    {},  # No translations
    allow_compression=True
)
```

### Custom Language Selection

```python
from translation.translator import translate_caption

caption = "A dog playing fetch in the park."

# Translate to specific languages
languages = ["hi", "fr", "es", "ja", "ru"]
translations = translate_caption(caption, languages)

for lang, translated in translations.items():
    print(f"{lang.upper()}: {translated}")
```

---

## Troubleshooting

### **Issue: CUDA Out of Memory (OOM)**

**Solution 1:** Reduce tile size
```powershell
# When prompted, choose "Small (400)" instead of "Large (1200)"
python main.py
```

**Solution 2:** Use Fast mode (x2 instead of x4)
```python
enhanced = enhance_image(image_path, mode="fast", tile_size=400)
```

**Solution 3:** Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

---

### **Issue: Translation Model Not Loading**

**Symptom:** `RuntimeError: sentencepiece not found`

**Solution:**
```powershell
pip install sentencepiece
python -m utils.downloader
```

---

### **Issue: Discord Webhook Fails**

**Symptom:** `discord.errors.HTTPException: 400 Bad Request`

1. Verify webhook URL in `.env` is correct
2. Check image file size (< 8MB)
3. Enable auto-compression:
   ```python
   send_to_discord(..., allow_compression=True)
   ```

---

### **Issue: Model Cache Growing Too Large**

**Solution:** Clear HuggingFace cache
```powershell
del /s weights\hf  # Windows
rm -rf weights/hf  # macOS/Linux
```

Models will re-download on next use.

---

### **Issue: Python 3.14 Compatibility**

**Status:** Fully compatible (spacy dependency removed in requirements.txt)

If you encounter issues:
```powershell
pip install --upgrade transformers torch
```

---

## Performance Notes

### Typical Execution Times
(Measured on RTX 3070 Ti with 8GB VRAM)

| Mode | Scale | Speed | VRAM |
|------|-------|-------|------|
| Fast (x2) | 2× | ~2 sec | 2GB |
| Vehicle/Sharp (x4) | 4× | ~10 sec | 4GB |
| General (x4plus) | 4× | ~150 sec | 6GB |
| Chained (2× + 4×) | 8× | ~160 sec | 6GB |

### Memory Optimization Tips

1. **Start with tile size 800** – Balanced for most GPUs
2. **Use Fast mode for quick iteration** – 2× is 75× faster than 4×
3. **Disable face enhancement if not needed** – Saves ~1-2 sec
4. **Monitor VRAM** – Use `nvidia-smi` (NVIDIA) or `gpustat` (AMD)

---

## Roadmap

### Completed
- [x] Multi-mode image enhancement (2×, 4×, 8×)
- [x] GIT-based image captioning
- [x] 45+ language translation support
- [x] Discord webhook integration
- [x] GFPGAN face restoration
- [x] Memory-adaptive tiling
- [x] Interactive terminal UI
- [x] Benchmark & comparison utilities
- [x] Python 3.14 compatibility

### Planned
- [ ] REST API for external integration
- [ ] Batch processing mode
- [ ] Additional captioning models (LLaVA, BLIP)
- [ ] Video frame processing
- [ ] Mobile ONNX export
- [ ] Web UI dashboard

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

- **Real-ESRGAN** – Xinntao (upscaling engine)
- **GFPGAN** – TencentARC (face restoration)
- **BasicSR** – OpenMMLab (SR infrastructure)
- **GIT Model** – Microsoft (image captioning)
- **Translation Models** – Meta (NLLB), Helsinki-NLP (MarianMT), FreshLLM (IndicTrans2)
- **Discord.py** – Community (webhook integration)
