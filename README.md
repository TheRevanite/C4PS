# C4PS - Captioning, Enhancement, and Multilingual Processing for Socials

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-ff69b4.svg)](https://huggingface.co/docs/transformers/index)
[![Real-ESRGAN](https://img.shields.io/badge/RealESRGAN-Image%20Enhancement-4b8bbe.svg)](https://github.com/xinntao/Real-ESRGAN)
[![Argos Translate](https://img.shields.io/badge/Argos%20Translate-Offline%20Translation-008080.svg)](https://github.com/argosopentech/argos-translate)
[![Discord](https://img.shields.io/badge/Discord-Webhooks-5865F2.svg)](https://discord.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> A pipeline for image enhancement, captioning, and multilingual translation, designed for social media automation and reporting.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview
C4PS is a modular Python system for automated image enhancement, caption generation, and translation. It leverages state-of-the-art models for image super-resolution (Real-ESRGAN), transformer-based captioning, and offline translation (Argos Translate), with optional Discord integration for online reporting.

The pipeline is designed for content creators, researchers, and social media managers who need high-quality, multilingual image captions and reporting, all without the need for model training or cloud APIs.

## Key Features
- **Image Enhancement**: Upscales and improves image quality using Real-ESRGAN
- **Automated Captioning**: Generates descriptive captions using a pre-trained transformer model
- **Offline Multilingual Translation**: Translates captions into multiple languages using Argos Translate
- **Discord Integration**: Sends results to Discord via webhooks for online sharing
- **Terminal UI**: Interactive and color-coded terminal interface for local reporting
- **Configurable Pipeline**: Easily adjust target languages, model paths, and output modes
- **No Training Required**: Uses pre-trained models for all tasks

## Tech Stack
- **Python 3.12+**
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Captioning model
- **Real-ESRGAN**: Image enhancement
- **Argos Translate**: Offline translation
- **NumPy, Pillow**: Image and array manipulation
- **Discord.py**: Online reporting
- **Inquirer**: Terminal UI
- **python-dotenv**: Environment variable management

## Project Structure
```
C4PS/
├── main.py                  # Main pipeline script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── assets/                  # Downloaded images and assets
├── weights/                 # Model weights and cache
│   ├── transformer_model_cache # Captioning model files (microsoft/git-base-coco)
│   └── RealESRGAN_x4plus.pth   # Real-ESRGAN model weights
├── captioning/
│   ├── generator.py         # Caption generation logic
│   ├── model.py             # Loads transformer model
│   └── vocabulary.py        # Vocabulary utilities
├── enhancement/
│   ├── enhancer.py          # Image enhancement logic
│   └── model.py             # Loads Real-ESRGAN model
├── translation/
│   └── translator.py        # Offline translation logic (Argos Translate)
├── utils/
│   ├── config.py            # Central configuration
│   ├── downloader.py        # Asset and model downloader
│   ├── output_handler.py    # Discord and terminal reporting
│   └── terminal_ui.py       # Terminal UI utilities
└── .env                     # Discord webhook and secrets
```

## Setup and Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/TheRevanite/C4PS.git
cd C4PS
```

### 2. (Optional) Create a Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Discord Webhook (Optional)
Create a `.env` file in the root directory:
```env
DISCORD_WEBHOOK_URL="your_discord_webhook_url_here"
DISCORD_SERVER_INVITE="your_discord_server_invite_here"
IMAGE_URL="your_image_link_here"
```

## How to Run

Run the main pipeline script:
```powershell
python main.py
```

The application will:
1. Download required assets and models
2. Enhance the input image
3. Generate an English caption
4. Translate the caption into multiple languages (offline)
5. Output results to Discord (online mode) or display in terminal (offline mode)

## Configuration

All key parameters are set in `utils/config.py`:

| Parameter                | Description                                 | Default Value                |
|--------------------------|---------------------------------------------|------------------------------|
| `IMAGE_URL`              | URL for sample image download               | Set in .env                 |
| `ENHANCER_WEIGHTS_URL`   | Real-ESRGAN weights download URL            | Provided in config.py        |
| `TARGET_LANGUAGES`       | Languages for translation                   | `[fr, es, de]`               |
| `CAPTIONING_MODEL_PATH`  | Local cache for captioning model            | `weights/transformer_model_cache` |
| `DISCORD_WEBHOOK_URL`    | Discord webhook for online reporting        | Set in .env                  |

## Roadmap
- [x] Image enhancement with Real-ESRGAN
- [x] Transformer-based captioning
- [x] Offline multilingual translation
- [x] Discord webhook integration
- [x] Interactive terminal UI
- [ ] Add more language support
- [ ] REST API for external integration
- [ ] Edge device optimization

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request


## Contributors

- [Mitul K M](https://github.com/switchtwitch12345)
- [Aadit Pani](https://github.com/AaditPani-RVU)
- [Manya Jain](https://github.com/Manyajain2435)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

**Made with care for intelligent content creation and sharing.**
