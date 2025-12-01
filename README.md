# SummarixAI Desktop Application

A standalone, offline-capable desktop application for document summarization with support for both extractive (TextRank) and abstractive (quantized T5-small) summarization methods.

## Features

- **Dual Summarization Modes**: 
  - Extractive summarization using TextRank algorithm
  - Abstractive summarization using quantized T5-small model
- **Fully Offline**: No internet dependency, all processing occurs locally
- **Privacy-First**: All data processing happens on your device, no data leaves your system
- **Professional GUI**: Modern, sleek interface built with PyQt6
- **Cross-Platform**: Supports Windows, macOS, and Linux
- **CPU-Optimized**: Uses quantized models for efficient CPU-only inference

## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd SummarixAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and prepare models (required for abstractive summarization):
```bash
python scripts/download_models.py
python scripts/quantize_model.py  # Optional: for INT8 quantization
```

**Note:** Model files are not included in the repository due to size. They will be downloaded automatically when you run the download script.

5. Run the application:
```bash
python src/main.py
```

### From Installer

Download the platform-specific installer from the releases page and follow the installation wizard.

## Usage

1. Launch SummarixAI
2. Select a PDF file using the file browser or drag-and-drop
3. Choose your preferred summarization mode (Extractive or Abstractive)
4. Click "Summarize" and wait for processing
5. Review the summary and export if needed (TXT, PDF, DOCX)

## Technical Details

- **PDF Processing**: PyMuPDF (fitz) for robust text extraction
- **Extractive Summarization**: TextRank algorithm with NetworkX
- **Abstractive Summarization**: Quantized T5-small model (INT8)
- **GUI Framework**: PyQt6
- **Packaging**: PyInstaller for single-executable distribution

## Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- ~500MB disk space for application and models

## License

MIT License

## Privacy

SummarixAI operates completely offline. No data is transmitted over the network, and all processing occurs locally on your device.

