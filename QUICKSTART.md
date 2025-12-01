# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- ~500MB disk space

## Installation Steps

### 1. Set up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

Before running the application, you need to download the T5-small model:

```bash
python scripts/download_models.py
```

This will download the model to `resources/models/t5-small/` for offline use.

### 4. (Optional) Quantize Model

For better performance on CPU, you can quantize the model:

```bash
python scripts/quantize_model.py
```

Note: This requires `bitsandbytes` which may not work on all systems. The application will work without quantization using FP32.

### 5. Run the Application

You can run the application in two ways:

**Option 1: Using the launcher script (Recommended)**
```bash
python run.py
```

**Option 2: Running main.py directly**
```bash
python src/main.py
```

Both methods will work correctly.

## Usage

1. **Select PDF**: Click "Browse Files" or drag-and-drop a PDF file
2. **Choose Mode**: 
   - **Extractive**: Faster, extracts key sentences (recommended for quick summaries)
   - **Abstractive**: Slower, generates new summary text (requires model loading)
3. **Summarize**: Click the "Summarize" button
4. **Export**: Use the export buttons to save summary as TXT, PDF, or DOCX

## Building Executable

To create a standalone executable:

```bash
python scripts/build_installer.py
```

This will create an executable in the `dist/` directory.

## Troubleshooting

### Model Not Found Error

If you see "Model not found locally", run:
```bash
python scripts/download_models.py
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### NLTK Data Missing

The application will automatically download required NLTK data on first run. If it fails, manually download:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Notes

- The application operates completely offline after models are downloaded
- First run of abstractive mode may take longer due to model loading
- Large PDFs (>100 pages) may take several minutes to process
- Extractive mode is recommended for very long documents

