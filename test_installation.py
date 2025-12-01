"""Quick test script to verify SummarixAI installation."""

import sys
from pathlib import Path

print("=" * 50)
print("Testing SummarixAI Installation...")
print("=" * 50)
print(f"Python version: {sys.version.split()[0]}")
print(f"Python path: {sys.executable}")
print()

errors = []

# Test core imports
print("Testing core modules...")
try:
    from src.core.pdf_processor import PDFProcessor
    print("✓ PDF processor imported successfully")
except Exception as e:
    print(f"✗ PDF processor error: {e}")
    errors.append("PDF processor")

try:
    from src.core.extractive.textrank import TextRankSummarizer
    print("✓ Extractive summarizer imported successfully")
except Exception as e:
    print(f"✗ Extractive summarizer error: {e}")
    errors.append("Extractive summarizer")

try:
    from src.core.abstractive.summarizer import AbstractiveSummarizer
    print("✓ Abstractive summarizer imported successfully")
except Exception as e:
    print(f"✗ Abstractive summarizer error: {e}")
    errors.append("Abstractive summarizer")

# Test GUI imports
print("\nTesting GUI modules...")
try:
    from PyQt6.QtWidgets import QApplication
    print("✓ PyQt6 imported successfully")
except Exception as e:
    print(f"✗ PyQt6 error: {e}")
    errors.append("PyQt6")

# Test model files
print("\nTesting model files...")
from src.utils.config import Config
model_path = Config.get_model_path()
if model_path.exists():
    config_file = model_path / "config.json"
    if config_file.exists():
        print(f"✓ Model files found at: {model_path}")
    else:
        print(f"⚠ Model directory exists but config.json not found")
        print("  Run: python scripts/download_models.py")
else:
    print(f"⚠ Model directory not found: {model_path}")
    print("  Run: python scripts/download_models.py")

# Summary
print("\n" + "=" * 50)
if errors:
    print(f"✗ Installation has {len(errors)} issue(s):")
    for error in errors:
        print(f"  - {error}")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
else:
    print("✓ All core modules imported successfully!")
    print("✓ Installation looks good!")
print("=" * 50)

