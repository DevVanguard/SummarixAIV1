"""Tests for PDF processor module."""

import pytest
from pathlib import Path

from src.core.pdf_processor import PDFProcessor


def test_pdf_processor_initialization():
    """Test PDF processor initialization."""
    processor = PDFProcessor()
    assert processor.doc is None


def test_pdf_processor_context_manager():
    """Test PDF processor as context manager."""
    with PDFProcessor() as processor:
        assert processor is not None


def test_load_nonexistent_pdf():
    """Test loading non-existent PDF."""
    processor = PDFProcessor()
    result = processor.load_pdf(Path("nonexistent.pdf"))
    assert result is False


def test_get_page_count_no_doc():
    """Test getting page count with no document loaded."""
    processor = PDFProcessor()
    assert processor.get_page_count() == 0

