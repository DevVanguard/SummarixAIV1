"""Tests for abstractive summarization."""

import pytest

from src.core.abstractive.summarizer import AbstractiveSummarizer


def test_abstractive_initialization():
    """Test abstractive summarizer initialization."""
    summarizer = AbstractiveSummarizer()
    assert summarizer.use_quantization is True
    assert summarizer.model_loader is not None
    assert summarizer.preprocessor is not None


def test_abstractive_not_ready_initially():
    """Test that summarizer is not ready initially."""
    summarizer = AbstractiveSummarizer()
    assert summarizer.is_ready() is False


def test_abstractive_summarize_empty_text():
    """Test summarizing empty text."""
    summarizer = AbstractiveSummarizer()
    result = summarizer.summarize("")
    assert result == ""


def test_abstractive_cleanup():
    """Test cleanup method."""
    summarizer = AbstractiveSummarizer()
    summarizer.cleanup()
    assert summarizer.is_ready() is False

