"""Tests for extractive summarization."""

import pytest

from src.core.extractive.textrank import TextRankSummarizer


def test_textrank_initialization():
    """Test TextRank summarizer initialization."""
    summarizer = TextRankSummarizer()
    assert summarizer.ratio == 0.3
    assert summarizer.preprocessor is not None


def test_textrank_summarize_empty_text():
    """Test summarizing empty text."""
    summarizer = TextRankSummarizer()
    result = summarizer.summarize("")
    assert result == ""


def test_textrank_summarize_short_text():
    """Test summarizing short text."""
    summarizer = TextRankSummarizer()
    text = "This is a short text. It has two sentences. That's all."
    result = summarizer.summarize(text, num_sentences=1)
    assert len(result) > 0


def test_textrank_summarize_with_ratio():
    """Test summarizing with ratio."""
    summarizer = TextRankSummarizer(ratio=0.5)
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    result = summarizer.summarize(text)
    assert len(result) > 0

