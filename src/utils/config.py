"""Configuration management for SummarixAI."""

import os
import sys
from pathlib import Path
from typing import Optional


def get_base_path() -> Path:
    """Get the base path, handling both development and PyInstaller executable modes."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        # PyInstaller sets _MEIPASS to the temp folder where it extracts files
        base_path = Path(sys._MEIPASS)
    else:
        # Running as script
        base_path = Path(__file__).parent.parent.parent
    return base_path


class Config:
    """Application configuration."""
    
    # Application info
    APP_NAME = "SummarixAI"
    APP_VERSION = "1.0.0"
    
    # Paths - handle both development and PyInstaller executable
    BASE_DIR = get_base_path()
    RESOURCES_DIR = BASE_DIR / "resources"
    MODELS_DIR = RESOURCES_DIR / "models"
    ICONS_DIR = RESOURCES_DIR / "icons"
    
    # Model configuration
    MODEL_NAME = "t5-small"
    MODEL_CACHE_DIR = MODELS_DIR / MODEL_NAME
    
    # Summarization settings - improved defaults for better quality
    MAX_INPUT_TOKENS = 512
    MAX_OUTPUT_TOKENS = 250  # Increased from 150 for more comprehensive summaries
    MIN_SUMMARY_LENGTH = 50
    
    # Summary length presets (as ratio of input)
    SUMMARY_LENGTH_SHORT = 0.15   # 15% of original
    SUMMARY_LENGTH_MEDIUM = 0.25  # 25% of original
    SUMMARY_LENGTH_LONG = 0.35    # 35% of original
    
    # Extractive summarization
    EXTRACTIVE_RATIO = 0.3  # 30% of original text
    
    # Abstractive summarization - QUALITY FIRST (time is not a constraint)
    TEMPERATURE = 1.0  # Default temperature
    NUM_BEAMS = 4  # Use beam search for quality (slower but much better)
    DO_SAMPLE = False  # Use deterministic generation
    LENGTH_PENALTY = 1.0  # Neutral penalty
    REPETITION_PENALTY = 1.3  # Moderate penalty (too high causes cutoffs)
    NO_REPEAT_NGRAM_SIZE = 3  # Avoid 3-gram repetition
    
    # CPU-specific optimizations
    USE_CACHE = True  # Enable KV cache for faster generation
    EARLY_STOPPING = True  # Stop when beam search finds good solution
    
    # File validation settings
    MAX_FILE_SIZE_MB = 100  # Maximum file size in MB (hard limit)
    WARNING_FILE_SIZE_MB = 50  # Warning threshold in MB
    MIN_TEXT_LENGTH = 100  # Minimum extractable text characters
    MAX_PAGES_WARNING = 500  # Warn if PDF has more than this many pages
    
    # Memory validation settings
    MIN_AVAILABLE_MEMORY_MB = 500  # Minimum available memory required
    MEMORY_MULTIPLIER = 4  # File size × this for processing memory estimate
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.ICONS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get the path to the model directory."""
        return cls.MODEL_CACHE_DIR
    
    @classmethod
    def is_offline_mode(cls) -> bool:
        """Check if running in offline mode (no internet)."""
        # Always return True for offline-first application
        return True
    
    @classmethod
    def get_output_length_for_preset(cls, preset: str) -> int:
        """
        Get max output tokens for a given preset (PER CHUNK).
        Shorter chunks = better quality on CPU. Multiple chunks are combined.
        
        Args:
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Maximum output tokens per chunk
        """
        preset = preset.lower()
        if preset == 'short':
            return 100  # Short: Brief per chunk
        elif preset == 'long':
            return 150  # Long: 150 tokens per chunk for stability, 10 chunks = 1500 tokens
        else:  # medium (default)
            return 120  # Medium: Balanced per chunk
    
    @classmethod
    def get_min_length_for_preset(cls, preset: str) -> int:
        """
        Get minimum output tokens for a given preset (PER CHUNK).
        These are per-chunk minimums - multiple chunks will be combined.
        
        Args:
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Minimum output tokens per chunk
        """
        preset = preset.lower()
        if preset == 'short':
            return 30  # Short: At least a good paragraph per chunk
        elif preset == 'long':
            return 60  # Long: 40% of max for quality
        else:  # medium (default)
            return 50  # Medium: Balanced per chunk
    
    @classmethod
    def get_length_penalty_for_preset(cls, preset: str) -> float:
        """
        Get length penalty for a given preset.
        Lower penalty encourages shorter summaries, higher encourages longer.
        Adjusted for better control over output length.
        
        Args:
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Length penalty value
        """
        preset = preset.lower()
        if preset == 'short':
            return 0.6  # Lower penalty to keep it concise
        elif preset == 'long':
            return 1.4  # Higher penalty to encourage comprehensive summaries
        else:  # medium (default)
            return 1.0  # Balanced penalty for medium summaries

