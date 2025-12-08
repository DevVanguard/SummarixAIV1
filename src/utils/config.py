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
    
    # Abstractive summarization - tuned for better quality
    TEMPERATURE = 0.7  # Balanced for natural output
    NUM_BEAMS = 5  # Good balance between quality and speed
    DO_SAMPLE = False  # Use deterministic beam search
    LENGTH_PENALTY = 1.2  # Default penalty (adjusted per preset)
    REPETITION_PENALTY = 1.3  # Reduce repetition without being too aggressive
    NO_REPEAT_NGRAM_SIZE = 3  # Avoid 3-gram repetition
    
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
        Get max output tokens for a given preset.
        Adjusted for better quality summaries.
        
        Args:
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Maximum output tokens
        """
        preset = preset.lower()
        if preset == 'short':
            return 120  # Short: 1 sentence + 2-3 key points (~100-120 tokens)
        elif preset == 'long':
            return 350  # Long: Executive summary with details (~300-350 tokens)
        else:  # medium (default)
            return 250  # Medium: 6-10 sentences or 4-8 bullets (~200-250 tokens)
    
    @classmethod
    def get_min_length_for_preset(cls, preset: str) -> int:
        """
        Get minimum output tokens for a given preset.
        Adjusted to ensure quality summaries.
        
        Args:
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Minimum output tokens
        """
        preset = preset.lower()
        if preset == 'short':
            return 30  # Short: At least enough for 1 sentence + 2-3 points
        elif preset == 'long':
            return 100  # Long: Enough for structured executive summary
        else:  # medium (default)
            return 60  # Medium: Enough for 6-10 sentences or 4-8 bullets
    
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

