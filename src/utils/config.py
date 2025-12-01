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
    
    # Summarization settings
    MAX_INPUT_TOKENS = 512
    MAX_OUTPUT_TOKENS = 150
    MIN_SUMMARY_LENGTH = 50
    
    # Extractive summarization
    EXTRACTIVE_RATIO = 0.3  # 30% of original text
    
    # Abstractive summarization
    TEMPERATURE = 0.7
    NUM_BEAMS = 4
    DO_SAMPLE = False
    
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

