"""Configuration management for SummarixAI."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration."""
    
    # Application info
    APP_NAME = "SummarixAI"
    APP_VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
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

