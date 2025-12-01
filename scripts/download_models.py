"""Script to download and prepare models for offline use."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def download_model():
    """Download T5-small model from Hugging Face."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = Config.MODEL_NAME
        model_path = Config.get_model_path()
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Target directory: {model_path}")
        
        # Ensure directory exists
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_path))
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(str(model_path))
        
        logger.info(f"Model downloaded successfully to {model_path}")
        logger.info("Model is ready for offline use")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting model download...")
    Config.ensure_directories()
    
    if download_model():
        logger.info("Model download completed successfully")
        sys.exit(0)
    else:
        logger.error("Model download failed")
        sys.exit(1)

