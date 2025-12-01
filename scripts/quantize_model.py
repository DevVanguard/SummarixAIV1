"""Script to quantize models for efficient CPU inference."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()


def quantize_model():
    """Quantize T5-small model to INT8."""
    try:
        from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
        import torch
        
        model_path = Config.get_model_path()
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Please run download_models.py first")
            return False
        
        logger.info("Quantizing model to INT8...")
        
        # Try to use bitsandbytes for quantization
        try:
            import bitsandbytes as bnb
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            logger.info("Loading model with INT8 quantization...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                quantization_config=quantization_config,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Save quantized model
            quantized_path = model_path / "quantized"
            quantized_path.mkdir(exist_ok=True)
            
            logger.info(f"Saving quantized model to {quantized_path}")
            model.save_pretrained(str(quantized_path))
            
            logger.info("Model quantized successfully")
            return True
            
        except ImportError:
            logger.warning("bitsandbytes not available. Model will use FP32.")
            logger.info("For INT8 quantization, install bitsandbytes: pip install bitsandbytes")
            return False
            
    except Exception as e:
        logger.error(f"Error quantizing model: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting model quantization...")
    
    if quantize_model():
        logger.info("Model quantization completed")
        sys.exit(0)
    else:
        logger.warning("Model quantization skipped or failed")
        logger.info("Application will use FP32 model (still functional)")
        sys.exit(0)  # Not a critical error

