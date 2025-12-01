"""Model loading module for quantized T5-small."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    import torch
except ImportError:
    raise ImportError(
        "Transformers and PyTorch are required. Install with: pip install transformers torch"
    )

from src.utils.config import Config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle loading and caching of quantized models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the model to load (default: from config)
        """
        self.model_name = model_name or Config.MODEL_NAME
        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5Tokenizer] = None
        self.device = torch.device("cpu")
        self._model_loaded = False
    
    def _get_model_path(self) -> Path:
        """Get local model path."""
        return Config.get_model_path()
    
    def _model_exists_locally(self) -> bool:
        """Check if model exists in local directory."""
        model_path = self._get_model_path()
        # Check for common model files
        required_files = ['config.json', 'tokenizer_config.json']
        for file in required_files:
            if not (model_path / file).exists():
                return False
        return True
    
    def _load_from_local(self) -> bool:
        """
        Load model from local directory.
        
        Returns:
            True if successful, False otherwise
        """
        model_path = self._get_model_path()
        
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        try:
            logger.info(f"Loading model from local path: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            # Load model with CPU optimization
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=torch.float32,  # Use FP32 for CPU
                low_cpu_mem_usage=True
            )
            
            # Move to CPU and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully from local path")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from local path: {str(e)}")
            return False
    
    def _try_quantization(self) -> bool:
        """
        Try to load model with INT8 quantization.
        
        Returns:
            True if quantization successful, False otherwise
        """
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            logger.info("Attempting INT8 quantization with bitsandbytes")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model_path = self._get_model_path()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                quantization_config=quantization_config,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            logger.info("Model loaded with INT8 quantization")
            return True
            
        except ImportError:
            logger.warning("bitsandbytes not available, skipping INT8 quantization")
            return False
        except Exception as e:
            logger.warning(f"INT8 quantization failed: {str(e)}, falling back to FP32")
            return False
    
    def load_model(self, use_quantization: bool = True) -> bool:
        """
        Load the model (with optional quantization).
        
        Args:
            use_quantization: Whether to attempt INT8 quantization
            
        Returns:
            True if successful, False otherwise
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return True
        
        # Check if model exists locally
        if not self._model_exists_locally():
            logger.error(
                f"Model not found locally at {self._get_model_path()}. "
                "Please run scripts/download_models.py first."
            )
            return False
        
        # Try quantization first if requested
        if use_quantization:
            if self._try_quantization():
                self._model_loaded = True
                return True
        
        # Fallback to regular FP32 loading
        if self._load_from_local():
            self._model_loaded = True
            return True
        
        return False
    
    def get_model(self) -> Tuple[Optional[T5ForConditionalGeneration], Optional[T5Tokenizer]]:
        """
        Get loaded model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer) or (None, None) if not loaded
        """
        if not self._model_loaded:
            logger.warning("Model not loaded. Call load_model() first.")
            return None, None
        
        return self.model, self.tokenizer
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._model_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Model unloaded from memory")

