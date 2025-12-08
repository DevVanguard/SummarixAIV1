"""
Model loading module for quantized T5-small.
This module handles downloading, loading, and managing the T5 model for abstractive summarization.
It supports both quantized (INT8) and full precision (FP32) model loading.
"""

"""
Model loading module for quantized T5-small.
This module handles downloading, loading, and managing the T5 model for abstractive summarization.
It supports both quantized (INT8) and full precision (FP32) model loading.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Lazy imports - only import transformers when actually needed
# This prevents import warnings when using extractive mode only
AutoTokenizer = None
AutoModelForSeq2SeqLM = None
T5ForConditionalGeneration = None
T5Tokenizer = None
torch = None

from src.utils.config import Config

logger = logging.getLogger(__name__)


def _ensure_transformers_imported():
    """
    Lazy import of transformers and torch.
    Only imports when actually needed (abstractive mode).
    """
    global AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, torch
    
    if AutoTokenizer is None:
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSeq2SeqLM,
                T5ForConditionalGeneration,
                T5Tokenizer
            )
            import torch
            logger.debug("Transformers and PyTorch imported successfully")
        except ImportError as e:
            raise ImportError(
                "Transformers and PyTorch are required for abstractive summarization. "
                "Install with: pip install transformers torch"
            ) from e


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
        self.device = None  # Will be set lazily when torch is imported
        self._model_loaded = False
    
    def _get_device(self):
        """
        Get the device (CPU) for model operations.
        Lazy initialization - only imports torch when needed.
        
        Returns:
            torch.device object
        """
        if self.device is None:
            _ensure_transformers_imported()  # This also imports torch
            self.device = torch.device("cpu")
        return self.device
    
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
        Load model from local directory using full precision (FP32).
        This is the fallback method if quantization fails or isn't available.
        
        Returns:
            True if successful, False otherwise
        """
        _ensure_transformers_imported()  # Ensure transformers is imported
        model_path = self._get_model_path()
        
        # Check if model directory exists
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        # Check if it's actually a directory
        if not model_path.is_dir():
            logger.error(f"Model path is not a directory: {model_path}")
            return False
        
        try:
            logger.info(f"Loading model from local path: {model_path}")
            
            # Load tokenizer first - this is usually fast
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True
                )
                logger.debug("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {str(e)}")
                return False
            
            # Load the actual model - this is the memory-intensive part
            # We use FP32 (float32) for CPU compatibility
            # low_cpu_mem_usage helps reduce memory spikes during loading
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    torch_dtype=torch.float32,  # Use FP32 for CPU (more compatible)
                    low_cpu_mem_usage=True  # Reduces memory usage during loading
                )
                logger.debug("Model loaded from disk")
            except RuntimeError as e:
                # RuntimeError often means out of memory
                logger.error(f"Failed to load model (possibly out of memory): {str(e)}")
                self.tokenizer = None  # Clean up tokenizer if model load failed
                return False
            except Exception as e:
                logger.error(f"Unexpected error loading model: {str(e)}")
                self.tokenizer = None
                return False
            
            # Move model to CPU (should already be there, but be explicit)
            try:
                device = self._get_device()  # Get device (lazy initialization)
                self.model.to(device)
                self.model.eval()  # Set to evaluation mode (no gradients needed)
                logger.info("Model loaded successfully and set to eval mode")
            except Exception as e:
                logger.error(f"Error moving model to device: {str(e)}")
                # Clean up
                del self.model
                self.model = None
                self.tokenizer = None
                return False
            
            return True
            
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.error(f"Unexpected error in _load_from_local: {str(e)}", exc_info=True)
            # Clean up any partial state
            self.model = None
            self.tokenizer = None
            return False
    
    def _try_quantization(self) -> bool:
        """Try to load model with INT8 quantization."""
        _ensure_transformers_imported()  # Ensure transformers is imported
        """
        Try to load model with INT8 quantization for reduced memory usage.
        Quantization reduces model size and memory requirements but may slightly
        reduce quality. This is optional - we fall back to FP32 if it fails.
        
        Returns:
            True if quantization successful, False otherwise
        """
        try:
            # Check if bitsandbytes is available
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.info("bitsandbytes not available, skipping INT8 quantization")
            return False
        
        try:
            logger.info("Attempting INT8 quantization with bitsandbytes")
            
            # Configure quantization - INT8 uses 8 bits instead of 32 bits per weight
            # This reduces memory usage by ~75% but may slightly reduce quality
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0  # Threshold for quantization
            )
            
            model_path = self._get_model_path()
            
            # Load tokenizer first (same as FP32 loading)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True
                )
            except Exception as e:
                logger.error(f"Failed to load tokenizer for quantization: {str(e)}")
                return False
            
            # Load model with quantization
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    quantization_config=quantization_config,
                    device_map="cpu",  # Explicitly use CPU
                    low_cpu_mem_usage=True
                )
            except RuntimeError as e:
                # Often means quantization isn't supported or out of memory
                logger.warning(f"INT8 quantization failed (runtime error): {str(e)}")
                self.tokenizer = None  # Clean up
                return False
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {str(e)}")
                self.tokenizer = None
                return False
            
            # Set to eval mode
            try:
                self.model.eval()
                logger.info("Model loaded successfully with INT8 quantization")
                return True
            except Exception as e:
                logger.warning(f"Error setting quantized model to eval mode: {str(e)}")
                del self.model
                self.model = None
                self.tokenizer = None
                return False
            
        except Exception as e:
            # Catch any other unexpected errors
            logger.warning(f"Unexpected error during quantization: {str(e)}, falling back to FP32")
            # Clean up
            self.model = None
            self.tokenizer = None
            return False
    
    def load_model(self, use_quantization: bool = True) -> bool:
        """
        Main entry point for loading the model.
        Tries quantization first if requested, then falls back to FP32.
        
        Args:
            use_quantization: Whether to attempt INT8 quantization first
            
        Returns:
            True if successful, False otherwise
        """
        # If model is already loaded, we're done
        if self._model_loaded:
            logger.info("Model already loaded, skipping reload")
            return True
        
        # Validate that model files exist before attempting to load
        if not self._model_exists_locally():
            error_msg = (
                f"Model not found locally at {self._get_model_path()}. "
                "Please run scripts/download_models.py first to download the model."
            )
            logger.error(error_msg)
            return False
        
        # Try quantization first if requested (saves memory)
        # If it fails, we'll fall back to FP32
        if use_quantization:
            logger.info("Attempting to load model with quantization...")
            if self._try_quantization():
                self._model_loaded = True
                logger.info("Model loaded successfully with quantization")
                return True
            else:
                logger.info("Quantization failed or unavailable, trying FP32 loading...")
        
        # Fallback to regular FP32 loading (more compatible, uses more memory)
        if self._load_from_local():
            self._model_loaded = True
            logger.info("Model loaded successfully in FP32 mode")
            return True
        
        # If we get here, both methods failed
        logger.error("Failed to load model with both quantization and FP32 methods")
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
        """
        Unload model from memory to free up RAM.
        This is useful when you're done with summarization and want to free memory.
        The model can be reloaded later if needed.
        """
        try:
            # Delete model first (it's the biggest memory consumer)
            if self.model is not None:
                del self.model
                self.model = None
                logger.debug("Model deleted from memory")
            
            # Delete tokenizer (smaller but still takes some memory)
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.debug("Tokenizer deleted from memory")
            
            # Update state
            self._model_loaded = False
            
            # Force Python's garbage collector to actually free the memory
            # This is important because PyTorch models can hold onto memory
            import gc
            gc.collect()
            
            logger.info("Model unloaded from memory successfully")
        except Exception as e:
            logger.error(f"Error during model unloading: {str(e)}")
            # Still mark as unloaded even if cleanup had issues
            self._model_loaded = False

