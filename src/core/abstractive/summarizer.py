"""Abstractive summarization using quantized T5-small."""

import logging
from typing import List, Optional

import torch

from src.core.abstractive.model_loader import ModelLoader
from src.core.preprocessor import TextPreprocessor
from src.utils.config import Config

logger = logging.getLogger(__name__)


class AbstractiveSummarizer:
    """Abstractive summarization using T5-small model."""
    
    def __init__(self, use_quantization: bool = True):
        """
        Initialize abstractive summarizer.
        
        Args:
            use_quantization: Whether to use quantized model
        """
        self.use_quantization = use_quantization
        self.model_loader = ModelLoader()
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize and load the model.
        
        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True
        
        logger.info("Initializing abstractive summarizer...")
        
        if not self.model_loader.load_model(use_quantization=self.use_quantization):
            logger.error("Failed to load model")
            return False
        
        self.model, self.tokenizer = self.model_loader.get_model()
        
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer is None")
            return False
        
        self._initialized = True
        logger.info("Abstractive summarizer initialized successfully")
        return True
    
    def _summarize_chunk(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None
    ) -> str:
        """
        Summarize a single text chunk.
        
        Args:
            text: Text chunk to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Summarized text
        """
        if not self._initialized:
            if not self.initialize():
                return text  # Return original if initialization fails
        
        if not text or not text.strip():
            return ""
        
        max_length = max_length or Config.MAX_OUTPUT_TOKENS
        min_length = min_length or Config.MIN_SUMMARY_LENGTH
        
        try:
            # Prepare input
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=Config.MAX_INPUT_TOKENS,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=Config.NUM_BEAMS,
                    temperature=Config.TEMPERATURE,
                    do_sample=Config.DO_SAMPLE,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            
            # Decode
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {str(e)}")
            return text  # Return original on error
    
    def summarize(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None,
        chunk: bool = True
    ) -> str:
        """
        Generate abstractive summary.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            chunk: Whether to chunk long texts
            
        Returns:
            Summarized text
        """
        if not text or not text.strip():
            return ""
        
        # Initialize if needed
        if not self._initialized:
            if not self.initialize():
                logger.error("Cannot summarize: model not initialized")
                return text
        
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Estimate tokens
        estimated_tokens = self.preprocessor.estimate_tokens(cleaned_text)
        
        # If text is short enough, summarize directly
        if not chunk or estimated_tokens <= Config.MAX_INPUT_TOKENS:
            return self._summarize_chunk(cleaned_text, max_length, min_length)
        
        # Chunk and summarize each chunk
        chunks = self.preprocessor.chunk_text(
            cleaned_text,
            max_tokens=Config.MAX_INPUT_TOKENS,
            overlap=50
        )
        
        if not chunks:
            return cleaned_text
        
        logger.info(f"Summarizing {len(chunks)} chunks")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_summary = self._summarize_chunk(chunk, max_length, min_length)
            if chunk_summary:
                summaries.append(chunk_summary)
        
        # Combine summaries
        combined_summary = ' '.join(summaries)
        
        # If combined summary is still too long, summarize again
        if self.preprocessor.estimate_tokens(combined_summary) > Config.MAX_INPUT_TOKENS:
            logger.info("Combined summary too long, summarizing again")
            combined_summary = self._summarize_chunk(combined_summary, max_length, min_length)
        
        return combined_summary
    
    def is_ready(self) -> bool:
        """Check if summarizer is ready to use."""
        return self._initialized
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model_loader:
            self.model_loader.unload_model()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        logger.info("Abstractive summarizer cleaned up")

