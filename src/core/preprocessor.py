"""Text preprocessing module for cleaning and chunking."""

import logging
import re
from typing import List, Optional

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    raise ImportError(
        "NLTK is required. Install it with: pip install nltk"
    )

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handle text cleaning, tokenization, and chunking."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize text preprocessor.
        
        Args:
            language: Language for tokenization (default: 'english')
        """
        self.language = language
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words(language))
    
    def _ensure_nltk_data(self) -> None:
        """Download required NLTK data if not present."""
        try:
            nltk.data.find(f'tokenizers/punkt/{self.language}.pickle')
        except LookupError:
            logger.info(f"Downloading NLTK punkt tokenizer for {self.language}")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find(f'corpora/stopwords/{self.language}.zip')
        except LookupError:
            logger.info(f"Downloading NLTK stopwords for {self.language}")
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text, language=self.language)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {str(e)}")
            return [text]  # Fallback to original text
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of words
        """
        if not text:
            return []
        
        try:
            words = word_tokenize(text.lower(), language=self.language)
            # Remove stopwords and non-alphabetic tokens
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            return words
        except Exception as e:
            logger.error(f"Error tokenizing words: {str(e)}")
            return []
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def chunk_text(
        self,
        text: str,
        max_tokens: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Estimate total tokens
        total_tokens = self.estimate_tokens(cleaned_text)
        
        if total_tokens <= max_tokens:
            return [cleaned_text]
        
        # Split into sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        if not sentences:
            return [cleaned_text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self.estimate_tokens(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def preprocess(self, text: str, chunk: bool = False, max_tokens: int = 512) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text to preprocess
            chunk: Whether to chunk the text
            max_tokens: Maximum tokens per chunk if chunking
            
        Returns:
            List of preprocessed text chunks (or single item if not chunking)
        """
        cleaned = self.clean_text(text)
        
        if chunk:
            return self.chunk_text(cleaned, max_tokens=max_tokens)
        else:
            return [cleaned]

