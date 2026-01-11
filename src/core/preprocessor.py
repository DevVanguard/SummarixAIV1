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
        # Try to find punkt tokenizer (old format)
        punkt_found = False
        try:
            nltk.data.find(f'tokenizers/punkt/{self.language}.pickle')
            punkt_found = True
        except LookupError:
            pass
        
        # Try to find punkt_tab tokenizer (new format in NLTK 3.8.1+)
        punkt_tab_found = False
        try:
            nltk.data.find(f'tokenizers/punkt_tab/{self.language}.pickle')
            punkt_tab_found = True
        except LookupError:
            pass
        
        # Download punkt_tab if neither format is found (punkt_tab is the newer format)
        if not punkt_found and not punkt_tab_found:
            logger.info(f"Downloading NLTK punkt_tab tokenizer for {self.language}")
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download punkt_tab, trying punkt: {str(e)}")
                # Fallback to old punkt if punkt_tab fails
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e2:
                    logger.error(f"Failed to download punkt tokenizer: {str(e2)}")
        
        # Download stopwords
        try:
            nltk.data.find(f'corpora/stopwords/{self.language}.zip')
        except LookupError:
            logger.info(f"Downloading NLTK stopwords for {self.language}")
            nltk.download('stopwords', quiet=True)
    
    def clean_academic_text(self, text: str) -> str:
        """
        Clean academic paper text - remove citations, LaTeX, URLs, special tokens, metadata.
        
        Args:
            text: Raw academic text
            
        Returns:
            Cleaned text suitable for summarization
        """
        if not text:
            return ""
        
        # Step 1: Fix common PDF encoding issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        text = text.replace('ﬃ', 'ffi')
        text = text.replace('ﬄ', 'ffl')
        text = text.replace('–', '-')  # en-dash to hyphen
        text = text.replace('—', '-')  # em-dash to hyphen
        text = text.replace(''', "'")  # smart quote to regular
        text = text.replace(''', "'")
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        
        # Step 2: Remove special model tokens (BERT, etc.)
        text = re.sub(r'\[CLS\]|\[SEP\]|\[MASK\]|\[PAD\]|\[UNK\]', '', text)
        
        # Step 3: Remove email addresses (including group formats like {user1,user2}@domain.com)
        text = re.sub(r'\{[^}]*\}@\S+', '', text)  # {user1,user2}@domain
        text = re.sub(r'\S+@\S+\.[A-Za-z]{2,}', '', text)  # regular emails
        
        # Step 4: Remove inline citations (various formats)
        # Pattern: (Author et al., YYYY) or (Author, YYYY) or (Author YYYY)
        text = re.sub(r'\([A-Z][a-zA-Z\s]+et al\.\s*,?\s*\d{4}[a-z]?\)', '', text)
        text = re.sub(r'\([A-Z][a-zA-Z\s]+\s*,?\s*\d{4}[a-z]?\)', '', text)
        text = re.sub(r'\([A-Z][a-zA-Z]+\s+and\s+[A-Z][a-zA-Z]+\s*,?\s*\d{4}[a-z]?\)', '', text)
        # Pattern: [1], [2,3], [1-5]
        text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)
        # Pattern: superscript numbers (citation markers)
        text = re.sub(r'\s*\d+\s*(?=[.,;:])', '', text)
        
        # Step 5: Remove title/author header patterns
        # Remove lines with "Abstract" alone
        text = re.sub(r'\n\s*Abstract\s*\n', '\n', text, flags=re.IGNORECASE)
        # Remove affiliation patterns
        text = re.sub(r'(?:University|Institute|Department|Lab|Laboratory) of [A-Za-z\s]+', '', text)
        
        # Step 6: Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Step 7: Remove LaTeX commands and symbols
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{arg}
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # \command
        text = re.sub(r'\$[^$]+\$', '', text)  # Inline math $...$
        text = re.sub(r'\\\(.*?\\\)', '', text)  # Math \(...\)
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)  # Math \[...\]
        
        # Step 8: Remove figure/table references and captions
        text = re.sub(r'(?:Figure|Fig\.|Table|Tab\.)\s*\d+[:\.]?[^\n]*', '', text, flags=re.IGNORECASE)
        
        # Step 9: Remove section numbering patterns
        text = re.sub(r'^\s*\d+(?:\.\d+)*\s+[A-Z]', lambda m: m.group(0).split()[-1], text, flags=re.MULTILINE)
        
        # Step 10: Remove common academic artifacts
        text = re.sub(r'(?:arXiv|DOI):\s*\S+', '', text)
        
        # Step 11: Remove mathematical notation remnants
        text = re.sub(r'[∈∉∀∃∧∨¬⊕⊗⊙∑∏∫]', '', text)
        text = re.sub(r'∈R[A-Z]', '', text)  # Like ∈RH
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove repetitive headers, footers, page numbers, and references.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with boilerplate removed
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        if len(lines) < 3:
            return text
        
        # Remove common header/footer patterns (page numbers, document titles repeated)
        # Remove lines that appear multiple times (likely headers/footers)
        line_counts = {}
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) > 5 and len(line_stripped) < 100:  # Reasonable length for headers/footers
                line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
        
        # Remove lines that appear more than 2 times (likely headers/footers)
        repeated_lines = {line for line, count in line_counts.items() if count > 2}
        
        # Also remove page number patterns and document identifiers
        page_patterns = [
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^Page\s+\d+',  # "Page 1", "Page 2", etc.
            r'^\d+\s+/\s+\d+$',  # "1 / 10" format
            r'\d{1,3}\s+(?:Five Year Plan|Year Plan)\s+\d+\s+Information and Communication Technologies',  # Document identifiers
            r'\d{1,3}\s+(?:Five Year Plan|Year Plan).*Information and Communication Technologies',  # Document identifiers (variation)
        ]
        
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip if it's a repeated header/footer
            if line_stripped in repeated_lines:
                continue
            # Skip if it matches page number patterns
            if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in page_patterns):
                continue
            # Skip very short lines that might be artifacts
            if len(line_stripped) < 3:
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text, with special handling for PDF artifacts.
        Removes boilerplate content (headers, footers, references).
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Remove boilerplate (headers, footers, page numbers)
        text = self.remove_boilerplate(text)
        
        # Step 2: Clean academic artifacts (citations, LaTeX, etc.)
        text = self.clean_academic_text(text)
        
        # Step 3: Remove reference sections (bibliography, works cited, etc.)
        # Look for common reference section headers
        ref_patterns = [
            r'(?i)(?:References|Bibliography|Works Cited|Works Consulted|Sources).*',
            r'(?i)(?:Appendix|Appendices).*',
        ]
        # Split by paragraphs and filter out reference sections
        paragraphs = text.split('\n\n')
        filtered_paragraphs = []
        skip_mode = False
        for para in paragraphs:
            para_stripped = para.strip()
            # Check if this paragraph starts a reference section
            if any(re.match(pattern, para_stripped[:50]) for pattern in ref_patterns):
                skip_mode = True
                continue
            # If we're in skip mode, skip short paragraphs (likely reference entries)
            if skip_mode and len(para_stripped.split()) < 10:
                continue
            if para_stripped:
                filtered_paragraphs.append(para)
        text = '\n\n'.join(filtered_paragraphs)
        
        # Step 4: Fix any remaining hyphenated line breaks
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Step 5: Normalize line breaks - preserve paragraph breaks (double newline)
        # but join lines within paragraphs
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines to spaces
        text = re.sub(r'\n{2,}', '\n\n', text)  # Multiple newlines to double
        
        # Step 6: Remove extra whitespace
        text = re.sub(r' +', ' ', text)
        
        # Step 7: Remove special characters but keep punctuation and newlines
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\n]', ' ', text)
        
        # Step 8: Clean up whitespace around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([^\s\d])', r'\1 \2', text)  # Add space after punctuation if missing
        
        # Step 9: Final whitespace normalization
        text = re.sub(r' +', ' ', text)
        
        # Step 10: Clean up line breaks - remove leading/trailing spaces on each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        
        # Step 11: Strip leading/trailing whitespace
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
        except LookupError as e:
            # NLTK resource missing - try to download it
            logger.warning(f"NLTK resource missing: {str(e)}")
            try:
                logger.info("Attempting to download missing NLTK resources...")
                nltk.download('punkt_tab', quiet=True)
                # Try again after download
                sentences = sent_tokenize(text, language=self.language)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                return sentences
            except Exception as e2:
                logger.error(f"Failed to download NLTK resources: {str(e2)}")
                # Fallback: simple sentence splitting
                return self._fallback_sentence_split(text)
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {str(e)}")
            # Fallback: simple sentence splitting
            return self._fallback_sentence_split(text)
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """
        Fallback sentence splitting when NLTK is unavailable.
        Uses simple regex-based splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        import re
        # Split on sentence-ending punctuation followed by space or end of string
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)
        # Combine punctuation with previous sentence
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i + 1]).strip()
            else:
                sentence = sentences[i].strip()
            if sentence and len(sentence) > 10:
                result.append(sentence)
        # If no sentences found, return the whole text
        if not result:
            return [text.strip()] if text.strip() else []
        return result
    
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

