"""
Abstractive summarization module using quantized T5-small model.
This module handles generating summaries by creating new text rather than just extracting sentences.
We use beam search and various parameters to improve summary quality.
"""

import logging
from typing import List, Optional

# Lazy import of torch - only needed when actually using abstractive mode
torch = None

from src.core.abstractive.model_loader import ModelLoader
from src.core.preprocessor import TextPreprocessor
from src.utils.config import Config

logger = logging.getLogger(__name__)

def _ensure_torch_imported():
    """Lazy import of torch - only when needed."""
    global torch
    if torch is None:
        try:
            import torch
            logger.debug("PyTorch imported successfully")
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for abstractive summarization. "
                "Install with: pip install torch"
            ) from e

class AbstractiveSummarizer:
    """Abstractive summarization using T5-small model optimized for CPU."""
    
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
        self.skip_overview = True  # Skip document overview for CPU speed
    
    def initialize(self) -> bool:
        """
        Initialize and load the model with CPU optimizations.
        
        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True
        
        logger.info("Initializing abstractive summarizer...")
        
        # Import torch for CPU optimizations
        _ensure_torch_imported()
        
        # CPU optimization: Set number of threads
        try:
            torch.set_num_threads(2)  # Use 2 threads for better CPU performance
            logger.info("Set torch to use 2 CPU threads")
        except Exception as e:
            logger.warning(f"Could not set thread count: {e}")
        
        if not self.model_loader.load_model(use_quantization=self.use_quantization):
            logger.error("Failed to load model")
            return False
        
        self.model, self.tokenizer = self.model_loader.get_model()
        
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer is None")
            return False
        
        # Ensure model is in eval mode
        self.model.eval()
        
        self._initialized = True
        logger.info("Abstractive summarizer initialized successfully")
        return True
    
    def _generate_document_overview(self, text: str) -> str:
        """
        Generate a brief, professional overview of what the document is about.
        This provides context before the main summary.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Brief 1-2 sentence professional overview of the document
        """
        if not text or not text.strip():
            return ""
        
        # Use a representative sample - first 800 chars to get better context
        # Also try to get text from middle if document is long
        if len(text) > 1000:
            # Use beginning and middle for better overview
            sample_text = text[:400] + " " + text[len(text)//2:len(text)//2+400]
        else:
            sample_text = text
        
        try:
            # Use a prompt that encourages a brief overview
            input_text = f"summarize: {sample_text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=min(300, Config.MAX_INPUT_TOKENS),  # Enough context for overview
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate brief, professional overview (40-60 tokens for 1-2 sentences)
            _ensure_torch_imported()
            with torch.inference_mode():  # Faster than no_grad() for inference
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=60,  # Enough for 1-2 sentences
                    min_length=25,  # Ensure at least one complete sentence
                    num_beams=2,  # Reduced for CPU speed
                    length_penalty=0.7,  # Keep it concise
                    repetition_penalty=1.2,  # Avoid repetition
                    early_stopping=True,
                    use_cache=Config.USE_CACHE,  # Enable KV cache
                )
            
            overview = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Clean up and ensure professional formatting
            # Remove any leading/trailing quotes or extra punctuation
            overview = overview.strip('"\'.,;:')
            
            # Ensure it's a proper sentence
            if overview and not overview.endswith(('.', '!', '?')):
                overview += "."
            
            # Limit to 1-2 sentences maximum
            sentences = self.preprocessor.tokenize_sentences(overview)
            if len(sentences) > 2:
                overview = '. '.join(sentences[:2])
                if not overview.endswith('.'):
                    overview += '.'
            
            return overview
            
        except Exception as e:
            logger.error(f"Error generating document overview: {str(e)}")
            return ""  # Return empty if overview generation fails
    
    def _summarize_chunk(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None,
        length_penalty: float = None,
        summary_type: str = "short"
    ) -> str:
        """
        Summarize a single text chunk using the T5 model.
        This is where the actual magic happens - we feed text to the model and get a summary back.
        
        Args:
            text: Text chunk to summarize (should be cleaned and preprocessed)
            max_length: Maximum summary length in tokens (defaults to config)
            min_length: Minimum summary length in tokens (defaults to config)
            length_penalty: Length penalty for generation (defaults to config)
            summary_type: Type of summary - 'short', 'medium', or 'long'
            
        Returns:
            Summarized text, or original text if summarization fails
        """
        # Make sure model is loaded before we try to use it
        if not self._initialized:
            try:
                if not self.initialize():
                    logger.error("Failed to initialize model for summarization")
                    return text  # Can't summarize without a model
            except Exception as e:
                logger.error(f"Exception during model initialization: {str(e)}")
                return text
        
        # Validate input - empty text means no summary
        if not text or not text.strip():
            logger.warning("Attempted to summarize empty text")
            return ""
        
        # Use config defaults if not specified
        max_length = max_length or Config.MAX_OUTPUT_TOKENS
        min_length = min_length or Config.MIN_SUMMARY_LENGTH
        length_penalty = length_penalty if length_penalty is not None else Config.LENGTH_PENALTY
        
        try:
            # T5 models expect a "summarize:" prefix
            # Use simple prompt - let length parameters control output size
            # Complex prompts can confuse the model and cause garbled output on CPU
            summary_type = summary_type.lower() if summary_type else "short"
            
            # Use simple, consistent prompt for all presets
            # T5-small performs better with simple prompts on CPU
            input_text = f"summarize: {text}"
            
            # Tokenize the input - convert text to numbers the model understands
            # We truncate if too long and pad if too short
            try:
                inputs = self.tokenizer(
                    input_text,
                    max_length=Config.MAX_INPUT_TOKENS,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
            except Exception as e:
                logger.error(f"Tokenization failed: {str(e)}")
                return text  # Can't proceed without tokenization
            
            # Generate the summary - optimized for CPU inference
            # We use torch.inference_mode() for better CPU performance
            _ensure_torch_imported()  # Ensure torch is imported
            try:
                # Quality-first generation parameters
                rep_penalty = Config.REPETITION_PENALTY  # Default is 1.3
                no_repeat = Config.NO_REPEAT_NGRAM_SIZE  # Default is 3
                
                with torch.inference_mode():  # Faster than no_grad() for inference-only
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=Config.NUM_BEAMS,  # Beam search for quality (slower but better)
                        length_penalty=length_penalty,  # Use provided length penalty
                        repetition_penalty=rep_penalty,  # Prevent repetition without stopping generation
                        temperature=Config.TEMPERATURE,
                        do_sample=Config.DO_SAMPLE,  # Deterministic beam search
                        early_stopping=Config.EARLY_STOPPING,  # Stop when beams converge
                        no_repeat_ngram_size=no_repeat,  # Avoid repeating phrases
                        use_cache=Config.USE_CACHE,  # Enable KV cache for speed
                    )
            except RuntimeError as e:
                # This usually means out of memory or model error
                logger.error(f"Model generation failed (possibly out of memory): {str(e)}")
                return text
            except Exception as e:
                logger.error(f"Unexpected error during generation: {str(e)}")
                return text
            
            # Convert the model's output (numbers) back into readable text
            try:
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                summary = summary.strip()
                
                # Basic validation - make sure we got something reasonable
                if not summary:
                    logger.warning("Generated summary is empty, using original text")
                    return text
                
                if len(summary) < 10:
                    logger.warning(f"Generated summary is very short ({len(summary)} chars), but returning it")
                    # Still return it - might be valid for very short inputs
                
                logger.info(f"Generated summary: {len(summary)} chars, {len(summary.split())} words")
                logger.info(f"Summary preview: {summary[:200]}...")
                return summary
            except Exception as e:
                logger.error(f"Failed to decode model output: {str(e)}")
                return text
            
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.error(f"Unexpected error in abstractive summarization: {str(e)}", exc_info=True)
            return text  # Return original text as fallback
    
    def _clean_model_output(self, text: str) -> str:
        """
        Clean up model output - remove fragments, garbage, fix encoding issues.
        Aggressive garbage detection while preserving valid content.
        
        Args:
            text: Raw model output
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Store original for fallback
        original_text = text
        
        # Remove markdown formatting
        text = text.replace('**', '')  # Remove bold markers
        text = text.replace('*', '')    # Remove italic markers
        
        # Clean up bullet points but keep the content
        text = text.replace('• ', '')
        text = text.replace('- ', '')
        
        # Fix common encoding issues from PDF
        text = text.replace('â€TM', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('&#160;', ' ')
        text = text.replace('&nbsp;', ' ')
        text = text.replace('»', '')  # Remove special quotation marks
        
        # Remove HTML entities and garbage patterns
        import re
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        text = re.sub(r'&#\d+;', '', text)
        
        # Remove garbage patterns that T5 generates on CPU
        # Pattern like: of)"'s  or  the)"  or  a)'s
        text = re.sub(r'\)\s*["\']\'*s\s*$', '', text)  # Ends with )"'s or )"'
        text = re.sub(r'\)\s*["\']\'*\s*$', '', text)   # Ends with )" or )'
        text = re.sub(r'\s+\)\s*["\']\'*s*\b', ' ', text)  # Middle of text: )"'s
        
        # Remove incomplete parentheses/brackets at end
        text = re.sub(r'\([^)]*$', '', text)  # Unclosed ( at end
        text = re.sub(r'\[[^\]]*$', '', text)  # Unclosed [ at end
        
        # Split into sentences for filtering
        try:
            sentences = self.preprocessor.tokenize_sentences(text)
        except:
            # If tokenization fails, return original without markdown
            return original_text.replace('**', '').replace('*', '')
        
        if not sentences:
            return original_text.replace('**', '').replace('*', '')
        
        clean_sentences = []
        
        # Known garbage patterns - specific to T5 CPU errors
        # Be conservative - only remove obvious garbage
        garbage_patterns = [
            r'click here',
            r'http[s]?://',
            r'www\.',
            r'\.{5,}',  # 5+ dots in a row
            r"ICT'do\?",  # Malformed text
            r'companies which.*?are two.*?»',  # Specific garbage pattern from earlier
            r'many u\.s\. infrastructure such',
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter sentences
            words = sentence.split()
            
            # Skip very short fragments only
            if len(words) < 3:
                continue
            
            # Skip sentences matching SPECIFIC garbage patterns only
            is_garbage = False
            for pattern in garbage_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    logger.debug(f"Filtering garbage: {sentence[:50]}...")
                    is_garbage = True
                    break
            
            if is_garbage:
                continue
            
            # Skip sentences with excessive special chars (very high threshold)
            special_chars = sum(1 for c in sentence if c in '?&#;@$%^*[]{}|\\')
            if special_chars > 8:
                continue
            
            # Skip if almost no alphabetic content (very low threshold)
            alpha_chars = sum(c.isalpha() for c in sentence)
            if len(sentence) > 0 and alpha_chars < len(sentence) * 0.3:
                continue
            
            clean_sentences.append(sentence)
        
        # If cleaning removed everything, return original without markdown
        if not clean_sentences:
            logger.warning("Cleaning removed all sentences, returning original")
            return original_text.replace('**', '').replace('*', '')
        
        result = ' '.join(clean_sentences)
        logger.info(f"After basic cleaning: {len(clean_sentences)} sentences, {len(result)} chars")
        
        # Final pass: Remove incomplete sentence at the end ONLY if it's clearly incomplete
        # Remove it if it doesn't end with punctuation AND is short (< 4 words)
        if result:
            sentences_final = self.preprocessor.tokenize_sentences(result)
            if sentences_final:
                last_sentence = sentences_final[-1].strip()
                # Check if last sentence is incomplete (no ending punctuation AND very short)
                if last_sentence and (last_sentence[-1] not in '.!?') and len(last_sentence.split()) < 4:
                    logger.info(f"Removing incomplete last sentence: {last_sentence[:50]}...")
                    sentences_final = sentences_final[:-1]
                    result = ' '.join(sentences_final)
                    logger.info(f"After removing incomplete sentence: {len(sentences_final)} sentences remaining")
        
        logger.info(f"Clean model output final: {len(result)} chars")
        return result
    
    def _format_summary(self, summary: str, preset: str, skip_cleaning: bool = False) -> str:
        """
        Format the summary based on the preset type according to professional requirements.
        Parses and reformats the model output to match exact specifications.
        No markdown formatting - plain text only.
        
        Args:
            summary: Summary text (already cleaned if skip_cleaning=True)
            preset: 'short', 'medium', or 'long'
            skip_cleaning: If True, skip the cleaning step (already done)
            
        Returns:
            Formatted summary text matching professional requirements
        """
        if not summary or not summary.strip():
            return summary
        
        # Clean the model output if not already done
        if not skip_cleaning:
            cleaned_summary = self._clean_model_output(summary)
            
            # If cleaning removed everything, use original
            if not cleaned_summary or not cleaned_summary.strip():
                logger.warning("Cleaned summary is empty, using original")
                cleaned_summary = summary.replace('**', '').replace('*', '')
            
            if not cleaned_summary:
                return summary  # Return original as last resort
            
            summary = cleaned_summary
        
        preset = preset.lower() if preset else "short"
        
        if preset == "short":
            # Short: Concise plain text - 1 main sentence + 2-3 key points
            sentences = self.preprocessor.tokenize_sentences(summary)
            
            if len(sentences) >= 2:
                # First sentence is the main purpose
                main_purpose = sentences[0].strip()
                
                # Extract 2-3 key points from remaining sentences
                key_points = []
                for sentence in sentences[1:]:
                    sentence = sentence.strip()
                    # Skip very short sentences or examples
                    if (len(sentence) > 20 and 
                        not sentence.startswith(('For example', 'For instance', 'Such as', 'Like')) and
                        '?' not in sentence):
                        key_points.append(sentence)
                        if len(key_points) >= 3:
                            break
                
                # Format: Plain text, no markdown
                formatted = f"{main_purpose}\n\n"
                if key_points:
                    formatted += "Key Points:\n"
                    for i, point in enumerate(key_points, 1):
                        formatted += f"{i}. {point}\n"
                
                return formatted.strip()
            else:
                return summary.strip()
        
        elif preset == "medium":
            # Medium: 6-10 sentences in plain text, natural flow
            sentences = self.preprocessor.tokenize_sentences(summary)
            
            if len(sentences) >= 4:
                # Select 6-10 best sentences
                selected_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20:
                        selected_sentences.append(sentence)
                        if len(selected_sentences) >= 10:
                            break
                
                # Format as natural paragraph with proper spacing
                if selected_sentences:
                    return ' '.join(selected_sentences)
            
            # Fallback: use all sentences up to 10
            return ' '.join(sentences[:10])
        
        elif preset == "long":
            # Long: Simple paragraph format - no complex formatting for CPU stability
            sentences = self.preprocessor.tokenize_sentences(summary)
            logger.info(f"Formatting LONG preset: {len(sentences)} sentences to format")
            
            if not sentences:
                return summary
            
            # Import re for pattern matching
            import re
            
            # Filter out only OBVIOUS garbage - be very conservative
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                words = sentence.split()
                
                # Skip only extremely short fragments
                if len(words) < 3:
                    continue
                
                # Skip only specific known garbage patterns
                garbage_indicators = ['»', '...........', 'click here', 'http://', 'www.', ')"\'']
                if any(indicator in sentence.lower() for indicator in garbage_indicators):
                    continue
                
                # Skip if sentence ends with malformed text like "of)"
                if re.search(r'\w+\)\s*["\']?\'?s?\s*$', sentence):
                    logger.debug(f"Skipping malformed sentence: {sentence[:50]}...")
                    continue
                
                clean_sentences.append(sentence)
            
            # If we have clean sentences, use them; otherwise use all
            if clean_sentences:
                sentences = clean_sentences
                logger.info(f"After garbage filtering: {len(clean_sentences)} clean sentences")
            else:
                sentences = [s.strip() for s in sentences if s.strip()]
                logger.info(f"No filtering applied, using all {len(sentences)} sentences")
            
            # Simple format: Just join sentences into paragraphs
            # Group into paragraphs of 3-4 sentences each for readability
            paragraphs = []
            current_para = []
            
            for sentence in sentences:
                current_para.append(sentence)
                if len(current_para) >= 3:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            
            # Add remaining sentences
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            logger.info(f"Created {len(paragraphs)} paragraphs from {len(sentences)} sentences")
            
            # Join paragraphs with double newline
            if paragraphs:
                result = '\n\n'.join(paragraphs)
                logger.info(f"Final formatted output: {len(result)} chars")
                return result
            else:
                result = ' '.join(sentences)
                logger.info(f"Final formatted output (no paragraphs): {len(result)} chars")
                return result
        
        else:  # fallback - just return clean text
            return summary.strip()
    
    def summarize(
        self,
        text: str,
        max_length: int = None,
        min_length: int = None,
        length_penalty: float = None,
        preset: str = "medium",
        chunk: bool = True
    ) -> str:
        """
        Main entry point for generating abstractive summaries.
        Handles both short texts (direct summarization) and long texts (chunked summarization).
        Generates a document overview first, then the main summary based on preset.
        
        Args:
            text: Input text to summarize (raw text from PDF)
            max_length: Maximum summary length in tokens (optional override)
            min_length: Minimum summary length in tokens (optional override)
            length_penalty: Length penalty for generation (optional override)
            preset: Summary preset - 'short', 'medium', or 'long' (default: 'medium')
            chunk: Whether to split long texts into chunks (recommended for long documents)
            
        Returns:
            Formatted summary with overview, or empty string if input is invalid
        """
        # Validate input early - no point processing empty text
        if not text or not text.strip():
            logger.warning("Attempted to summarize empty or None text")
            return ""
        
        # Make sure we have a working model before doing anything
        if not self._initialized:
            try:
                if not self.initialize():
                    logger.error("Cannot summarize: model initialization failed")
                    return text  # Return original as fallback
            except Exception as e:
                logger.error(f"Exception during model initialization: {str(e)}")
                return text
        
        # Clean up the text - remove extra whitespace, normalize, etc.
        try:
            logger.info(f"Starting text cleaning, input length: {len(text)} chars")
            cleaned_text = self.preprocessor.clean_text(text)
            if not cleaned_text or not cleaned_text.strip():
                logger.warning("Text became empty after cleaning")
                return ""
            logger.info(f"Text cleaned successfully, output length: {len(cleaned_text)} chars")
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            cleaned_text = text  # Use original if cleaning fails
        
        # Skip document overview for CPU performance (saves 50% inference time)
        # Overview generation requires a second model inference which is slow on CPU
        if self.skip_overview:
            overview = ""
            logger.info("Skipping document overview for CPU performance")
        else:
            # Generate document overview (optional, adds extra inference time)
            logger.info("Generating document overview...")
            try:
                overview = self._generate_document_overview(cleaned_text)
            except Exception as e:
                logger.warning(f"Failed to generate overview: {str(e)}")
                overview = ""
        
        # Normalize preset value
        preset = preset.lower() if preset else "medium"
        
        # Check if we need to chunk the text
        # Models have token limits, so long documents need to be split
        try:
            estimated_tokens = self.preprocessor.estimate_tokens(cleaned_text)
        except Exception as e:
            logger.error(f"Token estimation failed: {str(e)}")
            estimated_tokens = len(cleaned_text) // 4  # Rough fallback estimate
        
        # ADAPTIVE OUTPUT LENGTH: Calculate based on input size
        # For longer documents, generate proportionally longer summaries
        # Rough estimate: 1 page = ~500 tokens, aim for ~100-150 tokens per page of summary
        if estimated_tokens < 500:
            # Very short document (< 1 page)
            adaptive_max = 150
            adaptive_min = 50
        elif estimated_tokens < 2000:
            # Short document (1-4 pages)
            adaptive_max = 300
            adaptive_min = 100
        elif estimated_tokens < 5000:
            # Medium document (4-10 pages)  
            adaptive_max = 500
            adaptive_min = 200
        else:
            # Long document (10+ pages)
            adaptive_max = 800
            adaptive_min = 300
        
        # Override with adaptive lengths instead of preset-based
        if max_length is None:
            max_length = adaptive_max
        if min_length is None:
            min_length = adaptive_min
        
        logger.info(f"Adaptive summary length: min={min_length}, max={max_length} tokens for {estimated_tokens} input tokens")
        logger.info(f"chunk={chunk}, estimated_tokens={estimated_tokens}, MAX_INPUT_TOKENS={Config.MAX_INPUT_TOKENS}")
        
        # If text fits in one go, summarize directly (faster and better quality)
        if not chunk or estimated_tokens <= Config.MAX_INPUT_TOKENS:
            logger.info(f"Taking SINGLE-CHUNK path: Summarizing text directly ({estimated_tokens} estimated tokens)")
            summary = self._summarize_chunk(cleaned_text, max_length, min_length, length_penalty, preset)
            logger.info(f"Single-chunk summary generated: {len(summary)} chars")
            
            # Safety check - ensure we have output
            if not summary or not summary.strip():
                logger.error("Chunk summarization produced empty result")
                return cleaned_text[:1000]  # Return first 1000 chars as fallback
            
            # Format the summary based on preset
            logger.info(f"Formatting summary (preset={preset})...")
            formatted_summary = self._format_summary(summary, preset)
            logger.info(f"Formatted summary: {len(formatted_summary)} chars")
            
            # Final safety check
            if not formatted_summary or not formatted_summary.strip():
                logger.warning("Formatting produced empty result, using raw summary")
                formatted_summary = summary
            
            # Combine overview and summary
            result = self._combine_overview_and_summary(overview, formatted_summary)
            
            # Absolute final check
            if not result or not result.strip():
                logger.error("Final result is empty, returning raw summary")
                return summary if summary else cleaned_text[:1000]
            
            return result
        
        # CPU optimization: For very long texts, pre-extract using fast extractive method
        # Only do this for extremely long documents to preserve content
        if estimated_tokens > Config.MAX_INPUT_TOKENS * 2.5:
            logger.info(f"Text is very long ({estimated_tokens} tokens), pre-extracting key content for CPU")
            try:
                # Use extractive summarization to get ~50% of content (less aggressive)
                from src.core.extractive.textrank import TextRankSummarizer
                extractor = TextRankSummarizer(ratio=0.5)
                cleaned_text = extractor.summarize(cleaned_text)
                estimated_tokens = self.preprocessor.estimate_tokens(cleaned_text)
                logger.info(f"Pre-extraction reduced to {estimated_tokens} tokens")
            except Exception as e:
                logger.warning(f"Pre-extraction failed: {str(e)}, continuing with full text")
        
        # For long texts, we need to chunk and summarize each piece
        # CPU optimization: Use fewer, larger chunks to reduce number of inferences
        logger.info(f"Taking MULTI-CHUNK path: Text is long ({estimated_tokens} tokens), chunking before summarization")
        
        try:
            # Split into manageable chunks with minimal overlap for CPU speed
            # Use 90% of max tokens per chunk to reduce number of chunks
            chunk_size = int(Config.MAX_INPUT_TOKENS * 0.9)
            chunks = self.preprocessor.chunk_text(
                cleaned_text,
                max_tokens=chunk_size,
                overlap=30  # Minimal overlap for CPU speed
            )
        except Exception as e:
            logger.error(f"Text chunking failed: {str(e)}")
            # Fallback: try to summarize the whole thing anyway
            summary = self._summarize_chunk(cleaned_text, max_length, min_length, length_penalty, preset)
            formatted_summary = self._format_summary(summary, preset)
            return self._combine_overview_and_summary(overview, formatted_summary)
        
        if not chunks:
            logger.warning("Chunking produced no chunks, using original text")
            return cleaned_text
        
        # Quality first: Process more chunks for comprehensive coverage
        # User prioritizes quality over speed
        if preset == "long":
            max_chunks_cpu = 15  # Process up to 15 chunks for very comprehensive summary
        else:
            max_chunks_cpu = 8  # Process up to 8 chunks for short/medium
        
        if len(chunks) > max_chunks_cpu:
            logger.info(f"Limiting to first {max_chunks_cpu} chunks (of {len(chunks)}) for CPU performance")
            chunks = chunks[:max_chunks_cpu]
        
        logger.info(f"Summarizing {len(chunks)} chunks sequentially")
        
        # Summarize each chunk - this can take a while for long documents
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                chunk_summary = self._summarize_chunk(chunk, max_length, min_length, length_penalty, preset)
                
                # Clean each chunk summary individually
                if chunk_summary and chunk_summary.strip():
                    cleaned_chunk = self._clean_model_output(chunk_summary)
                    if cleaned_chunk and cleaned_chunk.strip():
                        summaries.append(cleaned_chunk.strip())
                        logger.info(f"Chunk {i+1} summary: {len(cleaned_chunk)} chars, {len(cleaned_chunk.split())} words")
                    else:
                        logger.warning(f"Chunk {i+1} cleaning removed all content")
                else:
                    logger.warning(f"Chunk {i+1} produced empty summary")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                # Continue with other chunks even if one fails
                continue
        
        logger.info(f"Collected {len(summaries)} summaries from {len(chunks)} chunks")
        
        # If no chunks produced summaries, we're in trouble
        if not summaries:
            logger.error("All chunks failed to produce summaries")
            return cleaned_text  # Return cleaned original as last resort
        
        # Combine all chunk summaries into one
        combined_summary = ' '.join(summaries)
        logger.info(f"Combined summary: {len(combined_summary)} chars, {len(combined_summary.split())} words")
        
        # Safety check
        if not combined_summary or not combined_summary.strip():
            logger.error("Combined summary is empty")
            return cleaned_text[:1000]  # Return first 1000 chars as fallback
        
        # If the combined summary is still too long, summarize it again
        # This creates a "summary of summaries" which is often more coherent
        # BUT: Skip this for "long" preset - we want to preserve all chunk content
        try:
            combined_tokens = self.preprocessor.estimate_tokens(combined_summary)
            if combined_tokens > Config.MAX_INPUT_TOKENS and preset != "long":
                logger.info("Combined summary is still long, creating final summary")
                final_summary = self._summarize_chunk(combined_summary, max_length, min_length, length_penalty, preset)
                if final_summary and final_summary.strip():
                    combined_summary = final_summary
            elif preset == "long":
                logger.info(f"Skipping final summarization for 'long' preset - preserving {combined_tokens} tokens")
        except Exception as e:
            logger.error(f"Error in final summarization step: {str(e)}")
            # Return what we have even if final step failed
        
        # Format the combined summary based on preset
        # Skip cleaning since we already cleaned each chunk individually
        formatted_summary = self._format_summary(combined_summary, preset, skip_cleaning=True)
        logger.info(f"After formatting: {len(formatted_summary)} chars, {len(formatted_summary.split())} words")
        
        # Safety check after formatting
        if not formatted_summary or not formatted_summary.strip():
            logger.warning("Formatting produced empty result, using combined summary")
            formatted_summary = combined_summary
        
        # Combine overview and formatted summary
        result = self._combine_overview_and_summary(overview, formatted_summary)
        
        # Final safety check
        if not result or not result.strip():
            logger.error("Final result is empty, returning combined summary")
            return combined_summary if combined_summary else cleaned_text[:1000]
        
        return result
    
    def _combine_overview_and_summary(self, overview: str, summary: str) -> str:
        """
        Combine document overview with the main summary in a professional format.
        
        Args:
            overview: Brief document overview
            summary: Main summary text
            
        Returns:
            Combined formatted text
        """
        if not summary:
            return overview if overview else ""
        
        if overview:
            # Combine with proper formatting
            return f"{overview}\n\n{summary}"
        else:
            return summary
    
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



