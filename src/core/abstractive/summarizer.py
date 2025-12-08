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
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=60,  # Enough for 1-2 sentences
                    min_length=25,  # Ensure at least one complete sentence
                    num_beams=4,  # Better quality for overview
                    length_penalty=0.7,  # Keep it concise
                    repetition_penalty=1.2,  # Avoid repetition
                    early_stopping=True,
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
            # T5 models expect a prefix to know what task to do
            # T5 is trained on "summarize:" prefix, but we can try variations
            # Different summary types will be handled with prompts + post-processing
            summary_type = summary_type.lower() if summary_type else "short"
            
            # Use prompts that T5 might understand better
            # T5 was trained on various summarization tasks, so we try task-specific prefixes
            if summary_type == "short":
                # For short: try to get concise output
                input_text = f"summarize briefly: {text}"
            elif summary_type == "long":
                # For long: try to get detailed output
                input_text = f"summarize in detail: {text}"
            else:  # medium
                # Standard abstractive summary prefix
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
            
            # Generate the summary - this is the expensive operation
            # We use torch.no_grad() to save memory since we're not training
            _ensure_torch_imported()  # Ensure torch is imported
            try:
                # Adjust repetition penalty based on summary type for better quality
                # Short summaries need less repetition control, long need more
                rep_penalty = Config.REPETITION_PENALTY
                if summary_type == "short":
                    rep_penalty = 1.2  # Less aggressive for short summaries
                elif summary_type == "long":
                    rep_penalty = 1.4  # More aggressive for long summaries
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=Config.NUM_BEAMS,  # More beams = better quality but slower
                        length_penalty=length_penalty,  # Use provided length penalty (adjusted for preset)
                        repetition_penalty=rep_penalty,  # Reduce repetition (adjusted per type)
                        temperature=Config.TEMPERATURE if Config.DO_SAMPLE else 1.0,
                        do_sample=Config.DO_SAMPLE,  # Use sampling or deterministic
                        early_stopping=True,  # Stop when we find a good summary
                        no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,  # Avoid repeating phrases
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
                if not summary or len(summary) < 10:
                    logger.warning("Generated summary is too short, using original text")
                    return text
                
                return summary
            except Exception as e:
                logger.error(f"Failed to decode model output: {str(e)}")
                return text
            
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.error(f"Unexpected error in abstractive summarization: {str(e)}", exc_info=True)
            return text  # Return original text as fallback
    
    def _format_summary(self, summary: str, preset: str) -> str:
        """
        Format the summary based on the preset type according to professional requirements.
        Parses and reformats the model output to match exact specifications.
        
        Args:
            summary: Raw summary text from model
            preset: 'short', 'medium', or 'long'
            
        Returns:
            Formatted summary text matching professional requirements
        """
        if not summary or not summary.strip():
            return summary
        
        preset = preset.lower() if preset else "short"
        
        if preset == "short":
            # Short: Extremely concise - 1 sentence purpose + 2-3 key points only
            # Remove minor details, examples, background information
            sentences = self.preprocessor.tokenize_sentences(summary)
            
            if len(sentences) >= 2:
                # First sentence is the main purpose
                main_purpose = sentences[0].strip()
                
                # Extract 2-3 key points from remaining sentences
                # Filter out sentences that are too short or seem like details/examples
                key_points = []
                for sentence in sentences[1:]:
                    sentence = sentence.strip()
                    # Skip very short sentences, questions, or sentences with examples
                    if (len(sentence) > 15 and 
                        not sentence.startswith(('For example', 'For instance', 'Such as', 'Like')) and
                        '?' not in sentence):
                        key_points.append(sentence)
                        if len(key_points) >= 3:  # Maximum 3 key points
                            break
                
                # Format: Main purpose + 2-3 key points
                formatted = f"{main_purpose}\n\n"
                if key_points:
                    formatted += "Key Points:\n"
                    for point in key_points:
                        formatted += f"• {point}\n"
                
                return formatted.strip()
            else:
                # If only one sentence, return as main purpose
                return summary.strip()
        
        elif preset == "medium":
            # Medium: 6-10 sentences or 4-8 bullet points, structured, groups related points
            sentences = self.preprocessor.tokenize_sentences(summary)
            
            # Determine if we should use bullets or sentences
            # Use bullets if we have clear distinct points, sentences if narrative flow
            if len(sentences) >= 4:
                # Format as bullet points (4-8 bullets)
                # Group related sentences together
                bullet_points = []
                current_group = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:
                        # Remove existing bullet markers
                        sentence = sentence.lstrip('•-* ').strip()
                        
                        # Group short related sentences
                        if len(sentence) < 80 and current_group:
                            # Combine with previous if both are short
                            if len(current_group[-1]) < 100:
                                current_group[-1] += f" {sentence}"
                            else:
                                current_group.append(sentence)
                        else:
                            if current_group:
                                bullet_points.extend([f"• {s}" for s in current_group])
                                current_group = []
                            current_group.append(sentence)
                        
                        # Limit to 8 bullets maximum
                        if len(bullet_points) + len(current_group) >= 8:
                            break
                
                # Add remaining group
                if current_group:
                    bullet_points.extend([f"• {s}" for s in current_group])
                
                # Ensure we have 4-8 bullets
                if len(bullet_points) < 4 and len(sentences) > len(bullet_points):
                    # Add more from remaining sentences
                    remaining = sentences[len(bullet_points):]
                    for sentence in remaining[:8-len(bullet_points)]:
                        sentence = sentence.strip().lstrip('•-* ').strip()
                        if sentence and len(sentence) > 10:
                            bullet_points.append(f"• {sentence}")
                
                if bullet_points:
                    return "\n".join(bullet_points[:8])  # Maximum 8 bullets
            
            # Fallback: format as 6-10 sentences
            formatted_sentences = sentences[:10]  # Maximum 10 sentences
            return ' '.join(formatted_sentences)
        
        elif preset == "long":
            # Long: Executive summary with Purpose + Key Insights (4-7 bullets) + Detailed Summary
            sentences = self.preprocessor.tokenize_sentences(summary)
            
            formatted = "Executive Summary\n\n"
            
            # Initialize key_insights to avoid UnboundLocalError
            key_insights = []
            
            # 1. Purpose - one strong sentence describing what the document is about
            if sentences:
                purpose = sentences[0].strip()
                # Ensure it's a strong, complete sentence
                if not purpose.endswith(('.', '!', '?')):
                    purpose += "."
                formatted += f"**Purpose:** {purpose}\n\n"
            
            # 2. Key Insights - 4-7 bullet points covering main arguments, findings, or themes
            if len(sentences) > 1:
                formatted += "**Key Insights:**\n"
                # Extract 4-7 key points from remaining sentences
                # Prioritize longer, more substantial sentences
                for sentence in sentences[1:]:
                    sentence = sentence.strip().lstrip('•-* ').strip()
                    if sentence and len(sentence) > 20:  # Substantial sentences only
                        key_insights.append(sentence)
                        if len(key_insights) >= 7:  # Maximum 7 insights
                            break
                
                # Ensure we have at least 4 if possible
                if len(key_insights) < 4 and len(sentences) > len(key_insights) + 1:
                    remaining = sentences[len(key_insights) + 1:]
                    for sentence in remaining[:4-len(key_insights)]:
                        sentence = sentence.strip().lstrip('•-* ').strip()
                        if sentence and len(sentence) > 15:
                            key_insights.append(sentence)
                
                # Format as bullets
                for insight in key_insights[:7]:  # Maximum 7
                    formatted += f"• {insight}\n"
                
                formatted += "\n"
            
            # 3. Detailed Summary - clear, coherent paragraph combining important details
            if len(sentences) > len(key_insights) + 1:
                detailed_sentences = sentences[len(key_insights) + 1:]
                detailed_text = ' '.join(detailed_sentences)
                formatted += f"**Detailed Summary:**\n{detailed_text}"
            elif len(sentences) == 1:
                # If only one sentence, use it as detailed summary too
                formatted += f"**Detailed Summary:**\n{sentences[0]}"
            elif len(key_insights) == 0:
                # If no key insights were extracted, use all remaining sentences as detailed summary
                if len(sentences) > 1:
                    detailed_text = ' '.join(sentences[1:])
                    formatted += f"**Detailed Summary:**\n{detailed_text}"
            
            return formatted.strip()
        
        else:  # fallback
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
            cleaned_text = self.preprocessor.clean_text(text)
            if not cleaned_text or not cleaned_text.strip():
                logger.warning("Text became empty after cleaning")
                return ""
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            cleaned_text = text  # Use original if cleaning fails
        
        # Generate document overview first (brief description of what the document is about)
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
        
        # If text fits in one go, summarize directly (faster and better quality)
        if not chunk or estimated_tokens <= Config.MAX_INPUT_TOKENS:
            logger.info(f"Summarizing text directly ({estimated_tokens} estimated tokens)")
            summary = self._summarize_chunk(cleaned_text, max_length, min_length, length_penalty, preset)
            # Format the summary based on preset
            formatted_summary = self._format_summary(summary, preset)
            # Combine overview and summary
            return self._combine_overview_and_summary(overview, formatted_summary)
        
        # For long texts, we need to chunk and summarize each piece
        # Then combine the summaries, and possibly summarize again if still too long
        logger.info(f"Text is long ({estimated_tokens} tokens), chunking before summarization")
        
        try:
            # Split into manageable chunks with overlap to preserve context
            # Overlap helps maintain coherence between chunks
            chunks = self.preprocessor.chunk_text(
                cleaned_text,
                max_tokens=Config.MAX_INPUT_TOKENS,
                overlap=100  # Increased overlap for better context preservation
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
        
        logger.info(f"Summarizing {len(chunks)} chunks sequentially")
        
        # Summarize each chunk - this can take a while for long documents
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_summary = self._summarize_chunk(chunk, max_length, min_length, length_penalty, preset)
                
                # Only add non-empty summaries
                if chunk_summary and chunk_summary.strip():
                    summaries.append(chunk_summary.strip())
                else:
                    logger.warning(f"Chunk {i+1} produced empty summary")
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                # Continue with other chunks even if one fails
                continue
        
        # If no chunks produced summaries, we're in trouble
        if not summaries:
            logger.error("All chunks failed to produce summaries")
            return cleaned_text  # Return cleaned original as last resort
        
        # Combine all chunk summaries into one
        combined_summary = ' '.join(summaries)
        
        # If the combined summary is still too long, summarize it again
        # This creates a "summary of summaries" which is often more coherent
        try:
            combined_tokens = self.preprocessor.estimate_tokens(combined_summary)
            if combined_tokens > Config.MAX_INPUT_TOKENS:
                logger.info("Combined summary is still long, creating final summary")
                combined_summary = self._summarize_chunk(combined_summary, max_length, min_length, length_penalty, preset)
        except Exception as e:
            logger.error(f"Error in final summarization step: {str(e)}")
            # Return what we have even if final step failed
        
        # Format the combined summary based on preset
        formatted_summary = self._format_summary(combined_summary, preset)
        
        # Combine overview and formatted summary
        return self._combine_overview_and_summary(overview, formatted_summary)
    
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


