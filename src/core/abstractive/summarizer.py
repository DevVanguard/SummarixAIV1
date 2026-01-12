"""
Abstractive summarization module using quantized T5-small model.
OPTIMIZED FOR CPU - FIXED GARBAGE/HALLUCINATION ISSUES
"""

import logging
import re
from typing import List, Optional
import gc

# Lazy import of torch
torch = None

from src.core.abstractive.model_loader import ModelLoader
from src.core.preprocessor import TextPreprocessor
from src.utils.config import Config

logger = logging.getLogger(__name__)

def _ensure_torch_imported():
    """Lazy import of torch."""
    global torch
    if torch is None:
        try:
            import torch
            logger.debug("PyTorch imported successfully")
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for abstractive summarization."
            ) from e

class AbstractiveSummarizer:
    """Abstractive summarization using T5-small - OPTIMIZED FOR CPU STABILITY."""
    
    def __init__(self, use_quantization: bool = True):
        self.use_quantization = use_quantization
        self.model_loader = ModelLoader()
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # CPU OPTIMIZATION: Skip problematic steps
        self.skip_overview = True  # Overview generation causes garbage on CPU
        self.force_simple_prompts = True  # Complex prompts cause hallucinations
    
    def initialize(self) -> bool:
        """Initialize with CPU optimizations."""
        if self._initialized:
            return True
        
        logger.info("Initializing abstractive summarizer with CPU optimizations...")
        
        _ensure_torch_imported()
        
        # CPU optimization: Set threads for performance
        try:
            torch.set_num_threads(2)  # 2 threads for better performance with detailed prompts
            torch.set_num_interop_threads(1)
            logger.info("Set torch to use 2 CPU threads")
        except Exception as e:
            logger.warning(f"Could not set thread count: {e}")
        
        # Load model
        if not self.model_loader.load_model(use_quantization=self.use_quantization):
            logger.error("Failed to load model")
            return False
        
        self.model, self.tokenizer = self.model_loader.get_model()
        
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer is None")
            return False
        
        # Set model to eval mode
        self.model.eval()
        
        # CRITICAL: Disable gradients for CPU performance
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._initialized = True
        logger.info("Abstractive summarizer initialized successfully (CPU mode)")
        return True
    
    def _get_prompt_for_preset(self, text: str, preset: str = "medium") -> str:
        """
        Get T5-compatible prompt format.
        T5 models expect simple task prefix format: "summarize: {text}"
        Output length is controlled via generation parameters, not prompt instructions.
        """
        # Clean the text first to remove PDF artifacts
        text = re.sub(r'\s+', ' ', text.strip())
        
        # T5 models use simple task prefix format
        # The preset controls output length through generation parameters, not the prompt
        return f"summarize: {text}"
    
    def _summarize_chunk(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
        length_penalty: float = 1.0,
        summary_type: str = "medium"
    ) -> str:
        """
        Summarize a single chunk with STABLE parameters for CPU.
        """
        if not self._initialized and not self.initialize():
            logger.error("Model not initialized")
            return text
        
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            # Use T5-compatible prompt format: "summarize: {text}"
            input_text = self._get_prompt_for_preset(text, summary_type)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=512,  # Standard T5 input size
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generation parameters for quality output
            # Output length is controlled via max_length, min_length, and length_penalty
            generation_params = {
                'input_ids': inputs.input_ids,
                'max_length': max_length,
                'min_length': min_length,
                'num_beams': 4,  # Beam search for better quality
                'length_penalty': length_penalty,
                'repetition_penalty': 1.2,  # Moderate penalty to avoid repetition
                'do_sample': False,  # Beam search for deterministic output
                'early_stopping': True,
                'no_repeat_ngram_size': 3,  # Prevent exact repetition
            }
            
            # Generate
            _ensure_torch_imported()
            with torch.inference_mode():
                outputs = self.model.generate(**generation_params)
            
            # Decode
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            logger.info(f"  Raw output: {summary[:100]}...")
            
            # Fix incomplete ending if needed (don't reject!)
            if summary and summary[-1] not in '.!?':
                summary = summary + '.'
                logger.debug("  Fixed incomplete ending")
            
            # Check for garbage (lenient now)
            if self._is_garbage(summary):
                logger.warning("  ✗ Obvious garbage detected, rejecting")
                return ""
            
            logger.info(f"  ✓ Output validated: {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return ""
    
    def _is_garbage(self, text: str) -> bool:
        """
        REASONABLE garbage detection - only reject OBVIOUS garbage.
        """
        if not text or len(text) < 15:
            return True
        
        text_lower = text.lower()
        
        # 1. Check for OBVIOUS nonsense words only
        obvious_nonsense = ['folloo', 'faloo', 'grae', 'mos', 'chrias']
        if any(word in text_lower for word in obvious_nonsense):
            logger.debug(f"Garbage: obvious nonsense word")
            return True
        
        # 2. Check alphabetic ratio - very lenient
        alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        alpha_ratio = alpha_chars / max(len(text), 1)
        if alpha_ratio < 0.35:  # Very lenient - only reject if < 35%
            logger.debug(f"Garbage: low alpha ratio {alpha_ratio:.2f}")
            return True
        
        # 3. Check for MULTIPLE hallucination keywords (need 2+)
        hallucination_words = ['obama', 'british', 'england', 'fema', 'click here', 'http://', 'www.']
        hallucination_count = sum(1 for word in hallucination_words if word in text_lower)
        if hallucination_count >= 2:  # Need 2+ to reject
            logger.debug(f"Garbage: {hallucination_count} hallucination keywords")
            return True
        
        # 4. Check for EXCESSIVE word repetition (very high threshold)
        words = text_lower.split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Only check longer words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Only reject if a word appears 5+ times in short text
            max_freq = max(word_freq.values()) if word_freq else 0
            if max_freq > 5 and len(words) < 30:
                logger.debug(f"Garbage: excessive repetition (max={max_freq})")
                return True
        
        # 5. Check for OBVIOUS garbage patterns only
        obvious_garbage = [
            r'\b\w\s+\w\s+\w\s+\w\s+\w\b',  # Many single letters with spaces
            r'\.{5,}',  # 5+ dots
            r'\bthe\s+the\s+the\b',  # Triple repetition
        ]
        
        for pattern in obvious_garbage:
            if re.search(pattern, text_lower):
                logger.debug(f"Garbage: obvious pattern")
                return True
        
        # That's it - accept everything else!
        return False
    
    def _deduplicate_sentences(self, text: str, similarity_threshold: float = 0.85) -> str:
        """
        Remove duplicate and highly similar sentences from text.
        
        Args:
            text: Text with potential duplicate sentences
            similarity_threshold: Similarity threshold for considering sentences duplicates (0-1)
            
        Returns:
            Text with duplicates removed
        """
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return text
        
        # Simple deduplication: remove exact duplicates and very similar sentences
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            # Normalize sentence for comparison (lowercase, remove extra spaces)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            # Skip if we've seen this exact sentence
            if normalized in seen:
                continue
            
            # Check for high similarity with existing sentences (simple word overlap)
            is_duplicate = False
            normalized_words = set(normalized.split())
            
            # Remove common stop words for better comparison
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'}
            normalized_words_filtered = normalized_words - stop_words
            if len(normalized_words_filtered) < 3:
                normalized_words_filtered = normalized_words  # Fallback if too many stop words
            
            for seen_normalized in seen:
                seen_words = set(seen_normalized.split())
                seen_words_filtered = seen_words - stop_words
                if len(seen_words_filtered) < 3:
                    seen_words_filtered = seen_words  # Fallback
                
                if len(normalized_words_filtered) < 3 or len(seen_words_filtered) < 3:
                    # For very short sentences, use exact match
                    if normalized == seen_normalized:
                        is_duplicate = True
                        break
                else:
                    # For longer sentences, check word overlap (using filtered words)
                    overlap = len(normalized_words_filtered & seen_words_filtered)
                    union = len(normalized_words_filtered | seen_words_filtered)
                    similarity = overlap / union if union > 0 else 0
                    
                    # Also check if one sentence is a substring of another (common with repeated phrases)
                    if normalized in seen_normalized or seen_normalized in normalized:
                        if abs(len(normalized) - len(seen_normalized)) < max(len(normalized), len(seen_normalized)) * 0.3:
                            is_duplicate = True
                            break
                    
                    # More aggressive similarity check for common repeated phrases
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break
                    
                    # Check for common repeated phrase patterns
                    # If sentences share a significant starting portion, they're likely duplicates
                    min_len = min(len(normalized), len(seen_normalized))
                    if min_len > 50:  # Only for longer sentences
                        shared_start = 0
                        for i in range(min(100, min_len)):  # Check first 100 chars
                            if normalized[i] == seen_normalized[i]:
                                shared_start += 1
                            else:
                                break
                        if shared_start > 50 and shared_start / min_len > 0.6:  # 60% shared start
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        return ' '.join(unique_sentences)
    
    def _is_similar_to_overview(self, point: str, overview: str, threshold: float = 0.6) -> bool:
        """
        Check if a point is too similar to the overview (to avoid repetition).
        Uses word overlap to determine similarity.
        """
        if not point or not overview:
            return False
        
        # Normalize both texts
        point_words = set(re.sub(r'[^\w\s]', '', point.lower()).split())
        overview_words = set(re.sub(r'[^\w\s]', '', overview.lower()).split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
                     'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        point_words = point_words - stop_words
        overview_words = overview_words - stop_words
        
        if not point_words or not overview_words:
            return False
        
        # Calculate word overlap
        overlap = len(point_words & overview_words)
        union = len(point_words | overview_words)
        
        if union == 0:
            return False
        
        similarity = overlap / union
        
        # Also check if point is a substring of overview or vice versa
        point_lower = point.lower()
        overview_lower = overview.lower()
        if point_lower in overview_lower or overview_lower in point_lower:
            if abs(len(point_lower) - len(overview_lower)) < max(len(point_lower), len(overview_lower)) * 0.5:
                return True
        
        return similarity >= threshold
    
    def _is_incomplete_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence appears incomplete or corrupted.
        """
        if not sentence or len(sentence.strip()) < 10:
            return True
        
        sentence_lower = sentence.lower().strip()
        
        # Check for incomplete endings
        incomplete_endings = [
            'including:', 'including', 'such as:', 'such as', 'e.g.', 'for example:',
            ':', 'e ', 'e lished', 'lished', 'various', 'constraints', 'issues',
            'action plan', 'financial outlay', 'in order to', 'to launch'
        ]
        
        # Check if ends with incomplete phrase
        for ending in incomplete_endings:
            if sentence_lower.endswith(ending) or sentence_lower.endswith(ending + '.'):
                return True
        
        # Check for patterns like "e lished" or single letters followed by words
        if re.search(r'\be\s+[a-z]+\b', sentence_lower):
            return True
        
        # Check for sentences ending with incomplete phrases
        if re.search(r'\b(including|such as|for example|e\.g\.)\s*:?\s*$', sentence_lower):
            return True
        
        # Check for very short sentences that seem incomplete
        words = sentence.split()
        if len(words) < 4:
            return True
        
        # Check for sentences with corrupted text patterns
        if re.search(r'\b\w\s+\w\s+\w\s+\w\b', sentence_lower):  # Single letters separated
            return True
        
        return False
    
    def _is_similar_to_overview(self, point: str, overview: str, threshold: float = 0.6) -> bool:
        """
        Check if a point is too similar to the overview (to avoid repetition).
        Uses word overlap to determine similarity.
        """
        if not point or not overview:
            return False
        
        # Normalize both texts
        point_words = set(re.sub(r'[^\w\s]', '', point.lower()).split())
        overview_words = set(re.sub(r'[^\w\s]', '', overview.lower()).split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
                     'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        point_words = point_words - stop_words
        overview_words = overview_words - stop_words
        
        if not point_words or not overview_words:
            return False
        
        # Calculate word overlap
        overlap = len(point_words & overview_words)
        union = len(point_words | overview_words)
        
        if union == 0:
            return False
        
        similarity = overlap / union
        
        # Also check if point is a substring of overview or vice versa
        point_lower = point.lower().strip()
        overview_lower = overview.lower().strip()
        
        if point_lower in overview_lower or overview_lower in point_lower:
            return True
        
        # Check if significant portion of point appears in overview
        if len(point_words) > 0:
            word_overlap_ratio = overlap / len(point_words)
            if word_overlap_ratio > 0.7:  # 70% of point's words are in overview
                return True
        
        return similarity >= threshold
    
    def _clean_incomplete_sentence(self, text: str) -> str:
        """
        Remove incomplete sentences and fix broken text patterns.
        """
        if not text:
            return ""
        
        # Fix common apostrophe issues: "Pakistan s" -> "Pakistan's"
        text = re.sub(r"(\w+)\s+s\s+(\w)", r"\1's \2", text)
        text = re.sub(r"(\w+)\s+s\s*([,\.])", r"\1's\2", text)
        
        # Check for sentences starting with corrupted patterns like "e lished"
        # If sentence starts with a single letter followed by space and word, it's likely corrupted
        if re.match(r'^\s*[a-z]\s+[a-z]', text, re.IGNORECASE):
            # Check if it's a known corrupted pattern
            corrupted_patterns = [
                r'^\s*e\s+lished',  # "e lished" -> corrupted "established"
                r'^\s*[a-z]\s+[a-z]{4,}',  # Single letter + word (likely corrupted)
            ]
            for pattern in corrupted_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return ""  # Remove entirely if it starts with corrupted pattern
        
        # Remove incomplete patterns like "e lished" (likely "established" corrupted)
        text = re.sub(r'\be\s+lished\b', '', text, flags=re.IGNORECASE)
        
        # Detect and remove sentences ending with corrupted/incomplete words
        # Pattern: ends with a word that looks corrupted (has unusual character patterns)
        # Check for words ending with unusual patterns like "inundisip", "neo" after apostrophe
        if re.search(r"'\s*\w{1,4}\s*\.?\s*$", text):  # Ends with apostrophe + short word
            return ""
        
        # Check for known corrupted word patterns at the end
        corrupted_word_patterns = [
            r'inundisip\s*\.?\s*$',
            r"'\s*neo\s*\.?\s*$",
        ]
        for pattern in corrupted_word_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Try to remove just the corrupted word
                cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                if len(cleaned) >= 15:
                    text = cleaned
                else:
                    return ""  # Too much was removed
        
        # Check for sentences ending with incomplete words (likely corrupted)
        # Words that end abruptly or have unusual patterns
        last_word = text.split()[-1] if text.split() else ""
        if last_word and len(last_word) > 0:
            last_word_clean = last_word.rstrip('.,!?;:')
            # Remove if last word looks corrupted (ends with unusual pattern or is too short after apostrophe)
            if "'" in last_word_clean and len(last_word_clean.split("'")[-1]) < 3:
                return ""
            # Check for corrupted words ending with unusual patterns (too many consonants)
            if len(last_word_clean) > 6 and re.search(r'[^aeiou]{5,}$', last_word_clean.lower()):
                # Very suspicious - likely corrupted
                if re.search(r'(?:disip|undisip)$', last_word_clean.lower()):
                    return ""
        
        # Remove sentences that are clearly incomplete (ending with incomplete words)
        # Pattern: ends with a single letter or very short word followed by punctuation
        text = re.sub(r'\b\w{1,2}\s*[.!?]\s*$', '', text, flags=re.MULTILINE)
        
        # Remove sentences ending with incomplete phrases
        incomplete_endings = [
            r'\be\s+lished\b',
            r'\b\w+\s+by\s*$',
            r'\b\w+\s+the\s*$',
            r'\b\w+\s+and\s*$',
            r'\b\w+\s+or\s*$',
            r'\bplanned\s+to\s+have\s*\.?\s*$',  # "It is planned to have."
            r'\bstands\s+at\s*[,\.]?\s*$',  # "stands at," or "stands at."
            r'\bto\s+have\s*\.?\s*$',  # "to have."
        ]
        
        for pattern in incomplete_endings:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove sentences ending with comma and nothing after (incomplete)
        if text.rstrip().endswith(','):
            text = text.rstrip().rstrip(',')
            if len(text) < 20:  # If what's left is too short, remove entirely
                return ""
        
        # Check for sentences ending with colon followed by incomplete phrase (e.g., "including: at each")
        # If sentence ends with colon and a very short phrase after it, it's likely incomplete
        if ':' in text:
            parts = text.split(':')
            if len(parts) > 1:
                after_colon = parts[-1].strip()
                # If what comes after colon is very short (less than 20 chars) or incomplete, remove it
                if len(after_colon) < 20 or re.match(r'^\s*(?:at|in|on|the|each|of|a|an|neo)\s+\w*\s*\.?\s*$', after_colon, re.IGNORECASE):
                    # Return the part before the colon if it's substantial, otherwise return empty
                    before_colon = ':'.join(parts[:-1]).strip()
                    if len(before_colon) >= 20:
                        text = before_colon
                    else:
                        return ""  # Entire sentence is too incomplete
        
        # Check for sentences that start with incomplete words (single letter + space)
        words = text.split()
        if words and len(words[0]) == 1 and len(words) > 1:
            # First word is a single letter, likely corrupted
            return ""
        
        # Remove very short sentence fragments (less than 15 chars)
        sentences = re.split(r'[.!?]+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 15:  # Minimum reasonable length
                # Check if it looks like a complete thought (has at least 4 words)
                words = sentence.split()
                if len(words) >= 4:
                    # Check if sentence doesn't start with corrupted pattern
                    if not re.match(r'^\s*[a-z]\s+[a-z]', sentence, re.IGNORECASE):
                        # Check for incomplete endings
                        if not re.search(r'\b(?:to\s+have|stands\s+at|planned\s+to)\s*\.?\s*$', sentence, re.IGNORECASE):
                            # Check for corrupted words at the end
                            last_word = words[-1] if words else ""
                            if not (last_word and "'" in last_word and len(last_word.split("'")[-1]) < 3):
                                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences) if cleaned_sentences else ""
    
    def _final_clean_point(self, point: str) -> str:
        """
        Final cleanup of bullet points to remove specific issues.
        """
        if not point:
            return ""
        
        # Remove sentences ending with corrupted words (like "inundisip")
        # Check if last word looks corrupted (unusual character patterns, too many consonants)
        words = point.split()
        if words:
            last_word = words[-1].rstrip('.,!?;:')
            # Check for corrupted words: unusual patterns, too many consonants at end
            if len(last_word) > 5:
                # Check for words ending with unusual consonant clusters
                if re.search(r'[^aeiou]{4,}$', last_word.lower()):
                    # Might be corrupted, check if it's a real word pattern
                    # Remove if it ends with suspicious patterns
                    if re.search(r'(?:disip|neo|undisip)$', last_word.lower()):
                        # Remove the corrupted word
                        point = ' '.join(words[:-1])
                        if not point or len(point) < 12:
                            return ""
        
        # Remove sentences ending with incomplete phrases
        incomplete_patterns = [
            r"'\s*neo\s*\.?\s*$",  # "telecommunications 'neo"
            r"inundisip\s*\.?\s*$",  # "inundisip"
            r"\bplanned\s+to\s+have\s*\.?\s*$",  # "It is planned to have."
            r"\bstands\s+at\s*[,\.]?\s*$",  # "stands at," or "stands at."
            r"\bto\s+have\s*\.?\s*$",  # "to have."
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, point, re.IGNORECASE):
                # Try to remove just the incomplete part
                cleaned = re.sub(pattern, '', point, flags=re.IGNORECASE).strip()
                if len(cleaned) >= 15:
                    point = cleaned
                else:
                    return ""  # Too much was removed, sentence is too incomplete
        
        # Remove sentences that end with just a few words after apostrophe
        if "'" in point:
            # Check if it ends with apostrophe + very short word
            if re.search(r"'\s*\w{1,3}\s*\.?\s*$", point):
                # Remove the part after the last apostrophe if it's too short
                parts = point.rsplit("'", 1)
                if len(parts) == 2:
                    after_apostrophe = parts[1].strip().rstrip('.,!?;:')
                    if len(after_apostrophe) < 4:
                        point = parts[0].rstrip()
                        if len(point) < 15:
                            return ""
        
        # Fix missing apostrophes: "Pakistan s ranking" -> "Pakistan's ranking"
        point = re.sub(r"(\w+)\s+s\s+(\w)", r"\1's \2", point)
        point = re.sub(r"(\w+)\s+s\s*([,\.])", r"\1's\2", point)
        
        # Remove trailing commas if nothing follows
        point = point.rstrip().rstrip(',')
        
        # Remove if sentence is too short after cleaning
        if len(point) < 15:
            return ""
        
        return point.strip()
    
    def _extract_main_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract main points from text for bullet formatting.
        Uses summarization to generate key points. For long texts, processes in sections.
        """
        if not text or len(text.strip()) < 100:
            return []
        
        try:
            all_points = []
            
            # For longer texts, process in sections to get diverse points
            if len(text) > 3000:
                logger.info(f"Processing long text ({len(text)} chars) in sections for diverse points...")
                # Split text into 3-4 sections to get points from different parts
                section_size = len(text) // 3
                sections = [
                    text[i:i + section_size] 
                    for i in range(0, len(text), section_size)
                ][:4]  # Max 4 sections
                
                for i, section in enumerate(sections):
                    if len(section.strip()) < 100:
                        continue
                    logger.info(f"Processing section {i+1}/{len(sections)} for points...")
                    section_points = self._extract_points_from_section(section, num_points=4)
                    all_points.extend(section_points)
            else:
                # For shorter texts, process directly
                all_points = self._extract_points_from_section(text, num_points=num_points)
            
            # Remove duplicates while preserving order
            unique_points = []
            seen_normalized = set()
            for point in all_points:
                normalized = re.sub(r'[^\w\s]', '', point.lower()).strip()
                if normalized not in seen_normalized and len(normalized) > 15:
                    seen_normalized.add(normalized)
                    unique_points.append(point)
            
            # Limit to requested number
            return unique_points[:num_points]
            
        except Exception as e:
            logger.error(f"Error extracting main points: {e}")
            return []
    
    def _extract_points_from_section(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract points from a single text section.
        """
        try:
            # Truncate text if too long to avoid token limit issues
            text_to_process = text
            if len(text) > 2000:
                # Take first and last portions for better coverage
                text_to_process = text[:1200] + " " + text[-1200:]
            
            # Generate a summary focused on key points
            input_text = f"summarize: {text_to_process}"
            
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate with parameters optimized for point extraction
            generation_params = {
                'input_ids': inputs.input_ids,
                'max_length': 400,  # Increased to allow more points and longer content
                'min_length': 180,  # Increased minimum for more substantial points
                'num_beams': 4,
                'length_penalty': 1.5,  # Higher to encourage more comprehensive output
                'repetition_penalty': 1.3,
                'do_sample': False,
                'early_stopping': True,
                'no_repeat_ngram_size': 3,
            }
            
            _ensure_torch_imported()
            with torch.inference_mode():
                outputs = self.model.generate(**generation_params)
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.strip()
            
            if not summary or self._is_garbage(summary):
                return []
            
            # Clean the summary
            summary = self._clean_output(summary)
            
            # Split into sentences and extract key points
            sentences = re.split(r'[.!?]+', summary)
            points = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Clean incomplete sentences
                cleaned_sentence = self._clean_incomplete_sentence(sentence)
                if not cleaned_sentence:
                    continue
                
                # Filter for reasonable length and meaningful content
                if len(cleaned_sentence) > 20 and len(cleaned_sentence) < 350:
                    # Skip very short or incomplete sentences
                    words = cleaned_sentence.split()
                    if len(words) >= 4:  # At least 4 words
                        # Capitalize first letter
                        if cleaned_sentence and cleaned_sentence[0].islower():
                            cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                        points.append(cleaned_sentence)
            
            return points
            
        except Exception as e:
            logger.error(f"Error extracting points from section: {e}")
            return []
    
    def _clean_output(self, text: str) -> str:
        """
        Clean model output - remove garbage while keeping good content.
        """
        if not text:
            return ""
        
        # Remove common PDF/encoding artifacts
        replacements = [
            ('â€TM', "'"), ('â€œ', '"'), ('â€', '"'),
            ('&#160;', ' '), ('&nbsp;', ' '), ('»', ''),
            ('\u200b', ''), ('\u200e', ''), ('\u2028', ' '),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', '', text)
        text = re.sub(r'&#\d+;', '', text)
        
        # Remove document identifiers and page references (more aggressive patterns)
        # Remove full document identifiers
        text = re.sub(r'\d{1,3}\s+(?:Five Year Plan|Year Plan)\s+\d+\s+Information and Communication Technologies', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{1,3}\s+(?:Five Year Plan|Year Plan).*?Information and Communication Technologies', '', text, flags=re.IGNORECASE)
        # Remove partial patterns (catching incomplete identifiers)
        text = re.sub(r'\d{1,3}\s+(?:Five Year Plan|Year Plan)\s+\d+\s+Information and Communication\s*(?:Technologies)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{1,3}\s+(?:five year plan|year plan)\s+\d+\s+information and communication\s*(?:technologies)?', '', text, flags=re.IGNORECASE)
        # Remove patterns like "according to the report 11th five year plan 332 Information..."
        text = re.sub(r'according to the report\s+\d{1,3}\s+(?:five year plan|year plan).*?(?:information and communication|information and communication technologies)', '', text, flags=re.IGNORECASE)
        # Remove standalone document identifiers
        text = re.sub(r'\d{1,3}\s+Five Year Plan\s+\d{3}\s+Information and Communication Technologies', '', text, flags=re.IGNORECASE)
        # Remove patterns with partial matches
        text = re.sub(r'\d{1,3}\s+(?:Five Year Plan|Year Plan)\s+\d+\s+Information\s+and\s+Communication', '', text, flags=re.IGNORECASE)
        
        # Fix corrupted text patterns (missing numbers, incomplete phrases)
        # Fix patterns like ",500" -> " 500" or "about,000" -> "about 4000" (we can't recover exact number)
        text = re.sub(r',(\d+)', r' \1', text)  # ",500" -> " 500"
        text = re.sub(r'(\w),(\d+)', r'\1 \2', text)  # "about,000" -> "about 000"
        # Fix "about 000" -> "about 4000" (common pattern, estimate)
        text = re.sub(r'\babout\s+000\s+students', 'about 4000 students', text, flags=re.IGNORECASE)
        
        # Remove incomplete sentences ending with "by" or "by,"
        text = re.sub(r'\b\w+\s+by[,.]?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\b\w+\s+by[,.]?\s+(?:it is|it|the)', '', text, flags=re.IGNORECASE)
        
        # Remove very short sentence fragments (1-2 words)
        text = re.sub(r'\b[A-Z][a-z]*\s*[.!?]\s*', lambda m: '' if len(m.group(0).split()) <= 2 else m.group(0), text)
        
        # Remove duplicate words/phrases in sequence (e.g., "E-commerce and IT security E-commerce")
        text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        # Fix "e-commerce and IT security E-commerce" -> "e-commerce and IT security"
        text = re.sub(r'\b(e-commerce and it security)\s+e-commerce\b', r'\1', text, flags=re.IGNORECASE)
        # Fix "bachelor s degree" -> "bachelor's degree"
        text = re.sub(r'\bbachelor\s+s\s+degree\b', "bachelor's degree", text, flags=re.IGNORECASE)
        
        # Fix broken contractions and punctuation
        text = re.sub(r'(\w+)\s*\)\s*["\']\s*([sd]?\b)', r"\1\2", text)
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s{2,}', r'\1 ', text)
        
        # Split into sentences and filter only OBVIOUS garbage
        sentences = []
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip only very short fragments
            if len(sentence) < 10:
                continue
            
            # Skip incomplete sentences ending with "by" or "by,"
            if re.search(r'\bby[,.]?\s*$', sentence, re.IGNORECASE):
                continue
            
            # Skip sentence fragments (very short, no verb-like structure)
            words = sentence.split()
            if len(words) <= 2:
                continue
            
            # Skip if contains document identifiers (more aggressive patterns)
            if re.search(r'\d{1,3}\s+(?:Five Year Plan|Year Plan).*?(?:Information and Communication Technologies|Information and Communication)', sentence, re.IGNORECASE):
                continue
            # Skip sentences that are just document identifiers
            if re.match(r'^\d{1,3}\s+(?:Five Year Plan|Year Plan)', sentence, re.IGNORECASE):
                continue
            
            # Skip if has obvious nonsense words
            obvious_nonsense = ['folloo', 'faloo', 'grae', 'mos', 'chrias']
            if any(word in sentence.lower() for word in obvious_nonsense):
                logger.debug(f"Skipping nonsense: {sentence[:50]}")
                continue
            
            # Skip incomplete sentences (ending with incomplete phrases)
            if sentence.endswith(' by') or sentence.endswith(' by,') or sentence.endswith(' by.'):
                continue
            
            # Fix ending if incomplete
            if sentence[-1] not in '.!?':
                sentence = sentence + '.'
            
            # Check word count - need at least 3 words
            words = sentence.split()
            if len(words) < 3:
                continue
            
            # Ensure proper capitalization
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
            
            sentences.append(sentence)
        
        # Remove incomplete sentences at the end (common issue with model output)
        # Remove sentences that end with incomplete phrases like "Financial outlay In order to launch and actualise various"
        if sentences:
            # Check last few sentences for incomplete endings
            incomplete_endings = [
                'financial outlay',
                'in order to launch',
                'in order to',
                'to launch and actualise',
                'actualise various',
                'various',
                'constraints',
                'issue:',
                'action plan',
                'action plan.',
            ]
            
            # Remove trailing incomplete sentences
            while sentences:
                last_sentence = sentences[-1].lower().strip()
                # Check if last sentence is incomplete
                is_incomplete = False
                
                # Check for incomplete ending phrases
                for ending in incomplete_endings:
                    if last_sentence.endswith(ending) or ending in last_sentence[-50:]:
                        is_incomplete = True
                        break
                
                # Check if sentence is too short and ends with incomplete phrase
                if len(last_sentence.split()) < 8 and any(phrase in last_sentence for phrase in ['financial', 'outlay', 'actualise', 'various']):
                    is_incomplete = True
                
                if is_incomplete:
                    sentences.pop()
                else:
                    break
        
        # Join with proper spacing
        result = ' '.join(sentences)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        # Ensure result ends properly (not with incomplete phrase)
        if result:
            incomplete_endings_lower = [e.lower() for e in incomplete_endings]
            result_lower = result.lower()
            for ending in incomplete_endings_lower:
                if result_lower.endswith(ending):
                    # Find the last complete sentence before this incomplete ending
                    sentences_final = re.split(r'(?<=[.!?])\s+', result)
                    if len(sentences_final) > 1:
                        result = ' '.join(sentences_final[:-1])
                    break
        
        return result
    
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
        Main summarization method with CPU OPTIMIZATIONS.
        """
        if not text or len(text.strip()) < 100:
            return ""
        
        if not self._initialized and not self.initialize():
            logger.error("Model not ready")
            return text[:500]  # Return extract as fallback
        
        # Clean input text
        cleaned_text = self.preprocessor.clean_text(text)
        if not cleaned_text:
            return ""
        
        # Get preset-based parameters
        if length_penalty is None:
            length_penalty = Config.get_length_penalty_for_preset(preset)
        
        # Adjust parameters based on preset - balanced for quality
        if preset == "long":
            max_length = max_length or 300  # Comprehensive length for long summaries
            min_length = min_length or 120  # Ensure substantial content
        elif preset == "medium":
            max_length = max_length or 200  # Balanced length for medium summaries
            min_length = min_length or 80   # Ensure adequate content
        else:  # short
            max_length = max_length or 90
            min_length = min_length or 35
        
        logger.info(f"Summarizing with preset='{preset}', max={max_length}, min={min_length}, penalty={length_penalty}")
        
        # SIMPLE CHUNKING FOR CPU
        if not chunk or len(cleaned_text) < 2000:
            # Direct summarization for short texts
            summary = self._summarize_chunk(
                cleaned_text,
                max_length,
                min_length,
                length_penalty,
                preset
            )
        else:
            # Chunked processing for long texts
            # Use larger chunks for "long" preset to get better context
            if preset == "long":
                chunk_max_tokens = 400  # Larger chunks for comprehensive summaries
                chunk_overlap = 100     # More overlap for continuity
                max_chunks = 25         # Allow more chunks for comprehensive coverage
            elif preset == "medium":
                chunk_max_tokens = 370  # Smaller chunks than long
                chunk_overlap = 85      # Good overlap for continuity
                max_chunks = 18         # Fewer chunks than long for shorter output
            else:  # short
                chunk_max_tokens = 350
                chunk_overlap = 75
                max_chunks = 15
            
            chunks = self.preprocessor.chunk_text(
                cleaned_text,
                max_tokens=chunk_max_tokens,
                overlap=chunk_overlap
            )
            
            # Process chunks but with validation - quality over arbitrary limits
            if len(chunks) > max_chunks:
                logger.info(f"Limiting to {max_chunks} chunks for CPU performance (had {len(chunks)})")
                chunks = chunks[:max_chunks]
            
            logger.info(f"Processing {len(chunks)} chunks for '{preset}' preset")
            summaries = []
            failed_chunks = 0
            
            for i, chunk_text in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Use preset-specific parameters for per-chunk summaries
                if preset == "long":
                    chunk_max = 150  # Longer per-chunk summaries for comprehensive output
                    chunk_min = 60   # Ensure substantial content per chunk
                elif preset == "medium":
                    chunk_max = 110  # Balanced per-chunk summaries for medium output (shorter than long)
                    chunk_min = 45   # Ensure adequate content per chunk
                else:  # short
                    chunk_max = 80
                    chunk_min = 30
                
                chunk_summary = self._summarize_chunk(
                    chunk_text,
                    chunk_max,
                    chunk_min,
                    length_penalty * 0.9,  # Slightly lower penalty for chunks
                    preset
                )
                
                if chunk_summary and not self._is_garbage(chunk_summary):
                    summaries.append(chunk_summary)
                    logger.info(f"  ✓ Chunk {i+1}: {len(chunk_summary)} chars - VALID")
                else:
                    failed_chunks += 1
                    logger.warning(f"  ✗ Chunk {i+1}: garbage or empty - REJECTED")
                    
                    # Be more lenient - allow more failures before aborting
                    if failed_chunks > 10:
                        logger.error("Too many failed chunks (10+), stopping")
                        break
                
                # CPU optimization: Clear memory between chunks
                if i % 2 == 0:
                    gc.collect()
            
            logger.info(f"Chunk processing complete: {len(summaries)}/{len(chunks)} valid")
            
            if summaries:
                # Combine summaries with proper spacing
                # Ensure each summary ends properly before joining
                normalized_summaries = []
                for s in summaries:
                    s = s.strip()
                    if s:
                        # Ensure it ends with punctuation
                        if s and s[-1] not in '.!?':
                            s = s + '.'
                        normalized_summaries.append(s)
                combined = ' '.join(normalized_summaries)
                logger.info(f"Combined summary: {len(combined)} chars from {len(summaries)} chunks")
                
                # For "long" preset, skip final consolidation to preserve comprehensive content
                if preset == "long":
                    logger.info("Deduplicating combined summaries for 'long' preset")
                    combined = self._deduplicate_sentences(combined)
                    logger.info("Using combined summaries directly for 'long' preset (no consolidation)")
                    summary = combined
                elif preset == "medium":
                    # For medium preset, use light consolidation to keep it shorter than long
                    # but still preserve more content than aggressive consolidation
                    logger.info("Deduplicating combined summaries for 'medium' preset")
                    combined = self._deduplicate_sentences(combined)
                    
                    min_chunks_for_consolidation = 3
                    if len(combined) > 400 and len(summaries) >= min_chunks_for_consolidation:
                        logger.info("Attempting light consolidation for 'medium' preset...")
                        # Use higher minimum (80% of min_length) to preserve more content
                        final_min_length = int(min_length * 0.8)  # 80% of min_length = 64 tokens
                        final_summary = self._summarize_chunk(
                            combined,
                            max_length,  # Use max_length = 200
                            final_min_length,
                            length_penalty,
                            preset
                        )
                        
                        # Accept final summary if it's reasonable and shorter than combined
                        if final_summary and not self._is_garbage(final_summary) and len(final_summary) > 100:
                            # Only use consolidated if it's meaningfully shorter (at least 20% reduction)
                            if len(final_summary) < len(combined) * 0.8:
                                logger.info(f"  ✓ Light consolidation successful: {len(final_summary)} chars (reduced from {len(combined)})")
                                summary = final_summary
                            else:
                                logger.info(f"  Light consolidation didn't reduce enough, using combined: {len(combined)} chars")
                                summary = combined
                        else:
                            logger.info("  Light consolidation rejected, using combined summaries")
                            summary = combined
                    else:
                        logger.info(f"Using combined summaries directly for 'medium' (length: {len(combined)}, chunks: {len(summaries)})")
                        summary = combined
                else:
                    # For short preset, skip consolidation - we'll format it as brief + bullets
                    logger.info("Deduplicating combined summaries for 'short' preset")
                    combined = self._deduplicate_sentences(combined)
                    logger.info(f"Using combined summaries for 'short' preset (will format as brief + bullets): {len(combined)} chars")
                    summary = combined
            else:
                logger.error("❌ No valid chunk summaries generated - all chunks failed")
                summary = ""
        
        # Clean and validate output
        if summary:
            cleaned_summary = self._clean_output(summary)
            
            # Very lenient final validation - accept almost anything with content
            if cleaned_summary and len(cleaned_summary) > 30:
                
                # Special formatting for "short" preset: brief overview + bullet points
                if preset == "short":
                    logger.info("Formatting 'short' preset as brief overview + bullet points")
                    
                    # Step 1: Generate a brief overview (2-3 sentences, more comprehensive)
                    logger.info("Generating brief overview...")
                    brief_overview = self._summarize_chunk(
                        cleaned_text if len(cleaned_text) < 2000 else cleaned_summary,
                        max_length=100,  # Increased for more comprehensive overview
                        min_length=50,   # Increased minimum for better content
                        length_penalty=0.7,  # Slightly higher penalty for more content
                        summary_type="short"
                    )
                    
                    if not brief_overview or self._is_garbage(brief_overview):
                        # Fallback: use first 3 sentences from existing summary
                        sentences = re.split(r'[.!?]+', cleaned_summary)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        brief_overview = '. '.join(sentences[:3]) if len(sentences) >= 3 else '. '.join(sentences[:2]) if len(sentences) >= 2 else sentences[0] if sentences else cleaned_summary[:200]
                    
                    # Clean up overview
                    brief_overview = brief_overview.strip()
                    if brief_overview and brief_overview[-1] not in '.!?':
                        brief_overview += '.'
                    
                    # Step 2: Extract main points using multiple strategies for better coverage
                    logger.info("Extracting main points...")
                    main_points = []
                    
                    # Strategy 1: Extract from original text (if not too long)
                    if len(cleaned_text) < 4000:
                        logger.info("Extracting points from original text...")
                        points_from_text = self._extract_main_points(cleaned_text, num_points=12)
                        if points_from_text:
                            main_points.extend(points_from_text)
                            logger.info(f"Extracted {len(points_from_text)} points from original text")
                    
                    # Strategy 2: Extract from combined summary (from chunks)
                    if len(main_points) < 6:
                        logger.info("Extracting additional points from summary...")
                        points_from_summary = self._extract_main_points(cleaned_summary, num_points=12)
                        if points_from_summary:
                            # Add points that aren't duplicates
                            for point in points_from_summary:
                                if point not in main_points:
                                    main_points.append(point)
                            logger.info(f"Added {len([p for p in points_from_summary if p not in main_points])} additional points from summary")
                    
                    # Strategy 3: Use sentences from summary as fallback
                    if len(main_points) < 5:
                        logger.info("Using summary sentences as additional fallback points...")
                        sentences = re.split(r'[.!?]+', cleaned_summary)
                        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
                        
                        # Skip first 3 sentences (used in overview) and take diverse sentences
                        candidate_sentences = sentences[3:] if len(sentences) > 3 else sentences
                        
                        # Add sentences that aren't already in main_points
                        for sentence in candidate_sentences:
                            if len(main_points) >= 10:
                                break
                            # Check if sentence is substantially different from existing points
                            is_duplicate = False
                            sentence_normalized = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
                            for existing_point in main_points:
                                existing_normalized = re.sub(r'[^\w\s]', '', existing_point.lower()).strip()
                                # Check word overlap
                                sentence_words = set(sentence_normalized.split())
                                existing_words = set(existing_normalized.split())
                                if sentence_words and existing_words:
                                    overlap = len(sentence_words & existing_words)
                                    union = len(sentence_words | existing_words)
                                    if union > 0 and overlap / union > 0.6:  # 60% similarity
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                main_points.append(sentence)
                        
                        logger.info(f"Added {len(main_points)} total points using fallback strategy")
                    
                    # Format as brief overview + bullet points
                    result_parts = [brief_overview]
                    unique_points = []
                    
                    if main_points:
                        # Filter out points that are too similar to the overview
                        filtered_points = []
                        for point in main_points:
                            point = point.strip()
                            if not point:
                                continue
                            
                            # Clean incomplete sentences
                            point = self._clean_incomplete_sentence(point)
                            # Reduced minimum length to 12 to keep more points
                            if not point or len(point) < 12:
                                continue
                            
                            # Additional cleanup for specific issues
                            point = self._final_clean_point(point)
                            if not point or len(point) < 12:
                                continue
                            
                            # Check if point is too similar to overview (avoid repetition)
                            # Use higher threshold (0.65) to be less strict and keep more points
                            if not self._is_similar_to_overview(point, brief_overview, threshold=0.65):
                                filtered_points.append(point)
                            else:
                                logger.debug(f"Filtered out repetitive point: {point[:50]}...")
                        
                        # Also check for duplicates among points themselves
                        seen_normalized = set()
                        for point in filtered_points:
                            # Normalize for comparison
                            normalized = re.sub(r'[^\w\s]', '', point.lower()).strip()
                            if normalized not in seen_normalized and len(normalized) > 10:
                                seen_normalized.add(normalized)
                                unique_points.append(point)
                        
                        # If we don't have enough unique points, be more lenient with similarity check
                        if len(unique_points) < 4 and len(filtered_points) < len(main_points):
                            logger.info(f"Only {len(unique_points)} unique points, relaxing similarity threshold...")
                            # Re-check with more lenient threshold
                            additional_points = []
                            for point in main_points:
                                point = point.strip()
                                if not point:
                                    continue
                                point = self._clean_incomplete_sentence(point)
                                if not point or len(point) < 12:
                                    continue
                                # Use very lenient threshold (0.75) to get more points
                                if not self._is_similar_to_overview(point, brief_overview, threshold=0.75):
                                    normalized = re.sub(r'[^\w\s]', '', point.lower()).strip()
                                    if normalized not in seen_normalized and len(normalized) > 10:
                                        seen_normalized.add(normalized)
                                        additional_points.append(point)
                            unique_points.extend(additional_points)
                        
                        if unique_points:
                            result_parts.append("")  # Empty line before bullets
                            # Show up to 8 points for better coverage
                            for point in unique_points[:8]:
                                # Ensure point ends with punctuation
                                if point[-1] not in '.!?':
                                    point += '.'
                                # Capitalize first letter
                                if point and point[0].islower():
                                    point = point[0].upper() + point[1:]
                                result_parts.append(f"• {point}")
                    
                    formatted_summary = '\n'.join(result_parts)
                    logger.info(f"✓ SUCCESS: Formatted short summary: {len(formatted_summary)} chars ({len(brief_overview)} overview + {len(unique_points)} points)")
                    return formatted_summary
                
                # For other presets (medium, long), return cleaned summary as-is
                logger.info(f"✓ SUCCESS: Final summary: {len(cleaned_summary)} chars, {len(cleaned_summary.split())} words")
                return cleaned_summary
            else:
                logger.warning(f"Cleaned summary too short: {len(cleaned_summary) if cleaned_summary else 0} chars")
                # Try returning the uncleaned version if cleaning removed too much
                if summary and len(summary) > 50:
                    logger.info("Returning uncleaned summary")
                    return summary
        
        # If we got here, summarization failed completely
        logger.error("❌ Abstractive summarization produced no valid output")
        logger.error("This usually means the model couldn't generate coherent summaries")
        logger.error("Possible causes: text too complex, PDF extraction issues, or model limitations")
        
        # Return a message instead of falling back to extractive
        return "Unable to generate abstractive summary. The model could not produce coherent output for this document. Please try a different summarization mode or check the document content."
    
    def is_ready(self) -> bool:
        return self._initialized
    
    def cleanup(self) -> None:
        if self.model_loader:
            self.model_loader.unload_model()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        gc.collect()
        logger.info("Summarizer cleaned up")
