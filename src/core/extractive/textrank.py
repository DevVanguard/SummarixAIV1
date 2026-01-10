"""TextRank algorithm implementation for extractive summarization - IMPROVED."""

import logging
from typing import List, Optional, Tuple
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class TextRankSummarizer:
    """Improved extractive summarization using enhanced TextRank algorithm."""
    
    def __init__(self, ratio: float = 0.3, language: str = 'english'):
        """
        Initialize TextRank summarizer.
        
        Args:
            ratio: Ratio of sentences to include in summary (0.0-1.0)
            language: Language for preprocessing
        """
        self.ratio = ratio
        self.preprocessor = TextPreprocessor(language=language)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            max_df=0.85  # Ignore very common terms
        )
    
    def _filter_quality_sentences(self, sentences: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter sentences based on quality criteria.
        Removes fragments, headers, noise, metadata, and very short sentences.
        
        Args:
            sentences: List of sentences to filter
            
        Returns:
            Tuple of (filtered_sentences, original_indices)
        """
        quality_sentences = []
        original_indices = []
        
        for idx, sentence in enumerate(sentences):
            # Skip if too short (likely fragment or noise)
            if len(sentence.split()) < 5:
                continue
            
            # Skip if too long (likely run-on, figure caption, or error)
            if len(sentence.split()) > 100:
                continue
            
            # Skip title-like sentences (mostly capitalized words in a row)
            words = sentence.split()
            if len(words) > 3:
                capitalized_count = sum(1 for w in words[:min(10, len(words))] if w and w[0].isupper())
                if capitalized_count / min(10, len(words)) > 0.7:
                    continue
            
            # Skip sentences that are just author names or affiliations
            name_indicators = ['university', 'department', 'lab', 'institute', 'google', 'microsoft', 'facebook', 'mit', 'stanford']
            if any(indicator in sentence.lower() for indicator in name_indicators) and len(words) < 15:
                continue
            
            # Skip abstract headers and section markers
            if sentence.lower().strip() in ['abstract', 'introduction', 'conclusion', 'references', 'acknowledgments']:
                continue
            
            # Skip if sentence starts with section number pattern
            if re.match(r'^\d+\.?\d*\s+[A-Z]', sentence):
                continue
            
            # Skip figure/table captions and descriptions (usually start with Figure/Table)
            if sentence.startswith(('Figure', 'Fig.', 'Table', 'Tab.', 'Equation', 'Eq.')):
                continue
            
            # Skip if contains figure/table description patterns
            figure_patterns = ['as shown in', 'is shown in', 'see figure', 'see table', 'shown in figure', 'shown in table']
            if any(pattern in sentence.lower() for pattern in figure_patterns):
                # Only skip if it's a short sentence (likely just a reference)
                if len(words) < 20:
                    continue
            
            # Skip if mostly numbers or symbols
            word_chars = sum(c.isalpha() for c in sentence)
            if word_chars / max(len(sentence), 1) < 0.5:
                continue
            
            # Skip incomplete sentences (ends with hyphen or comma without continuation)
            if sentence.rstrip().endswith(('-', 'and', 'or', 'but', 'with', 'from')):
                continue
            
            # Skip if no verb-like words (incomplete sentences)
            # Basic heuristic: look for common verb patterns
            words_lower = [w.lower() for w in words]
            has_verb = any(
                word.endswith(('ing', 'ed', 'es', 's')) or 
                word in ('is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did')
                for word in words_lower
            )
            if not has_verb:
                continue
            
            # Skip likely headers (all caps, very short, ends with colon)
            if sentence.isupper() or sentence.endswith(':'):
                continue
            
            # Skip if mostly stopwords (low information content)
            try:
                words_clean = [w.lower() for w in words if w.isalpha()]
                if len(words_clean) > 0:
                    stopword_ratio = sum(1 for w in words_clean if w in self.preprocessor.stop_words) / len(words_clean)
                    if stopword_ratio > 0.8:
                        continue
            except:
                pass
            
            quality_sentences.append(sentence)
            original_indices.append(idx)
        
        logger.debug(f"Filtered {len(sentences)} sentences to {len(quality_sentences)} quality sentences")
        return quality_sentences, original_indices
    
    def _compute_sentence_features(self, sentences: List[str], total_sentences: int) -> np.ndarray:
        """
        Compute additional features for sentences beyond TF-IDF.
        
        Args:
            sentences: List of sentences
            total_sentences: Total number of sentences in original document
            
        Returns:
            Feature matrix (n_sentences x n_features)
        """
        features = []
        
        for idx, sentence in enumerate(sentences):
            # Position bias: sentences at start and end are often important
            position_score = 1.0
            if idx < 3:  # First 3 sentences
                position_score = 1.5
            elif idx >= len(sentences) - 2:  # Last 2 sentences
                position_score = 1.3
            elif idx < len(sentences) * 0.2:  # First 20%
                position_score = 1.2
            
            # Length feature: prefer medium-length sentences
            word_count = len(sentence.split())
            length_score = 1.0
            if 10 <= word_count <= 30:  # Ideal length
                length_score = 1.2
            elif word_count < 5 or word_count > 50:  # Too short or too long
                length_score = 0.7
            
            # Named entity heuristic: sentences with capitalized words (proper nouns)
            words = sentence.split()
            capitalized_ratio = sum(1 for w in words if w and w[0].isupper() and not w.isupper()) / max(len(words), 1)
            entity_score = 1.0 + min(capitalized_ratio * 0.5, 0.3)  # Boost up to 1.3
            
            # Numeric content: sentences with numbers/dates often contain facts
            numeric_score = 1.0
            if any(c.isdigit() for c in sentence):
                numeric_score = 1.1
            
            # Keyword indicators: look for summary-indicating phrases
            keyword_score = 1.0
            sentence_lower = sentence.lower()
            
            # Boost for conclusion/finding indicators
            summary_phrases = [
                'in conclusion', 'in summary', 'to summarize', 'overall',
                'importantly', 'significant', 'key', 'main', 'primary',
                'therefore', 'thus', 'consequently', 'as a result',
                'found that', 'shows that', 'demonstrates', 'indicates',
                'we propose', 'we introduce', 'we present', 'we show',
                'our method', 'our approach', 'our results', 'achieves',
                'outperforms', 'improves', 'achieve', 'novel'
            ]
            if any(phrase in sentence_lower for phrase in summary_phrases):
                keyword_score = 1.5
            
            # Penalize background/introduction indicators
            background_phrases = [
                'has been shown', 'has been used', 'have been proposed',
                'recent work', 'prior work', 'previous work', 'traditionally',
                'in this paper we', 'in this work we', 'this paper',
                'language model pre-training', 'pre-training has'
            ]
            if any(phrase in sentence_lower for phrase in background_phrases):
                keyword_score = 0.7
            
            # Combine features
            combined_score = position_score * length_score * entity_score * numeric_score * keyword_score
            features.append(combined_score)
        
        # Normalize features
        features = np.array(features)
        if features.max() > 0:
            features = features / features.max()
        
        return features.reshape(-1, 1)
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build enhanced sentence similarity matrix using TF-IDF and cosine similarity.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix with additional weighting
        """
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            # Compute TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Apply additional weighting based on sentence features
            total_sentences = len(sentences)
            features = self._compute_sentence_features(sentences, total_sentences)
            
            # Weight similarity by sentence quality
            # Sentences with higher features should have more influence
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        # Weight by geometric mean of feature scores
                        weight = np.sqrt(features[i, 0] * features[j, 0])
                        similarity_matrix[i, j] *= weight
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error building similarity matrix: {str(e)}")
            # Return identity matrix as fallback
            return np.eye(len(sentences))
    
    def _build_graph(self, similarity_matrix: np.ndarray, threshold: float = 0.15) -> nx.Graph:
        """
        Build graph from similarity matrix with adaptive thresholding.
        
        Args:
            similarity_matrix: Matrix of sentence similarities
            threshold: Minimum similarity to create an edge (increased from 0.1)
            
        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        num_sentences = len(similarity_matrix)
        
        # Add nodes
        graph.add_nodes_from(range(num_sentences))
        
        # Calculate adaptive threshold based on distribution
        similarities = []
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarities.append(similarity_matrix[i][j])
        
        if similarities:
            # Use median as adaptive threshold, but not lower than specified minimum
            adaptive_threshold = max(threshold, np.median(similarities) * 0.7)
        else:
            adaptive_threshold = threshold
        
        # Add edges based on similarity
        edge_count = 0
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarity = similarity_matrix[i][j]
                if similarity > adaptive_threshold:
                    graph.add_edge(i, j, weight=similarity)
                    edge_count += 1
        
        # Ensure graph is connected by adding minimum edges if needed
        if edge_count == 0 and num_sentences > 1:
            logger.warning("No edges created with threshold, adding strongest connections")
            # Add edges to K strongest connections for each node
            k = min(2, num_sentences - 1)
            for i in range(num_sentences):
                # Get top-k similar sentences
                similarities_i = [(j, similarity_matrix[i][j]) for j in range(num_sentences) if j != i]
                similarities_i.sort(key=lambda x: x[1], reverse=True)
                for j, sim in similarities_i[:k]:
                    graph.add_edge(i, j, weight=sim)
        
        logger.debug(f"Built graph with {num_sentences} nodes and {graph.number_of_edges()} edges")
        return graph
    
    def _rank_sentences(self, graph: nx.Graph, sentences: List[str]) -> List[tuple]:
        """
        Rank sentences using enhanced PageRank algorithm with dampening.
        
        Args:
            graph: Sentence similarity graph
            sentences: Original sentences for feature computation
            
        Returns:
            List of (sentence_index, score) tuples sorted by score
        """
        try:
            # Use PageRank with higher dampening for more stable results
            scores = nx.pagerank(
                graph, 
                max_iter=200,  # More iterations for convergence
                alpha=0.90,  # Higher dampening factor (default 0.85)
                tol=1e-6  # Tighter convergence
            )
            
            # Boost scores with sentence features
            features = self._compute_sentence_features(sentences, len(sentences))
            for idx in scores:
                scores[idx] *= features[idx, 0]
            
            # Sort by score (descending)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Top 5 sentence scores: {ranked[:5]}")
            return ranked
        except Exception as e:
            logger.error(f"Error ranking sentences: {str(e)}")
            # Fallback: return sentences in order with equal weights
            return [(i, 1.0 / len(graph.nodes())) for i in graph.nodes()]
    
    def _ensure_coherence(
        self, 
        selected_indices: List[int], 
        sentences: List[str],
        similarity_matrix: np.ndarray
    ) -> List[int]:
        """
        Ensure selected sentences form a coherent summary by checking connectivity.
        May remove disconnected sentences or add bridging sentences.
        
        Args:
            selected_indices: Initially selected sentence indices
            sentences: All sentences
            similarity_matrix: Sentence similarity matrix
            
        Returns:
            Adjusted sentence indices for better coherence
        """
        if len(selected_indices) <= 2:
            return selected_indices
        
        # Sort by original position
        selected_indices = sorted(selected_indices)
        
        # Check for large gaps in positions
        adjusted_indices = [selected_indices[0]]
        for i in range(1, len(selected_indices)):
            prev_idx = adjusted_indices[-1]
            curr_idx = selected_indices[i]
            
            # If there's a large gap (more than 10 sentences), check connectivity
            if curr_idx - prev_idx > 10:
                # Check if sentences are semantically similar
                if similarity_matrix[prev_idx][curr_idx] < 0.1:
                    # Low similarity and large gap - might be disconnected
                    # Look for a bridging sentence
                    bridge_found = False
                    for bridge_idx in range(prev_idx + 1, curr_idx):
                        if (similarity_matrix[prev_idx][bridge_idx] > 0.15 and 
                            similarity_matrix[bridge_idx][curr_idx] > 0.15):
                            # Found a bridge sentence
                            adjusted_indices.append(bridge_idx)
                            bridge_found = True
                            break
                    
                    if not bridge_found and len(adjusted_indices) >= 3:
                        # Skip this sentence if no bridge found and we have enough
                        continue
            
            adjusted_indices.append(curr_idx)
        
        return adjusted_indices
    
    def _post_process_summary(self, summary: str) -> str:
        """
        Post-process the summary to ensure proper formatting, capitalization, and punctuation.
        Removes any remaining citation remnants, metadata, and fixes spacing issues.
        
        Args:
            summary: Raw summary text
            
        Returns:
            Cleaned and polished summary
        """
        if not summary:
            return ""
        
        # Step 1: Remove title/author/header remnants at the start
        # Remove lines that look like titles (multiple capitalized words)
        lines = summary.split('\n')
        cleaned_lines = []
        for line in lines:
            words = line.strip().split()
            if len(words) > 3:
                # Skip if more than 60% words are capitalized (likely title)
                cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
                if cap_ratio > 0.6:
                    continue
            cleaned_lines.append(line)
        summary = '\n'.join(cleaned_lines)
        
        # Step 2: Remove any remaining citation artifacts
        # Parentheses with just whitespace or fragments
        summary = re.sub(r'\(\s*\)', '', summary)
        summary = re.sub(r'\(\s*,\s*\)', '', summary)
        summary = re.sub(r'\(\s*et al\.\s*,?\s*\)', '', summary)
        summary = re.sub(r'\(\s*\d{4}[a-z]?\s*\)', '', summary)  # (2018a)
        
        # Step 3: Remove email/affiliation remnants
        summary = re.sub(r'@\S+', '', summary)
        summary = re.sub(r'Google\s+AI\s+Language', '', summary)
        
        # Step 4: Remove "Abstract" if at the start
        summary = re.sub(r'^Abstract\s*[:\.]?\s*', '', summary, flags=re.IGNORECASE)
        
        # Step 5: Fix multiple spaces
        summary = re.sub(r' +', ' ', summary)
        
        # Step 6: Fix spacing around punctuation
        summary = re.sub(r'\s+([.,;:!?])', r'\1', summary)  # Remove space before punctuation
        summary = re.sub(r'([.,;:!?])([A-Z])', r'\1 \2', summary)  # Add space after punctuation before capital
        summary = re.sub(r'([.,;:!?])([a-z])', r'\1 \2', summary)  # Add space after punctuation before lowercase
        
        # Step 7: Fix double punctuation
        summary = re.sub(r'\.{2,}', '.', summary)
        summary = re.sub(r',{2,}', ',', summary)
        
        # Step 8: Remove incomplete sentence fragments at the end
        summary = summary.rstrip('-')
        
        # Step 9: Ensure sentences start with capital letters
        sentences = self.preprocessor.tokenize_sentences(summary)
        capitalized_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Skip very short fragments
                if len(sentence.split()) < 4:
                    continue
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        # Step 10: Join sentences with proper spacing
        summary = ' '.join(capitalized_sentences)
        
        # Step 11: Ensure summary ends with proper punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Step 12: Remove any remaining artifacts
        # Empty parentheses or brackets
        summary = re.sub(r'\[\s*\]', '', summary)
        summary = re.sub(r'\(\s*\)', '', summary)
        
        # Step 13: Fix spacing after cleanup
        summary = re.sub(r' +', ' ', summary)
        summary = summary.strip()
        
        return summary
    
    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None
    ) -> str:
        """
        Generate extractive summary using improved TextRank.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (overrides ratio)
            ratio: Ratio of sentences to include (0.0-1.0)
            
        Returns:
            Summarized text
        """
        if not text or not text.strip():
            return ""
        
        # Use provided ratio or default
        summary_ratio = ratio if ratio is not None else self.ratio
        
        # Tokenize into sentences
        all_sentences = self.preprocessor.tokenize_sentences(text)
        
        if not all_sentences:
            return text
        
        if len(all_sentences) == 1:
            return all_sentences[0]
        
        # Filter for quality sentences
        sentences, original_indices = self._filter_quality_sentences(all_sentences)
        
        if not sentences:
            logger.warning("No quality sentences found, using original")
            sentences = all_sentences
            original_indices = list(range(len(all_sentences)))
        
        if len(sentences) == 1:
            return sentences[0]
        
        # Determine number of sentences to extract
        if num_sentences is not None:
            num_summary_sentences = min(num_sentences, len(sentences))
        else:
            # Adaptive ratio based on document length
            if len(all_sentences) < 10:
                adaptive_ratio = 0.5  # Keep more from short documents
            elif len(all_sentences) > 50:
                adaptive_ratio = min(summary_ratio, 0.2)  # Keep less from long documents
            else:
                adaptive_ratio = summary_ratio
            
            num_summary_sentences = max(3, int(len(sentences) * adaptive_ratio))
            num_summary_sentences = min(num_summary_sentences, len(sentences))
        
        if num_summary_sentences >= len(sentences):
            return ' '.join(sentences)
        
        try:
            # Build similarity matrix with enhanced features
            similarity_matrix = self._build_similarity_matrix(sentences)
            
            # Build graph with adaptive threshold
            graph = self._build_graph(similarity_matrix)
            
            # Rank sentences with features
            ranked_sentences = self._rank_sentences(graph, sentences)
            
            # Select top sentences
            selected_indices = [idx for idx, _ in ranked_sentences[:num_summary_sentences]]
            
            # Ensure coherence
            selected_indices = self._ensure_coherence(selected_indices, sentences, similarity_matrix)
            
            # Sort selected indices to maintain original order
            selected_indices.sort()
            
            # Extract selected sentences
            summary_sentences = [sentences[i] for i in selected_indices]
            
            # Clean up sentences before joining
            summary_sentences = [s.strip() for s in summary_sentences if s.strip()]
            
            # Join with single space for proper flow
            summary = ' '.join(summary_sentences)
            
            # Final cleanup: remove any double spaces
            import re
            summary = re.sub(r' +', ' ', summary).strip()
            
            # Post-process for academic papers: fix capitalization, punctuation, spacing
            summary = self._post_process_summary(summary)
            
            logger.info(
                f"Generated extractive summary: {len(summary_sentences)}/{len(all_sentences)} sentences "
                f"(from {len(sentences)} quality sentences)"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in TextRank summarization: {str(e)}", exc_info=True)
            # Fallback: return first N sentences (usually contain key info)
            fallback_sentences = [s.strip() for s in sentences[:num_summary_sentences] if s.strip()]
            summary = ' '.join(fallback_sentences)
            # Clean up spacing
            import re
            summary = re.sub(r' +', ' ', summary).strip()
            # Post-process even for fallback
            summary = self._post_process_summary(summary)
            return summary
    
    def summarize_with_scores(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None
    ) -> List[tuple]:
        """
        Generate summary with sentence scores.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            ratio: Ratio of sentences to include
            
        Returns:
            List of (sentence, score) tuples
        """
        if not text or not text.strip():
            return []
        
        summary_ratio = ratio if ratio is not None else self.ratio
        
        all_sentences = self.preprocessor.tokenize_sentences(text)
        
        if not all_sentences:
            return []
        
        if len(all_sentences) == 1:
            return [(all_sentences[0], 1.0)]
        
        sentences, original_indices = self._filter_quality_sentences(all_sentences)
        
        if not sentences:
            sentences = all_sentences
        
        if len(sentences) == 1:
            return [(sentences[0], 1.0)]
        
        num_summary_sentences = (
            num_sentences if num_sentences is not None
            else max(3, int(len(sentences) * summary_ratio))
        )
        
        try:
            similarity_matrix = self._build_similarity_matrix(sentences)
            graph = self._build_graph(similarity_matrix)
            ranked_sentences = self._rank_sentences(graph, sentences)
            
            # Get top sentences with scores
            top_sentences = ranked_sentences[:num_summary_sentences]
            
            # Create result with sentences and scores
            result = [(sentences[idx], score) for idx, score in top_sentences]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TextRank with scores: {str(e)}")
            return [(sent, 1.0) for sent in sentences[:num_summary_sentences]]

