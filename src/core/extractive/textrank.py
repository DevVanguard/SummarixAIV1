"""TextRank algorithm implementation for extractive summarization."""

import logging
from typing import List, Optional

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class TextRankSummarizer:
    """Extractive summarization using TextRank algorithm."""
    
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
            max_features=5000
        )
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build sentence similarity matrix using TF-IDF and cosine similarity.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix
        """
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            # Compute TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error building similarity matrix: {str(e)}")
            # Return identity matrix as fallback
            return np.eye(len(sentences))
    
    def _build_graph(self, similarity_matrix: np.ndarray, threshold: float = 0.1) -> nx.Graph:
        """
        Build graph from similarity matrix.
        
        Args:
            similarity_matrix: Matrix of sentence similarities
            threshold: Minimum similarity to create an edge
            
        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        num_sentences = len(similarity_matrix)
        
        # Add nodes
        graph.add_nodes_from(range(num_sentences))
        
        # Add edges based on similarity
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)
        
        return graph
    
    def _rank_sentences(self, graph: nx.Graph) -> List[tuple]:
        """
        Rank sentences using PageRank algorithm.
        
        Args:
            graph: Sentence similarity graph
            
        Returns:
            List of (sentence_index, score) tuples sorted by score
        """
        try:
            # Use PageRank to compute sentence importance
            scores = nx.pagerank(graph, max_iter=100)
            
            # Sort by score (descending)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return ranked
        except Exception as e:
            logger.error(f"Error ranking sentences: {str(e)}")
            # Fallback: return sentences in order
            return [(i, 1.0 / len(graph.nodes())) for i in graph.nodes()]
    
    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None
    ) -> str:
        """
        Generate extractive summary using TextRank.
        
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
        sentences = self.preprocessor.tokenize_sentences(text)
        
        if not sentences:
            return text
        
        if len(sentences) == 1:
            return sentences[0]
        
        # Determine number of sentences to extract
        if num_sentences is not None:
            num_summary_sentences = min(num_sentences, len(sentences))
        else:
            num_summary_sentences = max(1, int(len(sentences) * summary_ratio))
        
        if num_summary_sentences >= len(sentences):
            return ' '.join(sentences)
        
        try:
            # Build similarity matrix
            similarity_matrix = self._build_similarity_matrix(sentences)
            
            # Build graph
            graph = self._build_graph(similarity_matrix)
            
            # Rank sentences
            ranked_sentences = self._rank_sentences(graph)
            
            # Select top sentences
            selected_indices = [idx for idx, _ in ranked_sentences[:num_summary_sentences]]
            
            # Sort selected indices to maintain original order
            selected_indices.sort()
            
            # Extract selected sentences
            summary_sentences = [sentences[i] for i in selected_indices]
            
            summary = ' '.join(summary_sentences)
            
            logger.info(
                f"Generated extractive summary: {len(summary_sentences)}/{len(sentences)} sentences"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in TextRank summarization: {str(e)}")
            # Fallback: return first N sentences
            fallback_sentences = sentences[:num_summary_sentences]
            return ' '.join(fallback_sentences)
    
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
        
        sentences = self.preprocessor.tokenize_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [(sentences[0], 1.0)]
        
        num_summary_sentences = (
            num_sentences if num_sentences is not None
            else max(1, int(len(sentences) * summary_ratio))
        )
        
        try:
            similarity_matrix = self._build_similarity_matrix(sentences)
            graph = self._build_graph(similarity_matrix)
            ranked_sentences = self._rank_sentences(graph)
            
            # Get top sentences with scores
            top_sentences = ranked_sentences[:num_summary_sentences]
            
            # Create result with sentences and scores
            result = [(sentences[idx], score) for idx, score in top_sentences]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TextRank with scores: {str(e)}")
            return [(sent, 1.0) for sent in sentences[:num_summary_sentences]]

