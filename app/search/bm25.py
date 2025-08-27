"""BM25 implementation for keyword-based search."""

import math
import pickle
import re
from collections import Counter
from typing import Any

from app.logging import get_logger

logger = get_logger(__name__)


class BM25:
    """BM25 ranking function implementation."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 with parameters.

        Args:
            k1: Controls term frequency saturation (default: 1.2)
            b: Controls field-length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b

        # Document statistics
        self.doc_freqs: dict[str, int] = {}  # Term document frequencies
        self.doc_lens: list[int] = []  # Document lengths
        self.avgdl: float = 0.0  # Average document length
        self.corpus_size: int = 0

        # Tokenized documents
        self.tokenized_docs: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms."""
        # Simple tokenization - split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def fit(self, documents: list[str]) -> None:
        """Fit BM25 on document corpus."""
        logger.info(f"Fitting BM25 on {len(documents)} documents")

        # Tokenize all documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.corpus_size = len(documents)

        # Calculate document lengths
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc_tokens in self.tokenized_docs:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        logger.info(
            f"BM25 fitted: {len(self.doc_freqs)} unique terms, avgdl={self.avgdl:.1f}"
        )

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0

        # IDF calculation: log((N - df + 0.5) / (df + 0.5))
        idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5))
        return max(idf, 0.0)  # Ensure non-negative

    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        if doc_idx >= len(self.tokenized_docs):
            return 0.0

        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue

            # BM25 formula
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Search for documents matching query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (document_index, score) tuples
        """
        if not self.tokenized_docs:
            logger.warning("BM25 not fitted, returning empty results")
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores = []
        for doc_idx in range(len(self.tokenized_docs)):
            score = self._score_document(query_terms, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_term_stats(self) -> dict[str, Any]:
        """Get BM25 statistics."""
        return {
            "corpus_size": self.corpus_size,
            "unique_terms": len(self.doc_freqs),
            "avg_doc_length": self.avgdl,
            "total_tokens": sum(self.doc_lens),
            "k1": self.k1,
            "b": self.b,
        }

    def save(self, filepath: str) -> None:
        """Save BM25 model to file."""
        model_data = {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": self.doc_freqs,
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
            "corpus_size": self.corpus_size,
            "tokenized_docs": self.tokenized_docs,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"BM25 model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load BM25 model from file."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.k1 = model_data["k1"]
        self.b = model_data["b"]
        self.doc_freqs = model_data["doc_freqs"]
        self.doc_lens = model_data["doc_lens"]
        self.avgdl = model_data["avgdl"]
        self.corpus_size = model_data["corpus_size"]
        self.tokenized_docs = model_data["tokenized_docs"]

        logger.info(f"BM25 model loaded from {filepath}")


class BM25Plus(BM25):
    """BM25+ variant with improved handling of long documents."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 1.0):
        super().__init__(k1, b)
        self.delta = delta

    def _score_document(self, query_terms: list[str], doc_idx: int) -> float:
        """Calculate BM25+ score for a document."""
        if doc_idx >= len(self.tokenized_docs):
            return 0.0

        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue

            # BM25+ formula (adds delta to handle long documents better)
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * ((numerator / denominator) + self.delta)

        return score
