"""Cross-encoder reranking for improved search relevance."""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.search.hybrid import SearchResult

try:
    from sentence_transformers import CrossEncoder

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from app.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model: CrossEncoder | None = None
        self.is_loaded = False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, reranking disabled")
            return

        # Lazy loading - model will be loaded on first use
        logger.info(f"Cross-encoder reranker initialized: {model_name}")

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self.is_loaded or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return

        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.is_loaded = True
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.model = None

    async def rerank(
        self, query: str, search_results: "list[SearchResult]", batch_size: int = 32
    ) -> "list[SearchResult]":
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            search_results: List of SearchResult objects to rerank
            batch_size: Batch size for processing

        Returns:
            Reranked list of SearchResult objects
        """
        if not search_results:
            return search_results

        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.model:
            self._load_model()
            if not self.model:
                logger.warning(
                    "Cross-encoder not available, returning original results"
                )
                return search_results

        try:
            # Prepare query-document pairs
            pairs = [(query, result.document.text) for result in search_results]

            # Process in batches to avoid memory issues
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]

                # Run cross-encoder scoring in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                batch_scores = await loop.run_in_executor(
                    None, self.model.predict, batch_pairs
                )
                all_scores.extend(batch_scores)

            # Update search results with rerank scores
            for result, score in zip(search_results, all_scores, strict=False):
                result.rerank_score = float(score)

            # Sort by rerank score
            reranked_results = sorted(
                search_results, key=lambda x: x.rerank_score or 0.0, reverse=True
            )

            logger.debug(f"Reranked {len(search_results)} results")
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return search_results

    def get_model_info(self) -> dict:
        """Get reranker model information."""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "available": SENTENCE_TRANSFORMERS_AVAILABLE,
        }


class MockReranker:
    """Mock reranker for testing without model dependencies."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        logger.info("Mock reranker initialized")

    async def rerank(
        self, query: str, search_results: "list[SearchResult]", batch_size: int = 32
    ) -> "list[SearchResult]":
        """Mock reranking - adds small random perturbation to hybrid scores."""
        import random

        for result in search_results:
            # Add small random perturbation to simulate reranking
            perturbation = random.uniform(-0.1, 0.1)
            result.rerank_score = max(0.0, result.hybrid_score + perturbation)

        # Sort by mock rerank score
        return sorted(search_results, key=lambda x: x.rerank_score or 0.0, reverse=True)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "is_loaded": True,
            "available": True,
            "mock": True,
        }


def create_reranker(
    model_name: str | None = None, use_mock: bool = False
) -> CrossEncoderReranker | None:
    """Create reranker instance."""
    if use_mock:
        return MockReranker(model_name or "mock")

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("sentence-transformers not available, using mock reranker")
        return MockReranker(model_name or "mock")

    return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-2-v2")
