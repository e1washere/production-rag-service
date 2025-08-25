"""Hybrid search combining BM25 and embeddings with optional reranking."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.logging import get_logger
from app.search.bm25 import BM25
from app.search.reranker import CrossEncoderReranker

logger = get_logger(__name__)


@dataclass
class Document:
    """Document with metadata."""

    id: str
    text: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    """Search result with scores."""

    document: Document
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0


class HybridSearchEngine:
    """Hybrid search engine combining BM25 and embeddings."""

    def __init__(
        self,
        alpha: float = 0.7,
        enable_reranking: bool = False,
        reranker_model: str | None = None,
    ):
        """
        Initialize hybrid search engine.

        Args:
            alpha: Weight for hybrid scoring (0=BM25 only, 1=embeddings only)
            enable_reranking: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model name
        """
        self.alpha = alpha
        self.enable_reranking = enable_reranking

        # Search components
        self.bm25 = BM25()
        self.reranker = None

        if enable_reranking:
            model_name = reranker_model or settings.reranker_model
            self.reranker = CrossEncoderReranker(model_name)

        # Document storage
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

        logger.info(
            f"Hybrid search initialized: alpha={alpha}, reranking={enable_reranking}"
        )

    def add_documents(
        self, documents: list[Document], embeddings: np.ndarray | None = None
    ) -> None:
        """Add documents to the search index."""
        self.documents = documents

        # Fit BM25 on document texts
        doc_texts = [doc.text for doc in documents]
        self.bm25.fit(doc_texts)

        # Store embeddings
        if embeddings is not None:
            self.embeddings = embeddings
            logger.info(f"Added {len(documents)} documents with embeddings")
        else:
            logger.info(f"Added {len(documents)} documents (BM25 only)")

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Perform BM25 search."""
        return self.bm25.search(query, top_k)

    def _embedding_search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Perform embedding-based search."""
        if self.embeddings is None:
            return []

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top_k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    async def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
        rerank_top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            rerank_top_k: Number of results to rerank (if None, uses top_k * 2)

        Returns:
            List of SearchResult objects
        """
        if not self.documents:
            logger.warning("No documents in search index")
            return []

        # Determine search candidates pool size
        search_pool_size = max(
            top_k * 3, 50
        )  # Search more candidates for better recall

        # Perform BM25 search
        bm25_results = self._bm25_search(query, search_pool_size)
        bm25_doc_ids = {doc_idx for doc_idx, _ in bm25_results}

        # Perform embedding search if available
        embedding_results = []
        embedding_doc_ids = set()

        if query_embedding is not None and self.embeddings is not None:
            embedding_results = self._embedding_search(
                query_embedding, search_pool_size
            )
            embedding_doc_ids = {doc_idx for doc_idx, _ in embedding_results}

        # Combine candidate documents
        all_doc_ids = bm25_doc_ids | embedding_doc_ids

        # Create score mappings
        bm25_scores = dict(bm25_results)
        embedding_scores = dict(embedding_results)

        # Normalize scores for fair combination
        if bm25_scores:
            bm25_score_values = list(bm25_scores.values())
            normalized_bm25 = self._normalize_scores(bm25_score_values)
            bm25_scores = dict(zip(bm25_scores.keys(), normalized_bm25, strict=False))

        if embedding_scores:
            embedding_score_values = list(embedding_scores.values())
            normalized_embedding = self._normalize_scores(embedding_score_values)
            embedding_scores = dict(
                zip(embedding_scores.keys(), normalized_embedding, strict=False)
            )

        # Create search results
        search_results = []
        for doc_idx in all_doc_ids:
            if doc_idx >= len(self.documents):
                continue

            document = self.documents[doc_idx]
            bm25_score = bm25_scores.get(doc_idx, 0.0)
            embedding_score = embedding_scores.get(doc_idx, 0.0)

            # Calculate hybrid score
            hybrid_score = (1 - self.alpha) * bm25_score + self.alpha * embedding_score

            result = SearchResult(
                document=document,
                bm25_score=bm25_score,
                embedding_score=embedding_score,
                hybrid_score=hybrid_score,
                final_score=hybrid_score,
            )
            search_results.append(result)

        # Sort by hybrid score
        search_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Apply reranking if enabled
        if self.enable_reranking and self.reranker:
            rerank_count = rerank_top_k or min(top_k * 2, len(search_results))
            if rerank_count > 0:
                candidates = search_results[:rerank_count]
                reranked = await self.reranker.rerank(query, candidates)

                # Update final scores with rerank scores
                for result in reranked:
                    result.final_score = result.rerank_score or result.hybrid_score

                # Re-sort by final score
                search_results = reranked + search_results[rerank_count:]
                search_results.sort(key=lambda x: x.final_score, reverse=True)

        # Return top_k results
        final_results = search_results[:top_k]

        logger.debug(
            f"Hybrid search completed: query_len={len(query)}, "
            f"candidates={len(all_doc_ids)}, final_results={len(final_results)}"
        )

        return final_results

    def get_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        stats = {
            "num_documents": len(self.documents),
            "has_embeddings": self.embeddings is not None,
            "embedding_dim": (
                self.embeddings.shape[1] if self.embeddings is not None else None
            ),
            "alpha": self.alpha,
            "reranking_enabled": self.enable_reranking,
        }

        # Add BM25 stats
        if self.documents:
            stats.update(self.bm25.get_term_stats())

        return stats

    def update_alpha(self, new_alpha: float) -> None:
        """Update hybrid search weight."""
        if 0.0 <= new_alpha <= 1.0:
            self.alpha = new_alpha
            logger.info(f"Updated hybrid search alpha to {new_alpha}")
        else:
            raise ValueError("Alpha must be between 0.0 and 1.0")


def create_hybrid_search_engine(
    enable_hybrid: bool = True, alpha: float = 0.7, enable_reranking: bool = False
) -> HybridSearchEngine:
    """Create hybrid search engine with configuration."""
    if not enable_hybrid:
        alpha = 0.0  # BM25 only

    return HybridSearchEngine(
        alpha=alpha,
        enable_reranking=enable_reranking,
        reranker_model=settings.reranker_model,
    )
