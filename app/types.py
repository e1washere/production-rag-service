"""Type definitions for the RAG service."""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """A single retrieval result with score and metadata."""

    id: str = Field(..., description="Unique identifier for the document chunk")
    text: str = Field(..., description="The retrieved text content")
    score: float = Field(..., description="Relevance score for this retrieval")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_0",
                "text": "Machine learning is a subset of artificial intelligence...",
                "score": 0.85,
                "metadata": {"source": "data/docs/ml_guide.txt", "chunk_index": 0},
            }
        }


class QueryContext(BaseModel):
    """Context information for a query."""

    question: str = Field(..., description="The user's question")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to retrieve"
    )

    class Config:
        json_schema_extra = {
            "example": {"question": "What is machine learning?", "top_k": 5}
        }


class QueryResult(BaseModel):
    """Complete query result with answer and context."""

    answer: str = Field(..., description="Generated answer to the question")
    contexts: list[RetrievalHit] = Field(..., description="Retrieved context documents")
    timings: dict[str, float] = Field(
        default_factory=dict, description="Performance timings"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn...",
                "contexts": [
                    {
                        "id": "chunk_0",
                        "text": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.85,
                        "metadata": {"source": "data/docs/ml_guide.txt"},
                    }
                ],
                "timings": {
                    "retrieve_ms": 150.5,
                    "generate_ms": 500.2,
                    "total_ms": 650.7,
                },
            }
        }
