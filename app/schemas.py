"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    version: str = Field(default="1.0.0", description="Service version")
    index_loaded: bool = Field(..., description="Whether search index is loaded")
    embedding_model: str = Field(..., description="Embedding model name")
    total_documents: int = Field(ge=0, description="Number of documents in index")
    cache_available: bool = Field(
        default=False, description="Whether cache is available"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "index_loaded": True,
                "embedding_model": "all-MiniLM-L6-v2",
                "total_documents": 150,
                "cache_available": True,
            }
        }


class DocumentMetadata(BaseModel):
    """Document metadata schema."""

    source: str | None = Field(None, description="Document source file")
    title: str | None = Field(None, description="Document title")
    author: str | None = Field(None, description="Document author")
    created_at: datetime | None = Field(None, description="Document creation date")
    tags: list[str] = Field(default_factory=list, description="Document tags")
    chunk_index: int | None = Field(
        None, ge=0, description="Chunk index within document"
    )

    class Config:
        schema_extra = {
            "example": {
                "source": "ai_fundamentals.txt",
                "title": "AI Fundamentals",
                "author": "AI Expert",
                "created_at": "2024-01-01T12:00:00Z",
                "tags": ["ai", "fundamentals"],
                "chunk_index": 0,
            }
        }


class RetrievedDocument(BaseModel):
    """Retrieved document schema."""

    id: str = Field(..., description="Document unique identifier")
    text: str = Field(..., min_length=1, description="Document text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

    class Config:
        schema_extra = {
            "example": {
                "id": "doc_001_chunk_0",
                "text": "Artificial Intelligence (AI) is the simulation of human intelligence...",
                "score": 0.85,
                "metadata": {
                    "source": "ai_fundamentals.txt",
                    "title": "AI Fundamentals",
                    "chunk_index": 0,
                },
            }
        }


class SearchTimings(BaseModel):
    """Search operation timings."""

    retrieve_ms: float = Field(..., ge=0, description="Retrieval time in milliseconds")
    generate_ms: float = Field(..., ge=0, description="Generation time in milliseconds")
    rerank_ms: float | None = Field(
        None, ge=0, description="Reranking time in milliseconds"
    )
    total_ms: float = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "retrieve_ms": 150.5,
                "generate_ms": 1200.8,
                "rerank_ms": 45.2,
                "total_ms": 1396.5,
            }
        }


class CostBreakdown(BaseModel):
    """Cost breakdown schema."""

    provider: str = Field(..., description="LLM provider name")
    model: str = Field(..., description="Model name")
    input_tokens: int = Field(..., ge=0, description="Input tokens used")
    output_tokens: int = Field(..., ge=0, description="Output tokens generated")
    input_cost: float = Field(..., ge=0, description="Input cost in USD")
    output_cost: float = Field(..., ge=0, description="Output cost in USD")
    total_cost: float = Field(..., ge=0, description="Total cost in USD")

    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "input_tokens": 150,
                "output_tokens": 200,
                "input_cost": 0.000225,
                "output_cost": 0.0004,
                "total_cost": 0.000625,
            }
        }


class QueryRequest(BaseModel):
    """Query request schema."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="User question to answer"
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of documents to retrieve"
    )
    include_sources: bool = Field(
        default=True, description="Whether to include source documents in response"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="LLM temperature override"
    )
    max_tokens: int | None = Field(
        None, ge=1, le=4000, description="Maximum tokens to generate"
    )
    filters: dict[str, Any] | None = Field(
        None, description="Metadata filters for document retrieval"
    )
    enable_hybrid_search: bool | None = Field(
        None, description="Enable hybrid BM25 + embeddings search"
    )
    enable_reranking: bool | None = Field(
        None, description="Enable cross-encoder reranking"
    )

    @validator("question")
    def validate_question(cls, v):
        """Validate question content."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

    class Config:
        schema_extra = {
            "example": {
                "question": "What is machine learning and how does it work?",
                "top_k": 3,
                "include_sources": True,
                "temperature": 0.1,
                "max_tokens": 500,
                "filters": {"tags": ["ml", "fundamentals"]},
                "enable_hybrid_search": True,
                "enable_reranking": False,
            }
        }


class QueryResponse(BaseModel):
    """Query response schema."""

    answer: str = Field(..., description="Generated answer")
    sources: list[RetrievedDocument] = Field(
        default_factory=list, description="Source documents used for answer"
    )
    timings: SearchTimings = Field(..., description="Performance timings")
    cost: CostBreakdown | None = Field(None, description="Cost breakdown")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    correlation_id: str = Field(..., description="Request correlation ID")

    class Config:
        schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    {
                        "id": "doc_001_chunk_0",
                        "text": "Machine learning is a method of data analysis...",
                        "score": 0.92,
                        "metadata": {"source": "ml_basics.txt"},
                    }
                ],
                "timings": {
                    "retrieve_ms": 120.5,
                    "generate_ms": 1150.2,
                    "total_ms": 1270.7,
                },
                "cost": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "total_cost": 0.000625,
                },
                "correlation_id": "abc12345",
            }
        }


class StatsResponse(BaseModel):
    """Statistics response schema."""

    total_documents: int = Field(ge=0, description="Total documents in index")
    unique_terms: int = Field(ge=0, description="Unique terms in BM25 index")
    avg_document_length: float = Field(ge=0, description="Average document length")
    embedding_dimension: int | None = Field(
        None, description="Embedding vector dimension"
    )
    cache_stats: dict[str, Any] = Field(
        default_factory=dict, description="Cache statistics"
    )
    cost_stats: dict[str, Any] = Field(
        default_factory=dict, description="Cost statistics"
    )
    search_config: dict[str, Any] = Field(
        default_factory=dict, description="Search configuration"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_documents": 150,
                "unique_terms": 5420,
                "avg_document_length": 512.3,
                "embedding_dimension": 384,
                "cache_stats": {"enabled": True, "hit_rate": 0.75, "total_keys": 1250},
                "cost_stats": {
                    "total_requests": 1000,
                    "total_cost_usd": 2.45,
                    "avg_cost_per_request": 0.00245,
                },
                "search_config": {
                    "hybrid_search": True,
                    "reranking": False,
                    "alpha": 0.7,
                },
            }
        }


class IngestRequest(BaseModel):
    """Document ingestion request schema."""

    documents: list[str] = Field(
        ..., min_items=1, max_items=1000, description="List of document texts to ingest"
    )
    metadata_list: list[DocumentMetadata] | None = Field(
        None, description="Metadata for each document (optional)"
    )
    chunk_size: int | None = Field(
        None, ge=100, le=2000, description="Text chunk size override"
    )
    chunk_overlap: int | None = Field(
        None, ge=0, le=500, description="Chunk overlap size override"
    )
    invalidate_cache: bool = Field(
        default=True, description="Whether to invalidate existing cache"
    )

    @validator("metadata_list")
    def validate_metadata_list(cls, v, values):
        """Validate metadata list matches documents."""
        if v is not None:
            documents = values.get("documents", [])
            if len(v) != len(documents):
                raise ValueError(
                    "Metadata list length must match documents list length"
                )
        return v

    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    "This is the first document about AI...",
                    "This is the second document about ML...",
                ],
                "metadata_list": [
                    {"source": "ai_doc.txt", "tags": ["ai"]},
                    {"source": "ml_doc.txt", "tags": ["ml"]},
                ],
                "chunk_size": 512,
                "chunk_overlap": 50,
                "invalidate_cache": True,
            }
        }


class IngestResponse(BaseModel):
    """Document ingestion response schema."""

    success: bool = Field(..., description="Whether ingestion succeeded")
    documents_processed: int = Field(ge=0, description="Number of documents processed")
    chunks_created: int = Field(ge=0, description="Number of text chunks created")
    processing_time_ms: float = Field(
        ge=0, description="Processing time in milliseconds"
    )
    index_updated: bool = Field(..., description="Whether search index was updated")
    cache_invalidated: bool = Field(
        default=False, description="Whether cache was invalidated"
    )
    errors: list[str] = Field(default_factory=list, description="Any processing errors")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "documents_processed": 2,
                "chunks_created": 8,
                "processing_time_ms": 1250.5,
                "index_updated": True,
                "cache_invalidated": True,
                "errors": [],
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    correlation_id: str = Field(..., description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid query parameters",
                "error_code": "VALIDATION_ERROR",
                "details": {
                    "field": "top_k",
                    "message": "Value must be between 1 and 50",
                },
                "correlation_id": "abc12345",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }
