"""FastAPI application for RAG service."""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.rag_pipeline import RAGPipeline
from app.routers.health import router as health_router
from app.routers.metrics import setup_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Production-ready RAG microservice with observability and evaluation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["health"])

# Setup Prometheus metrics
setup_metrics(app)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="User question")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of documents to retrieve"
    )


class ContextResponse(BaseModel):
    """Response model for retrieved context."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    contexts: list[ContextResponse]
    timings: dict[str, float]


# Global RAG pipeline instance (lazy loaded)
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_llm() -> callable:
    """Get LLM callable based on configuration."""
    try:
        from app.llm_providers import create_llm_provider

        provider_type = settings.llm_provider.lower()

        if provider_type == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            provider = create_llm_provider(
                provider_type="openai",
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
            )
        elif provider_type == "groq":
            if not settings.groq_api_key:
                raise ValueError("Groq API key not configured")
            provider = create_llm_provider(
                provider_type="groq",
                api_key=settings.groq_api_key,
                model=settings.groq_model,
            )
        elif provider_type == "ollama":
            provider = create_llm_provider(
                provider_type="ollama",
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
        else:
            provider = create_llm_provider("mock")

        logger.info(f"Using LLM provider: {provider.get_provider_name()}")
        return provider.generate

    except Exception as e:
        logger.warning(f"Failed to initialize LLM provider: {e}, falling back to mock")
        from app.llm_providers import MockProvider

        return MockProvider().generate


# Observability setup
def setup_observability():
    """Setup Langfuse/LangSmith observability."""
    try:
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse

            langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse observability enabled")
            return langfuse
        elif settings.langsmith_api_key:
            import os

            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            logger.info("LangSmith observability enabled")
            return None
    except Exception as e:
        logger.warning(f"Failed to setup observability: {e}")
    return None


# Initialize observability
langfuse = setup_observability()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting RAG Service...")

    # Setup MLflow if configured
    if settings.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        logger.info(f"MLflow tracking enabled: {settings.mlflow_tracking_uri}")

    # Auto-ingest if index missing and auto_ingest_on_startup enabled
    try:
        pipeline = get_rag_pipeline()
        _ = pipeline.index  # triggers load
        _ = pipeline.documents
        logger.info("FAISS index loaded on startup")
    except Exception:
        if settings.auto_ingest_on_startup:
            try:
                docs_dir = (
                    settings.sample_docs_dir
                    if Path(settings.docs_dir).exists() is False
                    and Path(settings.sample_docs_dir).exists()
                    else settings.docs_dir
                )
                logger.info(f"Index not found. Auto-ingesting from: {docs_dir}")
                pipeline = get_rag_pipeline()
                pipeline.ingest(docs_dir)
                logger.info("Auto-ingest completed successfully")
            except Exception as e:
                logger.warning(f"Auto-ingest failed: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG Service is running", "version": "0.1.0"}


# Remove the old health endpoint - now handled by router


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query through the RAG pipeline."""
    try:
        pipeline = get_rag_pipeline()
        llm = get_llm()

        # Run RAG pipeline (simplified for now)
        result = pipeline.run(request.question, llm, request.top_k)

        # Log request
        logger.info(f"Processed query: {request.question[:100]}...")

        # Handle both dict and Document objects in contexts
        contexts = []
        for ctx in result["contexts"]:
            if hasattr(ctx, 'id'):  # Document object
                contexts.append(ContextResponse(
                    id=ctx.id,
                    text=ctx.text,
                    score=ctx.metadata.get("score", 0.0),
                    metadata=ctx.metadata,
                ))
            else:  # Dict object (legacy)
                contexts.append(ContextResponse(
                    id=ctx["id"],
                    text=ctx["text"],
                    score=ctx["score"],
                    metadata=ctx.get("metadata", {}),
                ))
        
        return QueryResponse(
            answer=result["answer"],
            contexts=contexts,
            timings=result.get("timings", result.get("timing", {})),
        )

    except FileNotFoundError as e:
        logger.error(f"Index not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="RAG index not available. Please run ingestion first.",
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics."""
    try:
        pipeline = get_rag_pipeline()
        return {
            "total_documents": len(pipeline.documents),
            "embedding_model": pipeline.embedding_model_name,
            "chunk_size": pipeline.chunk_size,
            "chunk_overlap": pipeline.chunk_overlap,
            "index_type": "FAISS",
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app", host=settings.host, port=settings.port, reload=settings.debug
    )
