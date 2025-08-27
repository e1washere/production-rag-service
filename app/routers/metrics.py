"""Metrics router for Prometheus monitoring."""

from fastapi import APIRouter
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

router = APIRouter()

# Custom metrics
rag_queries_total = Counter(
    "rag_queries_total", "Total number of RAG queries processed", ["status", "provider"]
)

rag_query_duration = Histogram(
    "rag_query_duration_seconds",
    "Time spent processing RAG queries",
    ["component"],  # retrieve, generate, total
)

rag_retrieval_hits = Histogram(
    "rag_retrieval_hits_at_k",
    "Number of relevant hits at different k values",
    ["k_value"],
)

rag_cost_per_request = Histogram(
    "rag_cost_per_request_usd", "Cost per request in USD", ["provider", "model"]
)

index_documents_count = Gauge(
    "rag_index_documents_total", "Total number of documents in the index"
)

# Initialize instrumentator
instrumentator = Instrumentator()


def setup_metrics(app):
    """Setup Prometheus metrics for the FastAPI app."""

    # Add default FastAPI metrics
    instrumentator.instrument(app)

    # Add custom metrics endpoint
    instrumentator.expose(app, endpoint="/metrics")

    # Basic instrumentator provides default metrics

    return instrumentator


def record_query_metrics(
    status: str,
    provider: str,
    retrieve_time: float,
    generate_time: float,
    total_time: float,
    cost: float = 0.0,
    model: str = "unknown",
    hits_at_k: dict = None,
):
    """Record metrics for a RAG query."""

    # Record query count
    rag_queries_total.labels(status=status, provider=provider).inc()

    # Record timing metrics
    rag_query_duration.labels(component="retrieve").observe(retrieve_time)
    rag_query_duration.labels(component="generate").observe(generate_time)
    rag_query_duration.labels(component="total").observe(total_time)

    # Record cost metrics
    if cost > 0:
        rag_cost_per_request.labels(provider=provider, model=model).observe(cost)

    # Record retrieval hit metrics
    if hits_at_k:
        for k, hits in hits_at_k.items():
            rag_retrieval_hits.labels(k_value=str(k)).observe(hits)


def update_index_size(document_count: int):
    """Update the index document count metric."""
    index_documents_count.set(document_count)
