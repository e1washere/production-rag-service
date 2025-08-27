"""Simple FastAPI application for testing."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="RAG Service",
    description="Simple RAG service for testing",
    version="0.1.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def root():
    return {"message": "RAG Service is running", "version": "0.1.0"}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.get("/docs")
async def docs():
    return {"message": "API documentation available at /docs"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return QueryResponse(
        answer=f"Mock response to: {request.question}",
        sources=["mock_source_1", "mock_source_2"]
    )

@app.get("/metrics")
async def metrics():
    return {"requests_total": 0, "latency_p95": 0.1}
