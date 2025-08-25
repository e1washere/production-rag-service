"""End-to-end tests for RAG service."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.rag_pipeline import RAGPipeline


class TestE2EFlow:
    """End-to-end tests for complete RAG workflow."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""

        def mock_generate(query, contexts):
            return f"Based on the context, here's what I found about: {query}"

        return mock_generate

    def test_complete_rag_flow(self, temp_index_dir, mock_llm):
        """Test complete flow: ingest → query → response."""

        # Create test documents
        docs_dir = temp_index_dir / "docs"
        docs_dir.mkdir()

        (docs_dir / "test1.txt").write_text(
            "Paris is the capital of France. It is known for the Eiffel Tower."
        )
        (docs_dir / "test2.txt").write_text(
            "Machine learning is a subset of artificial intelligence."
        )

        # Mock the LLM in the API
        with patch("app.api.get_llm", return_value=mock_llm):
            with TestClient(app) as client:
                # Test health before ingestion
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"

                # Test stats before ingestion
                response = client.get("/stats")
                assert response.status_code == 200
                data = response.json()
                # May have documents from previous runs
                assert "total_documents" in data

                # Test query (may work if index exists from previous runs)
                response = client.post(
                    "/query", json={"question": "What is the capital of France?"}
                )
                # Should either work (200) or fail gracefully (503)
                assert response.status_code in [200, 503]

                # Ingest documents
                pipeline = RAGPipeline(index_dir=str(temp_index_dir / "index"))
                pipeline.ingest(docs_dir)

                # Test query after ingestion
                with patch("app.api.get_rag_pipeline", return_value=pipeline):
                    response = client.post(
                        "/query", json={"question": "What is the capital of France?"}
                    )
                    assert response.status_code == 200
                    data = response.json()
                    assert "answer" in data
                    assert "contexts" in data
                    assert len(data["contexts"]) > 0

    def test_query_with_mock_pipeline(self):
        """Test query endpoint with mocked pipeline."""
        from app.rag_pipeline import Document
        
        mock_pipeline = MagicMock()
        mock_contexts = [Document(id="1", text="Mock context", metadata={"score": 0.9})]
        mock_pipeline.run.return_value = {
            "answer": "Mock answer",
            "contexts": mock_contexts,
            "timings": {"total_ms": 100.0},
        }

        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            with TestClient(app) as client:
                response = client.post(
                    "/query", json={"question": "Test question", "top_k": 3}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "Mock answer"
                assert len(data["contexts"]) == 1

    def test_query_validation(self):
        """Test query endpoint validation."""
        with TestClient(app) as client:
            # Test missing question
            response = client.post("/query", json={})
            assert response.status_code == 422

            # Test invalid top_k
            response = client.post(
                "/query", json={"question": "Test", "top_k": 25}
            )
            assert response.status_code == 422

    def test_api_endpoints(self):
        """Test all API endpoints."""
        with TestClient(app) as client:
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "version" in data

            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "checks" in data

            # Test stats endpoint
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_documents" in data

            # Test metrics endpoint
            response = client.get("/metrics")
            assert response.status_code == 200
            # Should return Prometheus metrics format


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_pipeline_ingest_and_retrieve(self, temp_dir):
        """Test pipeline ingestion and retrieval."""
        # Create test document
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text(
            "Paris is the capital city of France, located in the north-central part of the country."
        )

        # Test ingestion
        pipeline = RAGPipeline(index_dir=str(temp_dir / "index"))
        pipeline.ingest(docs_dir)

        # Test retrieval
        results = pipeline.retrieve("Where is Paris?", k=2)

        assert len(results) > 0
        assert all(hasattr(doc, "text") for doc in results)
        assert all("score" in doc.metadata for doc in results)
        assert all(hasattr(doc, "metadata") for doc in results)

    def test_pipeline_generate(self, temp_dir):
        """Test pipeline generation with mock LLM."""

        def mock_llm(query, contexts):
            return f"Mock response for: {query}"

        pipeline = RAGPipeline(index_dir=str(temp_dir / "index"))

        # Test generation with Document objects
        from app.rag_pipeline import Document

        context_docs = [
            Document(id="1", text="context1", metadata={}),
            Document(id="2", text="context2", metadata={}),
        ]
        answer = pipeline.generate("test query", context_docs, mock_llm)

        assert isinstance(answer, str)
        assert "test query" in answer
