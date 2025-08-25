"""Tests for FastAPI application."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.rag_pipeline import Document


class TestAPI:
    """Test cases for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock RAG pipeline."""
        pipeline = Mock()

        # Mock documents
        documents = [
            Document("doc1", "Test document 1"),
            Document("doc2", "Test document 2"),
        ]
        pipeline.documents = documents
        pipeline.embedding_model_name = "all-MiniLM-L6-v2"
        pipeline.chunk_size = 500
        pipeline.chunk_overlap = 100

        # Mock run method
        pipeline.run.return_value = {
            "answer": "Test answer",
            "contexts": [
                {"id": "doc1", "text": "Test document 1", "score": 0.9, "metadata": {}},
                {"id": "doc2", "text": "Test document 2", "score": 0.8, "metadata": {}},
            ],
            "timing": {"retrieve_ms": 50.0, "generate_ms": 100.0, "total_ms": 150.0},
        }

        return pipeline

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "RAG Service is running"

    def test_health_endpoint_success(self, client, mock_pipeline):
        """Test health endpoint with successful pipeline."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data
        assert data["checks"]["app"] is True
        assert data["checks"]["config"] is True

    @pytest.mark.skip("Recursion issue with Mock serialization")
    def test_health_endpoint_failure(self, client):
        """Test health endpoint with pipeline failure."""
        mock_pipeline = Mock()
        mock_pipeline.index = Mock(side_effect=FileNotFoundError("Index not found"))

        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            response = client.get("/health")
            assert response.status_code == 503

    def test_stats_endpoint(self, client, mock_pipeline):
        """Test stats endpoint."""
        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_documents"] == 2
            assert data["embedding_model"] == "all-MiniLM-L6-v2"
            assert data["chunk_size"] == 500
            assert data["chunk_overlap"] == 100
            assert data["index_type"] == "FAISS"

    def test_query_endpoint_success(self, client, mock_pipeline):
        """Test query endpoint with successful request."""
        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            request_data = {"question": "What is machine learning?", "top_k": 3}

            response = client.post("/query", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert "answer" in data
            assert "contexts" in data
            assert "timings" in data

            # Check answer
            assert data["answer"] == "Test answer"

            # Check contexts
            assert len(data["contexts"]) == 2
            for ctx in data["contexts"]:
                assert "id" in ctx
                assert "text" in ctx
                assert "score" in ctx
                assert "metadata" in ctx

            # Check timings
            assert "retrieve_ms" in data["timings"]
            assert "generate_ms" in data["timings"]
            assert "total_ms" in data["timings"]

    def test_query_endpoint_default_top_k(self, client, mock_pipeline):
        """Test query endpoint with default top_k."""
        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            request_data = {"question": "What is machine learning?"}

            response = client.post("/query", json=request_data)
            assert response.status_code == 200

            # Verify pipeline was called with default top_k=5
            mock_pipeline.run.assert_called_once()
            call_args = mock_pipeline.run.call_args
            assert call_args[0][2] == 5  # top_k parameter

    def test_query_endpoint_invalid_top_k(self, client):
        """Test query endpoint with invalid top_k."""
        request_data = {
            "question": "What is machine learning?",
            "top_k": 25,  # Too high
        }

        response = client.post("/query", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_missing_question(self, client):
        """Test query endpoint with missing question."""
        request_data = {"top_k": 5}

        response = client.post("/query", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_pipeline_error(self, client):
        """Test query endpoint with pipeline error."""
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = FileNotFoundError("Index not found")

        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            request_data = {"question": "What is machine learning?"}

            response = client.post("/query", json=request_data)
            assert response.status_code == 503
            data = response.json()
            assert "RAG index not available" in data["detail"]

    def test_query_endpoint_general_error(self, client):
        """Test query endpoint with general error."""
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = Exception("Unexpected error")

        with patch("app.api.get_rag_pipeline", return_value=mock_pipeline):
            request_data = {"question": "What is machine learning?"}

            response = client.post("/query", json=request_data)
            assert response.status_code == 500
            data = response.json()
            assert "Unexpected error" in data["detail"]


class TestAPIIntegration:
    """Integration tests for API functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_api_smoke(self, client):
        """Smoke test for API endpoints."""
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200

        # Test health endpoint (may fail if no index, but should not crash)
        response = client.get("/health")
        assert response.status_code in [
            200,
            503,
        ]  # Either healthy or service unavailable

        # Test stats endpoint (may fail if no index, but should not crash)
        response = client.get("/stats")
        assert response.status_code in [200, 500]  # Either success or internal error

    def test_query_request_validation(self, client):
        """Test query request validation."""
        # Valid request
        valid_request = {"question": "What is machine learning?", "top_k": 5}
        response = client.post("/query", json=valid_request)
        # Should either succeed or fail gracefully, but not crash
        assert response.status_code in [200, 503, 500]

        # Invalid request - missing question
        invalid_request = {"top_k": 5}
        response = client.post("/query", json=invalid_request)
        assert response.status_code == 422

        # Invalid request - top_k too high
        invalid_request = {"question": "What is machine learning?", "top_k": 25}
        response = client.post("/query", json=invalid_request)
        assert response.status_code == 422

        # Invalid request - top_k too low
        invalid_request = {"question": "What is machine learning?", "top_k": 0}
        response = client.post("/query", json=invalid_request)
        assert response.status_code == 422
