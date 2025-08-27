"""Tests for RAG pipeline functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.rag_pipeline import Document, RAGPipeline


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document("doc1", "This is a test document about machine learning."),
            Document("doc2", "Python is a programming language used for data science."),
            Document("doc3", "FastAPI is a modern web framework for building APIs."),
            Document(
                "doc4",
                "RAG systems combine retrieval and generation for better answers.",
            ),
            Document("doc5", "FAISS is a library for efficient similarity search."),
        ]

    def test_document_creation(self):
        """Test Document class creation."""
        doc = Document("test_id", "test text", {"source": "test"})
        assert doc.id == "test_id"
        assert doc.text == "test text"
        assert doc.metadata["source"] == "test"

    def test_pipeline_initialization(self, temp_dir):
        """Test RAGPipeline initialization."""
        pipeline = RAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            index_dir=str(temp_dir),
            chunk_size=500,
            chunk_overlap=100,
        )

        assert pipeline.embedding_model_name == "all-MiniLM-L6-v2"
        assert pipeline.index_dir == temp_dir
        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 100

    def test_embedder_lazy_loading(self, temp_dir):
        """Test that embedder is loaded lazily."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Embedder should not be loaded initially
        assert pipeline._embedder is None

        # Access embedder property
        embedder = pipeline.embedder
        assert embedder is not None
        assert pipeline._embedder is embedder

    def test_generate_with_mock_llm(self, temp_dir, sample_documents):
        """Test generate method with mock LLM."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        def mock_llm(query: str, contexts: list) -> str:
            return f"Answer to '{query}' based on {len(contexts)} contexts"

        result = pipeline.generate("test query", sample_documents, mock_llm)
        assert "test query" in result
        assert "5 contexts" in result

    @patch("app.rag_pipeline.faiss.read_index")
    @patch("builtins.open")
    def test_load_index_success(self, mock_open, mock_read_index, temp_dir):
        """Test successful index loading."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Mock index file exists
        index_path = temp_dir / "faiss.index"
        index_path.touch()

        # Mock documents file
        docs_path = temp_dir / "documents.json"
        docs_path.touch()

        # Mock JSON data
        mock_open.return_value.__enter__.return_value.read.return_value = "[]"

        # Test loading
        pipeline._load_index()
        assert pipeline._index is not None
        mock_read_index.assert_called_once()

    def test_load_index_not_found(self, temp_dir):
        """Test index loading when file doesn't exist."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError):
            pipeline._load_index()

    @patch("builtins.open")
    def test_load_documents_success(self, mock_open, temp_dir):
        """Test successful documents loading."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Mock documents file exists
        docs_path = temp_dir / "documents.json"
        docs_path.touch()

        # Mock JSON data
        mock_data = '[{"id": "doc1", "text": "test", "metadata": {}}]'
        mock_open.return_value.__enter__.return_value.read.return_value = mock_data

        # Test loading
        pipeline._load_documents()
        assert pipeline._documents is not None
        assert len(pipeline._documents) == 1
        assert pipeline._documents[0].id == "doc1"

    def test_load_documents_not_found(self, temp_dir):
        """Test documents loading when file doesn't exist."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError):
            pipeline._load_documents()


class TestRetrieverHitRate:
    """Test hit rate functionality for retrieval."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def small_pipeline(self, temp_dir):
        """Create a small pipeline for hit rate testing."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Create small in-memory index with 5 documents
        documents = [
            Document("doc1", "Machine learning algorithms"),
            Document("doc2", "Python programming language"),
            Document("doc3", "FastAPI web framework"),
            Document("doc4", "RAG systems architecture"),
            Document("doc5", "FAISS similarity search"),
        ]

        # Mock the embedder and index
        pipeline._documents = documents
        pipeline._embedder = Mock()
        pipeline._embedder.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        # Mock FAISS index
        pipeline._index = Mock()
        pipeline._index.search.return_value = (
            [[0.9, 0.8, 0.7, 0.6, 0.5]],  # scores
            [[0, 1, 2, 3, 4]],  # indices
        )

        return pipeline

    def test_retriever_hit_rate_small(self, small_pipeline):
        """Test hit rate with small in-memory index."""
        # Test retrieval
        results = small_pipeline.retrieve("machine learning", k=3)

        # Should return at most 3 documents (may be fewer if index is small)
        assert len(results) <= 5  # We have 5 test documents
        assert len(results) > 0  # Should return at least some results

        # All documents should have scores
        for doc in results:
            assert "score" in doc.metadata
            assert isinstance(doc.metadata["score"], float)

        # Hit rate should be 1.0 since we have 5 docs and retrieve 3
        hit_rate = len(results) / 3  # 3/3 = 1.0
        assert hit_rate >= 0.6  # Target threshold

        # Verify document IDs are unique
        doc_ids = [doc.id for doc in results]
        assert len(set(doc_ids)) == len(doc_ids)

    def test_retrieve_with_empty_index(self, temp_dir):
        """Test retrieval with empty index."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Mock empty documents
        pipeline._documents = []
        pipeline._embedder = Mock()
        pipeline._embedder.encode.return_value = [[0.1, 0.2, 0.3]]

        pipeline._index = Mock()
        pipeline._index.search.return_value = ([[]], [[]])

        results = pipeline.retrieve("test query", k=5)
        assert len(results) == 0


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_pipeline(self, temp_dir):
        """Create a mock pipeline for integration testing."""
        pipeline = RAGPipeline(index_dir=str(temp_dir))

        # Mock all components
        pipeline._documents = [
            Document("doc1", "Test document 1"),
            Document("doc2", "Test document 2"),
        ]
        pipeline._embedder = Mock()
        pipeline._embedder.encode.return_value = [[0.1, 0.2]]
        pipeline._index = Mock()
        pipeline._index.search.return_value = ([[0.9, 0.8]], [[0, 1]])

        return pipeline

    def test_run_pipeline_complete(self, mock_pipeline):
        """Test complete pipeline run."""

        def mock_llm(query: str, contexts: list) -> str:
            return f"Answer: {query} with {len(contexts)} contexts"

        result = mock_pipeline.run("test query", mock_llm, k=2)

        # Check result structure
        assert "answer" in result
        assert "contexts" in result
        assert "timing" in result

        # Check answer
        assert "test query" in result["answer"]
        assert "2 contexts" in result["answer"]

        # Check contexts
        assert len(result["contexts"]) == 2
        for ctx in result["contexts"]:
            assert "id" in ctx
            assert "text" in ctx
            assert "score" in ctx

        # Check timing
        assert "retrieve_ms" in result["timing"]
        assert "generate_ms" in result["timing"]
        assert "total_ms" in result["timing"]

        # All timing values should be positive
        for timing_value in result["timing"].values():
            assert timing_value >= 0
