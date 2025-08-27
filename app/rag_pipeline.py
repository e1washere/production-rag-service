"""RAG Pipeline implementation with FAISS indexing and retrieval."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Document:
    """Document representation for RAG pipeline."""

    def __init__(self, id: str, text: str, metadata: dict[str, Any] | None = None):
        self.id = id
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(id='{self.id}', text='{self.text[:50]}...')"


class RAGPipeline:
    """RAG Pipeline with FAISS indexing and retrieval capabilities."""

    def __init__(
        self,
        embedding_model: str | None = None,
        index_dir: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            embedding_model: Name of the sentence transformer model
            index_dir: Directory to store/load FAISS index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.index_dir = Path(index_dir or settings.index_dir)
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Lazy loading components
        self._embedder: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._documents: list[Document] | None = None

        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load the sentence transformer embedder."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def index(self) -> faiss.Index:
        """Lazy load the FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index

    @property
    def documents(self) -> list[Document]:
        """Lazy load the documents."""
        if self._documents is None:
            self._load_documents()
        return self._documents

    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        index_path = self.index_dir / "faiss.index"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

    def _load_documents(self) -> None:
        """Load documents from disk."""
        docs_path = self.index_dir / "documents.json"
        if docs_path.exists():
            import json

            with open(docs_path) as f:
                data = json.load(f)
                self._documents = [
                    Document(doc["id"], doc["text"], doc.get("metadata", {}))
                    for doc in data
                ]
        else:
            raise FileNotFoundError(f"Documents not found at {docs_path}")

    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        index_path = self.index_dir / "faiss.index"
        faiss.write_index(self._index, str(index_path))

    def _save_documents(self) -> None:
        """Save documents to disk."""
        docs_path = self.index_dir / "documents.json"
        import json

        data = [
            {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
            for doc in self._documents
        ]
        with open(docs_path, "w") as f:
            json.dump(data, f, indent=2)

    def ingest(self, docs_dir: str) -> None:
        """Ingest documents from directory and build FAISS index.

        Args:
            docs_dir: Directory containing documents to ingest
        """
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            UnstructuredMarkdownLoader,
        )
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        # Load documents
        documents = []
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
                try:
                    if file_path.suffix.lower() == ".txt":
                        loader = TextLoader(str(file_path))
                    elif file_path.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == ".md":
                        loader = UnstructuredMarkdownLoader(str(file_path))
                    else:
                        continue

                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        if not documents:
            raise ValueError(f"No documents found in {docs_dir}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        # Convert to our Document format
        self._documents = [
            Document(id=f"chunk_{i}", text=chunk.page_content, metadata=chunk.metadata)
            for i, chunk in enumerate(chunks)
        ]

        # Create embeddings
        texts = [doc.text for doc in self._documents]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(
            dimension
        )  # Inner product for cosine similarity
        self._index.add(embeddings.astype("float32"))

        # Save index and documents
        self._save_index()
        self._save_documents()

        print(f"Built FAISS index with {len(self._documents)} documents")

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve top-k documents for a query.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of top-k documents with scores
        """
        # Encode query
        query_embedding = self.embedder.encode([query])

        # Search index
        scores, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )

        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.metadata["score"] = float(score)
                results.append(doc)

        return results

    def generate(
        self, query: str, contexts: list[Document], llm: Callable[[str, list[str]], str]
    ) -> str:
        """Generate answer using retrieved contexts and LLM.

        Args:
            query: User query
            contexts: Retrieved document contexts
            llm: LLM callable that takes (query, contexts) and returns answer

        Returns:
            Generated answer
        """
        context_texts = [doc.text for doc in contexts]
        return llm(query, context_texts)

    def run(
        self, query: str, llm: Callable[[str, list[str]], str], k: int = 5
    ) -> dict[str, Any]:
        """Run complete RAG pipeline: retrieve + generate.

        Args:
            query: User query
            llm: LLM callable
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, contexts, and timing information
        """
        start_time = time.time()

        # Retrieve documents
        retrieve_start = time.time()
        contexts = self.retrieve(query, k)
        retrieve_time = (time.time() - retrieve_start) * 1000

        # Generate answer
        generate_start = time.time()
        answer = self.generate(query, contexts, llm)
        generate_time = (time.time() - generate_start) * 1000

        total_time = (time.time() - start_time) * 1000

        # Log metrics to MLflow if configured
        if settings.mlflow_tracking_uri:
            try:
                mlflow.log_metric("latency_retrieve_ms", retrieve_time)
                mlflow.log_metric("latency_generate_ms", generate_time)
                mlflow.log_metric("total_latency_ms", total_time)
                mlflow.log_param("top_k", k)
                mlflow.log_param("embedding_model", self.embedding_model_name)
            except Exception as e:
                print(f"Failed to log metrics to MLflow: {e}")

        return {
            "answer": answer,
            "contexts": [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": doc.metadata.get("score", 0.0),
                    "metadata": doc.metadata,
                }
                for doc in contexts
            ],
            "timing": {
                "retrieve_ms": retrieve_time,
                "generate_ms": generate_time,
                "total_ms": total_time,
            },
        }
