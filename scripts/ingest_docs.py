#!/usr/bin/env python3
"""Document ingestion script for RAG service."""

import argparse
import logging
import sys
from pathlib import Path

from app.config import settings
from app.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents for RAG service")
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="data/docs",
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=".cache/index",
        help="Directory to store FAISS index",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=settings.chunk_size,
        help="Size of text chunks",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=settings.embedding_model,
        help="Sentence transformer model name",
    )

    args = parser.parse_args()

    # Validate inputs
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {args.docs_dir}")
        sys.exit(1)

    # Check for supported file types
    supported_extensions = {".txt", ".md", ".pdf"}
    found_files = []
    for file_path in docs_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            found_files.append(file_path)

    if not found_files:
        logger.error(f"No supported files found in {args.docs_dir}")
        logger.info(f"Supported extensions: {supported_extensions}")
        sys.exit(1)

    logger.info(f"Found {len(found_files)} documents to ingest")
    for file_path in found_files:
        logger.info(f"  - {file_path}")

    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=args.embedding_model,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        # Ingest documents
        logger.info("Starting document ingestion...")
        pipeline.ingest(args.docs_dir)

        logger.info("Document ingestion completed successfully!")
        logger.info(f"Index saved to: {args.index_dir}")
        logger.info(f"Total documents: {len(pipeline.documents)}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
