# Document Q&A API — portfolio prototype

A Python project exploring document retrieval and HTTP APIs. The repository contains a FAISS retrieval pipeline, separate BM25/hybrid-search experiments, and deployment examples.

**Status:** experimental and incomplete. The default container runs a mock API. This repository is not a production deployment or evidence of production operating experience.

## Implementation map

| Component | Source | Current behavior |
| --- | --- | --- |
| Mock HTTP API | [app/simple_api.py](app/simple_api.py) | Returns a canned answer and example sources; does not retrieve documents or call an LLM |
| Document ingestion and vector retrieval | [app/rag_pipeline.py](app/rag_pipeline.py) | Loads documents, splits text, creates embeddings, persists a FAISS index, and retrieves chunks |
| RAG API wiring | [app/api.py](app/api.py) | Calls `RAGPipeline`; its LLM-provider module is missing from this revision |
| Sparse and hybrid search | [app/search/](app/search/) | Separate BM25, weighted hybrid scoring, and reranking components; not connected to the `app.api` query path |
| Deployment examples | [infra/](infra/), [scripts/](scripts/) | Infrastructure and release experiments; not verified production deployments |

## Follow the document and request paths

The intended document path in `RAGPipeline.ingest()` is:

1. Load supported text, Markdown, or PDF files.
2. Split documents into chunks.
3. Encode chunks with a sentence-transformer model.
4. Write the FAISS index and document metadata to disk.

The vector-query path in `RAGPipeline.run()` encodes a question, retrieves chunks, and passes their text to an injected LLM callable. It does not invoke the separate hybrid-search engine.

The root [Dockerfile](Dockerfile) instead starts `app.simple_api:app`. Its successful HTTP response demonstrates the mock endpoint only.

## Inspect the mock API locally

The following is a minimal setup for the mock API, not the document-retrieval application. It has not been verified in a clean environment as part of this documentation update.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install fastapi uvicorn pydantic
python -m uvicorn app.simple_api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is retrieval?"}'
```

The handler constructs this illustrative response:

```json
{
  "answer": "Mock response to: What is retrieval?",
  "sources": ["mock_source_1", "mock_source_2"]
}
```

These sources are placeholders, not citations. The mock `/metrics` endpoint also returns fixed example values.

## Current limitations

- `app/api.py` imports `app.llm_providers`, which is absent from this revision. The document-grounded HTTP flow needs repair before it can be demonstrated.
- Hybrid retrieval and reranking exist as separate experiments; their presence does not establish end-to-end integration.
- The retrieval code needs boundary-case checks, including requests for more results than indexed documents.
- Files under `tests/` include test code and a retrieval-test placeholder. No current passing-suite or coverage claim is made.
- No reproducible RAGAS score, latency benchmark, nightly evaluation result, or production SLO is reported here.
- Redis caching, provider resilience, and deployment automation must be checked against the executed request path before being presented as working features.

## Next engineering milestone

Demonstrate one reproducible flow from a small local corpus through retrieval to an answer with traceable sources. Add focused checks for empty indexes, invalid requests, provider errors, and result counts before expanding the architecture.

## License

[MIT](LICENSE).
