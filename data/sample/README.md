# Sample Corpus

This directory contains a curated set of documents for testing and evaluation of the RAG service.

## Contents

- `ai_fundamentals.txt` - Comprehensive overview of AI and ML concepts
- `machine_learning.txt` - Detailed explanation of machine learning
- `deep_learning.txt` - Introduction to deep learning and neural networks
- `nlp_basics.txt` - Natural language processing fundamentals
- `computer_vision.txt` - Computer vision concepts and applications

## Usage

```bash
# Ingest the sample corpus
make ingest-sample

# Run evaluation
make eval-sample

# Query the service
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 3}'
```

## Expected Results

With this corpus, you should achieve:
- Hit Rate @3: ~85-90%
- Hit Rate @5: ~92-95%
- Average retrieval time: <500ms
- Context relevance scores: >0.7
