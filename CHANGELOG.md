# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-modal retrieval support (text + images)
- Semantic caching improvements
- Query expansion and reformulation
- Multi-agent orchestration framework

### Changed
- Enhanced hybrid retrieval algorithm
- Improved cost tracking accuracy

### Fixed
- Memory leak in long-running sessions
- Race condition in concurrent requests

## [1.0.0] - 2024-01-15

### Added
- Production-grade RAG service with FastAPI
- Hybrid retrieval system (BM25 + dense embeddings)
- Cross-encoder reranking support
- Redis caching with TTL and versioning
- Comprehensive observability (Langfuse, App Insights, Prometheus)
- Structured JSON logging with correlation IDs
- Cost tracking and optimization
- Resilience patterns (retries, circuit breakers, rate limiting)
- Canary deployment with automatic rollback
- Comprehensive test suite (unit, integration, e2e)
- RAGAS-based offline evaluation
- Infrastructure as Code with Terraform
- Azure Container Apps deployment
- GitHub Actions CI/CD pipeline
- Operational procedures (SLOs, runbooks)
- API documentation with OpenAPI
- Health checks and monitoring endpoints

### Technical Features
- Python 3.11 with type hints
- FAISS vector similarity search
- Sentence Transformers for embeddings
- Rank BM25 for sparse retrieval
- Uvicorn ASGI server
- Pydantic for data validation
- Azure Managed Identity integration
- OIDC authentication for CI/CD

### Infrastructure
- Azure Container Registry for image storage
- Azure Key Vault for secret management
- Azure Redis Cache for distributed caching
- Azure Application Insights for monitoring
- Prometheus metrics collection
- Structured logging pipeline

### Testing & Quality
- 85% test coverage
- Unit tests for all components
- Integration tests for API endpoints
- End-to-end tests for full pipeline
- Operational tests for deployment procedures
- Code quality tools (ruff, black, mypy)
- Security scanning and dependency checks

### Documentation
- Comprehensive README with architecture diagram
- API documentation with examples
- Deployment guides and troubleshooting
- Contributing guidelines
- FAQ section for common questions
- Interview preparation materials

## [0.9.0] - 2024-01-10

### Added
- Initial RAG pipeline implementation
- Basic FastAPI application structure
- Simple document retrieval
- Mock LLM provider for development
- Basic health check endpoint

### Changed
- Refactored project structure for scalability
- Improved error handling

### Fixed
- Memory issues with large document sets
- API response formatting

## [0.8.0] - 2024-01-05

### Added
- Document ingestion pipeline
- Basic vector search functionality
- Simple caching mechanism
- Initial test framework

### Changed
- Updated dependencies to latest versions
- Improved code organization

## [0.7.0] - 2024-01-01

### Added
- Project initialization
- Basic project structure
- Development environment setup
- Initial documentation

---

## Version History

- **1.0.0**: Production-ready release with full MLOps capabilities
- **0.9.0**: Core RAG functionality with basic API
- **0.8.0**: Document processing and search capabilities
- **0.7.0**: Project foundation and setup

## Migration Guide

### From 0.9.0 to 1.0.0

1. **Environment Variables**: Update `.env` file with new observability and deployment variables
2. **API Changes**: Query endpoint now returns enhanced metadata
3. **Deployment**: Switch to Azure Container Apps deployment
4. **Testing**: Update test configuration for new endpoints

### From 0.8.0 to 0.9.0

1. **Dependencies**: Update requirements.txt with new packages
2. **Configuration**: Add LLM provider configuration
3. **API**: Update client code for new response format

## Support

For questions about version compatibility or migration issues, please refer to the [FAQ](README.md#faq) section or open an issue on GitHub.
