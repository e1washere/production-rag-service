# Contributing to Production RAG Service

Thank you for your interest in contributing to the Production RAG Service! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to professional standards. Please:

- Be respectful and constructive in all interactions
- Focus on technical merit and evidence-based discussions
- Follow established coding standards and practices
- Maintain confidentiality of any sensitive information

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (for containerized development)
- Azure CLI (for deployment testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/production-rag-service.git
   cd production-rag-service
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/e1washere/production-rag-service.git
   ```

## Development Setup

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### Branch Strategy

- Create feature branches from `main`
- Use descriptive branch names: `feature/amazing-feature`, `fix/bug-description`
- Keep branches focused and small

### Commit Guidelines

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(retrieval): add cross-encoder reranking support
fix(api): resolve memory leak in concurrent requests
docs(readme): update deployment instructions
test(pipeline): add integration tests for hybrid search
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for public methods
- Keep functions small and focused
- Use meaningful variable names

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test files
pytest tests/test_api.py -v

# Run tests in parallel
pytest -n auto
```

### Test Requirements

- Maintain test coverage above 80%
- Write tests for new functionality
- Update tests when modifying existing code
- Include integration tests for API endpoints
- Add operational tests for deployment procedures

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Operational Tests**: Test deployment and operational procedures

## Code Quality

### Linting and Formatting

```bash
# Run all quality checks
make lint

# Format code
black app/ tests/

# Check types
mypy app/

# Security checks
bandit -r app/
```

### Quality Standards

- All code must pass linting checks
- Type checking must pass with mypy
- No security vulnerabilities (bandit)
- Code must be properly formatted (black)

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Add or update documentation for new features
2. **Add Tests**: Include tests for new functionality
3. **Update CHANGELOG**: Add entries to CHANGELOG.md
4. **Check Quality**: Ensure all quality checks pass

### Pull Request Template

Use the following template for pull requests:

```markdown
## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] End-to-end tests pass
- [ ] Code quality checks pass

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)

## Additional Notes

Any additional information or context.
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer must approve
3. **Testing**: Ensure tests pass in CI environment
4. **Documentation**: Verify documentation is updated

## Release Process

### Version Management

- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `app/__init__.py`
- Update CHANGELOG.md with release notes
- Create git tags for releases

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Release notes prepared
- [ ] Deployment tested

## Areas for Contribution

### High Priority

- Performance optimizations
- Additional LLM provider integrations
- Enhanced evaluation metrics
- Security improvements
- Documentation improvements

### Medium Priority

- Additional retrieval algorithms
- New caching strategies
- Monitoring enhancements
- Testing improvements

### Low Priority

- UI/UX improvements
- Additional deployment options
- Tool integrations

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check existing documentation first

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes

Thank you for contributing to the Production RAG Service!
