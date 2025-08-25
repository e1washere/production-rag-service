.PHONY: help install dev-install run test test-e2e lint format clean ingest ingest-sample eval eval-sample docker-build docker-run docker-clean setup ci dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

dev: ## Run development server with hot reload
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

run: ## Run the production server
	uvicorn app.api:app --host 0.0.0.0 --port 8000

test: ## Run all tests
	pytest tests/ -v --cov=app --cov-report=term-missing

test-e2e: ## Run end-to-end tests
	pytest tests/test_e2e.py -v

test-watch: ## Run tests in watch mode
	pytest tests/ -v --cov=app --cov-report=term-missing -f

format: ## Format code with black and ruff
	python3 -m ruff check app tests scripts --fix || true
	python3 -m black app tests scripts

lint: ## Run linting checks
	python3 -m ruff check app tests scripts
	python3 -m mypy app

test: ## Run tests with coverage
	python3 -m pytest -q --maxfail=1 --disable-warnings --cov=app --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	python3 -m pytest -q --maxfail=1 --disable-warnings

up: ## Start the development server
	python3 -c "import sys; sys.path.append('.'); from app.api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)"

up-prod: ## Start production server
	python3 -c "import sys; sys.path.append('.'); from app.api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .cache/

ingest: ## Ingest documents from data/docs
	python scripts/ingest_docs.py --docs_dir data/docs --index_dir .cache/index

ingest-sample: ## Ingest sample corpus
	python scripts/ingest_docs.py --docs_dir data/sample --index_dir .cache/index

eval: ## Run evaluation on full dataset
	python -m app.eval_offline --eval_dir data/eval --index_dir .cache/index --output_dir .cache/eval

eval-sample: ## Run evaluation on sample dataset
	python -m app.eval_offline --eval_dir data/eval --index_dir .cache/index --output_dir .cache/eval --eval_file sample_eval.jsonl

docker-build: ## Build Docker image
	docker build -t rag-service:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env rag-service:latest

docker-clean: ## Clean Docker images
	docker rmi rag-service:latest || true

setup: dev-install ## Setup development environment
	mkdir -p data/docs data/eval data/sample .cache/index .cache/eval
	cp .env.example .env
	make ingest-sample

ci: lint test ## Run CI checks locally

# Azure deployment commands
azure-login: ## Login to Azure CLI
	az login

azure-deploy: ## Deploy to Azure App Service
	./scripts/deploy_azure.sh

azure-destroy: ## Destroy Azure infrastructure
	cd infra/terraform && terraform destroy -auto-approve

# Infrastructure commands
terraform-init: ## Initialize Terraform
	cd infra/terraform && terraform init

terraform-plan: ## Plan Terraform changes
	cd infra/terraform && terraform plan

terraform-apply: ## Apply Terraform changes
	cd infra/terraform && terraform apply -auto-approve
