"""Configuration settings for the RAG service with Azure Key Vault integration."""

import logging
import os
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with Azure Key Vault integration."""

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # Azure Key Vault
    key_vault_url: str | None = Field(default=None, description="Azure Key Vault URL")
    use_key_vault: bool = Field(
        default=False, description="Enable Azure Key Vault for secrets"
    )

    # API Configuration
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Security
    api_key: str | None = Field(default=None, description="API key for authentication")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")

    # RAG Configuration
    index_dir: str = Field(default=".cache/index", description="Vector index directory")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(
        default=100, ge=0, le=500, description="Chunk overlap size"
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of retrieved documents"
    )

    # Hybrid Search Configuration
    enable_hybrid_search: bool = Field(
        default=True, description="Enable BM25 + embeddings hybrid search"
    )
    hybrid_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight (0=BM25, 1=embeddings)",
    )
    enable_reranker: bool = Field(
        default=False, description="Enable cross-encoder reranking"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-2-v2", description="Reranker model"
    )

    # LLM Configuration
    llm_provider: str = Field(
        default="mock", description="LLM provider: openai, groq, ollama, mock"
    )
    llm_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature"
    )
    llm_max_tokens: int = Field(
        default=1000, ge=1, le=4000, description="Max tokens for generation"
    )
    llm_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="LLM request timeout"
    )

    # OpenAI Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    openai_base_url: str | None = Field(default=None, description="OpenAI base URL")
    openai_max_retries: int = Field(
        default=3, ge=1, le=10, description="OpenAI max retries"
    )

    # Groq Configuration
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="llama3-8b-8192", description="Groq model name")
    groq_max_retries: int = Field(
        default=3, ge=1, le=10, description="Groq max retries"
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    ollama_model: str = Field(default="llama2", description="Ollama model name")
    ollama_timeout: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Ollama timeout"
    )

    # Caching Configuration
    enable_cache: bool = Field(default=True, description="Enable Redis caching")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    cache_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Cache TTL in seconds"
    )

    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100, ge=1, le=10000, description="Requests per minute"
    )
    rate_limit_burst: int = Field(
        default=20, ge=1, le=1000, description="Burst capacity"
    )

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1, le=100, description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=1.0, le=3600.0, description="Circuit breaker timeout"
    )

    # MLflow Configuration
    mlflow_tracking_uri: str | None = Field(
        default=None, description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="rag-service", description="MLflow experiment name"
    )
    mlflow_enable: bool = Field(default=True, description="Enable MLflow tracking")

    # Langfuse Configuration
    langfuse_secret_key: str | None = Field(
        default=None, description="Langfuse secret key"
    )
    langfuse_public_key: str | None = Field(
        default=None, description="Langfuse public key"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host"
    )
    langfuse_enable: bool = Field(default=True, description="Enable Langfuse tracing")
    tracing_enabled: bool = Field(
        default=False, description="Explicit tracing control for production"
    )

    # LangSmith Configuration (alternative)
    langsmith_api_key: str | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(
        default="rag-service", description="LangSmith project name"
    )

    # Cost Tracking
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    cost_per_1k_input_tokens: dict[str, float] = Field(
        default={
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03,
            "gpt-4o": 0.005,
            "llama3-8b-8192": 0.0005,
            "llama3-70b-8192": 0.0008,
        },
        description="Cost per 1K input tokens by model",
    )
    cost_per_1k_output_tokens: dict[str, float] = Field(
        default={
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.06,
            "gpt-4o": 0.015,
            "llama3-8b-8192": 0.0008,
            "llama3-70b-8192": 0.0012,
        },
        description="Cost per 1K output tokens by model",
    )

    # Evaluation Configuration
    eval_dir: str = Field(default="data/eval", description="Evaluation data directory")
    eval_output_dir: str = Field(
        default=".cache/eval", description="Evaluation output directory"
    )
    eval_enabled: bool = Field(default=True, description="Enable evaluation")
    eval_sample_size: int = Field(
        default=10, ge=1, le=1000, description="Evaluation sample size"
    )
    eval_golden_set_path: str = Field(
        default="eval/golden_set.jsonl", description="Golden set path"
    )

    # Alerting Configuration
    enable_alerts: bool = Field(default=True, description="Enable alerting")
    alert_webhook_url: str | None = Field(
        default=None, description="Webhook URL for alerts"
    )
    alert_hr_at_3_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="HR@3 alert threshold"
    )
    alert_latency_p95_threshold: float = Field(
        default=3000.0, ge=100.0, le=30000.0, description="Latency P95 threshold (ms)"
    )
    alert_cost_per_request_threshold: float = Field(
        default=0.01, ge=0.001, le=1.0, description="Cost per request threshold ($)"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format: json or text")
    enable_request_logging: bool = Field(
        default=True, description="Enable request/response logging"
    )

    # Application Settings
    auto_ingest_on_startup: bool = Field(
        default=True, description="Auto-ingest documents on startup"
    )
    docs_dir: str = Field(default="data/docs", description="Documents directory")
    sample_docs_dir: str = Field(
        default="data/sample", description="Sample documents directory"
    )

    # Health Check
    health_check_timeout: float = Field(
        default=5.0, ge=1.0, le=30.0, description="Health check timeout"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("environment")
    def validate_environment(cls, v):
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @validator("llm_provider")
    def validate_llm_provider(cls, v):
        allowed = {"openai", "groq", "ollama", "mock"}
        if v not in allowed:
            raise ValueError(f"LLM provider must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @validator("log_format")
    def validate_log_format(cls, v):
        allowed = {"json", "text"}
        if v not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get secret from Azure Key Vault or environment variable."""
        if self.use_key_vault and self.key_vault_url and AZURE_AVAILABLE:
            try:
                credential = DefaultAzureCredential()
                client = SecretClient(
                    vault_url=self.key_vault_url, credential=credential
                )
                secret = client.get_secret(key.replace("_", "-"))
                return secret.value
            except Exception as e:
                logger.warning(f"Failed to get secret {key} from Key Vault: {e}")

        # Fallback to environment variable
        return os.getenv(key.upper(), default)

    def get_openai_api_key(self) -> str | None:
        """Get OpenAI API key with Key Vault fallback."""
        return self.openai_api_key or self.get_secret("openai_api_key")

    def get_groq_api_key(self) -> str | None:
        """Get Groq API key with Key Vault fallback."""
        return self.groq_api_key or self.get_secret("groq_api_key")

    def get_langfuse_secret_key(self) -> str | None:
        """Get Langfuse secret key with Key Vault fallback."""
        return self.langfuse_secret_key or self.get_secret("langfuse_secret_key")

    def get_langfuse_public_key(self) -> str | None:
        """Get Langfuse public key with Key Vault fallback."""
        return self.langfuse_public_key or self.get_secret("langfuse_public_key")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def index_path(self) -> Path:
        """Get the index directory path."""
        return Path(self.index_dir)

    @property
    def eval_path(self) -> Path:
        """Get the evaluation directory path."""
        return Path(self.eval_dir)

    @property
    def eval_output_path(self) -> Path:
        """Get the evaluation output directory path."""
        return Path(self.eval_output_dir)


# Global settings instance
settings = Settings()
