"""Structured logging setup with correlation IDs and PII filtering."""

import json
import logging
import logging.config
import re
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime

from app.config import settings

# Context variable for correlation ID
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "unknown"
        return True


class PIIFilter(logging.Filter):
    """Filter out potential PII from log messages."""

    def __init__(self):
        super().__init__()
        # Patterns for common PII
        self.pii_patterns = [
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                "[EMAIL]",
            ),
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
            (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
            (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
            (re.compile(r"\b(?:sk-|pk-)[a-zA-Z0-9]{20,}\b"), "[API_KEY]"),
            (re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b"), "[TOKEN]"),
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg") and isinstance(record.msg, str):
            message = record.msg
            for pattern, replacement in self.pii_patterns:
                message = pattern.sub(replacement, message)
            record.msg = message
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "unknown"),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "correlation_id",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with correlation ID."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging():
    """Configure logging based on settings."""

    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Choose formatter
    if settings.log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationFilter())
    console_handler.addFilter(PIIFilter())

    root_logger.addHandler(console_handler)

    # Configure specific loggers
    configure_library_loggers()

    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={settings.log_level}, format={settings.log_format}"
    )


def configure_library_loggers():
    """Configure third-party library loggers."""

    # Reduce noise from libraries
    library_configs = {
        "urllib3.connectionpool": logging.WARNING,
        "requests.packages.urllib3": logging.WARNING,
        "azure.core.pipeline.policies.http_logging_policy": logging.WARNING,
        "azure.identity": logging.WARNING,
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "langfuse": logging.INFO,
        "mlflow": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "faiss": logging.WARNING,
    }

    for logger_name, level in library_configs.items():
        logging.getLogger(logger_name).setLevel(level)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    current_id = correlation_id.get()
    if current_id is None:
        current_id = str(uuid.uuid4())[:8]  # Short correlation ID
        correlation_id.set(current_id)
    return current_id


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(cid)


def new_correlation_id() -> str:
    """Generate and set new correlation ID."""
    new_id = str(uuid.uuid4())[:8]
    correlation_id.set(new_id)
    return new_id


class StructuredLogger:
    """Structured logger with extra context."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with structured context."""
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.log(level, message, extra=extra, exc_info=kwargs.get("exc_info"))

    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def log_request_start(self, method: str, path: str, **kwargs):
        """Log request start."""
        self.info(
            f"Request started: {method} {path}",
            event_type="request_start",
            http_method=method,
            http_path=path,
            **kwargs,
        )

    def log_request_end(
        self, method: str, path: str, status_code: int, duration_ms: float, **kwargs
    ):
        """Log request completion."""
        self.info(
            f"Request completed: {method} {path} -> {status_code} ({duration_ms:.1f}ms)",
            event_type="request_end",
            http_method=method,
            http_path=path,
            http_status=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost: float,
        **kwargs,
    ):
        """Log LLM API call."""
        self.info(
            f"LLM call: {provider}/{model} - {input_tokens}→{output_tokens} tokens "
            f"({duration_ms:.1f}ms, ${cost:.6f})",
            event_type="llm_call",
            llm_provider=provider,
            llm_model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost,
            **kwargs,
        )

    def log_cache_operation(self, operation: str, key: str, hit: bool = None, **kwargs):
        """Log cache operation."""
        if hit is not None:
            result = "hit" if hit else "miss"
            message = f"Cache {operation}: {key} -> {result}"
        else:
            message = f"Cache {operation}: {key}"

        self.debug(
            message,
            event_type="cache_operation",
            cache_operation=operation,
            cache_key=key,
            cache_hit=hit,
            **kwargs,
        )

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "", **kwargs
    ):
        """Log performance metric."""
        self.info(
            f"Performance metric: {metric_name} = {value}{unit}",
            event_type="performance_metric",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs,
        )


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)


# Initialize logging on import
if not logging.getLogger().handlers:
    setup_logging()
