"""FastAPI middleware for request/response logging and monitoring."""

import json
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.config import settings
from app.cost_tracker import track_request_cost
from app.logging import get_correlation_id, get_logger, set_correlation_id

logger = get_logger(__name__)


async def correlation_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add correlation ID to request context."""
    # Extract correlation ID from header or generate new one
    correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())[:8]
    set_correlation_id(correlation_id)

    # Process request
    response = await call_next(request)

    # Add correlation ID to response headers
    response.headers["x-correlation-id"] = correlation_id

    return response


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log request and response details."""
    if not settings.enable_request_logging:
        return await call_next(request)

    start_time = time.time()
    correlation_id = get_correlation_id()

    # Extract request details
    method = request.method
    path = request.url.path
    query_params = dict(request.query_params)
    headers = dict(request.headers)

    # Remove sensitive headers
    sensitive_headers = {"authorization", "x-api-key", "cookie"}
    filtered_headers = {
        k: v if k.lower() not in sensitive_headers else "[REDACTED]"
        for k, v in headers.items()
    }

    # Log request start
    logger.log_request_start(
        method=method,
        path=path,
        query_params=query_params,
        headers=filtered_headers,
        client_ip=request.client.host if request.client else None,
        user_agent=headers.get("user-agent"),
    )

    # Process request with cost tracking
    async with track_request_cost(correlation_id) as cost_metrics:
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Update cost metrics with timing
            cost_metrics.total_latency_ms = duration_ms

            # Log successful response
            logger.log_request_end(
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                response_size=len(response.body) if hasattr(response, "body") else 0,
            )

            return response

        except Exception as e:
            # Calculate duration for error case
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                f"Request failed: {method} {path}",
                event_type="request_error",
                http_method=method,
                http_path=path,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "correlation_id": correlation_id,
                    "timestamp": time.time(),
                },
            )


async def performance_monitoring_middleware(
    request: Request, call_next: Callable
) -> Response:
    """Monitor performance metrics."""
    start_time = time.time()
    start_cpu = time.process_time()

    # Process request
    response = await call_next(request)

    # Calculate metrics
    wall_time = (time.time() - start_time) * 1000  # ms
    cpu_time = (time.process_time() - start_cpu) * 1000  # ms

    # Log performance metrics
    if wall_time > 1000:  # Log slow requests (>1s)
        logger.log_performance_metric(
            "slow_request",
            wall_time,
            "ms",
            path=request.url.path,
            method=request.method,
            cpu_time_ms=cpu_time,
        )

    # Add performance headers
    response.headers["x-response-time"] = f"{wall_time:.1f}ms"
    response.headers["x-cpu-time"] = f"{cpu_time:.1f}ms"

    return response


async def security_headers_middleware(
    request: Request, call_next: Callable
) -> Response:
    """Add security headers to response."""
    response = await call_next(request)

    # Add security headers
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
    }

    for header, value in security_headers.items():
        response.headers[header] = value

    return response


def sanitize_request_body(body: bytes, content_type: str) -> dict:
    """Sanitize request body for logging."""
    if not body:
        return {}

    try:
        if "application/json" in content_type:
            data = json.loads(body)
            # Remove sensitive fields
            sensitive_fields = {"password", "token", "key", "secret", "api_key"}

            def sanitize_dict(obj):
                if isinstance(obj, dict):
                    return {
                        k: (
                            "[REDACTED]"
                            if any(field in k.lower() for field in sensitive_fields)
                            else sanitize_dict(v)
                        )
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [sanitize_dict(item) for item in obj]
                else:
                    return obj

            return sanitize_dict(data)
        else:
            return {"content_type": content_type, "size": len(body)}
    except Exception:
        return {"content_type": content_type, "size": len(body), "parse_error": True}
