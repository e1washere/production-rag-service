"""Cost tracking instrumentation for LLM usage and request-level costs."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class CostProvider(Enum):
    """Supported cost providers."""

    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""

    provider: str
    model: str
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    def __post_init__(self):
        if self.total_cost == 0.0:
            self.total_cost = self.input_cost + self.output_cost


@dataclass
class RequestCostMetrics:
    """Request-level cost and performance metrics."""

    request_id: str
    timestamp: float = field(default_factory=time.time)

    # LLM Costs
    llm_costs: list[CostBreakdown] = field(default_factory=list)
    total_llm_cost: float = 0.0

    # Infrastructure Costs (estimated)
    compute_cost: float = 0.0  # Based on request processing time
    storage_cost: float = 0.0  # Based on data retrieved/stored
    network_cost: float = 0.0  # Based on data transfer

    # Performance Metrics
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Usage Metrics
    documents_retrieved: int = 0
    characters_generated: int = 0

    @property
    def total_cost(self) -> float:
        """Total request cost."""
        return (
            self.total_llm_cost
            + self.compute_cost
            + self.storage_cost
            + self.network_cost
        )


class CostCalculator:
    """Calculate costs for different providers and models."""

    def __init__(self):
        self.input_costs = settings.cost_per_1k_input_tokens
        self.output_costs = settings.cost_per_1k_output_tokens

        # Infrastructure cost rates (USD)
        self.compute_cost_per_second = 0.0001  # ~$0.36/hour for basic compute
        self.storage_cost_per_mb = 0.000001  # ~$0.001/GB
        self.network_cost_per_mb = 0.00001  # ~$0.01/GB

    def calculate_llm_cost(
        self, provider: str, model: str, token_usage: TokenUsage
    ) -> CostBreakdown:
        """Calculate LLM cost breakdown."""

        # Get cost rates for model
        input_rate = self.input_costs.get(model, 0.0)
        output_rate = self.output_costs.get(model, 0.0)

        # Calculate costs
        input_cost = (token_usage.input_tokens / 1000) * input_rate
        output_cost = (token_usage.output_tokens / 1000) * output_rate

        return CostBreakdown(
            provider=provider,
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            cost_per_1k_input=input_rate,
            cost_per_1k_output=output_rate,
        )

    def calculate_compute_cost(self, processing_time_seconds: float) -> float:
        """Calculate compute cost based on processing time."""
        return processing_time_seconds * self.compute_cost_per_second

    def calculate_storage_cost(self, data_size_bytes: int) -> float:
        """Calculate storage cost based on data size."""
        data_size_mb = data_size_bytes / (1024 * 1024)
        return data_size_mb * self.storage_cost_per_mb

    def calculate_network_cost(self, transfer_size_bytes: int) -> float:
        """Calculate network cost based on data transfer."""
        transfer_size_mb = transfer_size_bytes / (1024 * 1024)
        return transfer_size_mb * self.network_cost_per_mb


class CostTracker:
    """Track and aggregate costs across requests."""

    def __init__(self):
        self.calculator = CostCalculator()
        self.request_metrics: dict[str, RequestCostMetrics] = {}
        self._lock = asyncio.Lock()

        # Aggregated metrics
        self.total_requests = 0
        self.total_cost = 0.0
        self.cost_by_provider: dict[str, float] = {}
        self.cost_by_model: dict[str, float] = {}

    def create_request_metrics(self, request_id: str) -> RequestCostMetrics:
        """Create new request metrics."""
        metrics = RequestCostMetrics(request_id=request_id)
        self.request_metrics[request_id] = metrics
        return metrics

    async def track_llm_usage(
        self, request_id: str, provider: str, model: str, token_usage: TokenUsage
    ) -> CostBreakdown:
        """Track LLM usage and cost."""
        cost_breakdown = self.calculator.calculate_llm_cost(
            provider, model, token_usage
        )

        async with self._lock:
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id]
                metrics.llm_costs.append(cost_breakdown)
                metrics.total_llm_cost += cost_breakdown.total_cost

        logger.debug(f"LLM cost tracked: ${cost_breakdown.total_cost:.6f} for {model}")
        return cost_breakdown

    async def track_infrastructure_cost(
        self,
        request_id: str,
        processing_time_seconds: float,
        data_size_bytes: int = 0,
        transfer_size_bytes: int = 0,
    ) -> None:
        """Track infrastructure costs."""
        compute_cost = self.calculator.calculate_compute_cost(processing_time_seconds)
        storage_cost = self.calculator.calculate_storage_cost(data_size_bytes)
        network_cost = self.calculator.calculate_network_cost(transfer_size_bytes)

        async with self._lock:
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id]
                metrics.compute_cost += compute_cost
                metrics.storage_cost += storage_cost
                metrics.network_cost += network_cost

    async def finalize_request_metrics(
        self, request_id: str
    ) -> RequestCostMetrics | None:
        """Finalize and return request metrics."""
        async with self._lock:
            if request_id not in self.request_metrics:
                return None

            metrics = self.request_metrics[request_id]

            # Update aggregated metrics
            self.total_requests += 1
            self.total_cost += metrics.total_cost

            # Track by provider/model
            for cost_breakdown in metrics.llm_costs:
                provider = cost_breakdown.provider
                model = cost_breakdown.model

                self.cost_by_provider[provider] = (
                    self.cost_by_provider.get(provider, 0.0) + cost_breakdown.total_cost
                )
                self.cost_by_model[model] = (
                    self.cost_by_model.get(model, 0.0) + cost_breakdown.total_cost
                )

            # Remove from active tracking to prevent memory leaks
            completed_metrics = self.request_metrics.pop(request_id)

            return completed_metrics

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Get aggregated cost statistics."""
        avg_cost_per_request = self.total_cost / max(self.total_requests, 1)

        return {
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_per_request_usd": round(avg_cost_per_request, 6),
            "cost_by_provider": {
                k: round(v, 6) for k, v in self.cost_by_provider.items()
            },
            "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
            "active_requests": len(self.request_metrics),
        }

    def reset_stats(self) -> None:
        """Reset aggregated statistics."""
        self.total_requests = 0
        self.total_cost = 0.0
        self.cost_by_provider.clear()
        self.cost_by_model.clear()


# Global cost tracker instance
cost_tracker = CostTracker()


@asynccontextmanager
async def track_request_cost(request_id: str):
    """Context manager to track request-level costs."""
    start_time = time.time()
    metrics = cost_tracker.create_request_metrics(request_id)

    try:
        yield metrics
    finally:
        # Track processing time
        processing_time = time.time() - start_time
        await cost_tracker.track_infrastructure_cost(request_id, processing_time)

        # Finalize metrics
        final_metrics = await cost_tracker.finalize_request_metrics(request_id)

        if final_metrics and settings.enable_cost_tracking:
            logger.info(
                f"Request {request_id} cost: ${final_metrics.total_cost:.6f} "
                f"(LLM: ${final_metrics.total_llm_cost:.6f}, "
                f"Infra: ${final_metrics.total_cost - final_metrics.total_llm_cost:.6f})"
            )


async def track_llm_cost(
    request_id: str, provider: str, model: str, input_tokens: int, output_tokens: int
) -> CostBreakdown:
    """Convenience function to track LLM costs."""
    token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return await cost_tracker.track_llm_usage(request_id, provider, model, token_usage)


def get_cost_stats() -> dict[str, Any]:
    """Get current cost statistics."""
    return cost_tracker.get_aggregate_stats()


def estimate_query_cost(query: str, model: str = "gpt-3.5-turbo") -> float:
    """Estimate cost for a query (rough approximation)."""
    # Rough token estimation (4 chars = 1 token)
    estimated_input_tokens = len(query) // 4
    estimated_output_tokens = 200  # Assume average response length

    input_rate = settings.cost_per_1k_input_tokens.get(model, 0.0015)
    output_rate = settings.cost_per_1k_output_tokens.get(model, 0.002)

    input_cost = (estimated_input_tokens / 1000) * input_rate
    output_cost = (estimated_output_tokens / 1000) * output_rate

    return input_cost + output_cost
