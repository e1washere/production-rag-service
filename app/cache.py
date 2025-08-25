"""Redis caching layer for retriever results with TTL and corpus versioning."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    corpus_version: str
    timestamp: float
    hit_count: int = 0


class CacheManager:
    """Redis cache manager with corpus versioning."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self.ttl = settings.cache_ttl
        self.enabled = settings.enable_cache and REDIS_AVAILABLE
        self._client: Redis | None = None
        self._corpus_version = "v1"  # Will be updated based on corpus changes

        if not REDIS_AVAILABLE and settings.enable_cache:
            logger.warning("Redis not available, caching disabled")

    async def _get_client(self) -> Redis | None:
        """Get Redis client with lazy initialization."""
        if not self.enabled:
            return None

        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                # Test connection
                await self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.enabled = False
                return None

        return self._client

    def _generate_key(self, query: str, top_k: int, filters: dict | None = None) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
            "corpus_version": self._corpus_version,
            "embedding_model": settings.embedding_model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"retrieval:{hashlib.md5(key_str.encode()).hexdigest()}"

    async def get_retrieval_results(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[dict] | None:
        """Get cached retrieval results."""
        client = await self._get_client()
        if not client:
            return None

        try:
            key = self._generate_key(query, top_k, filters)
            cached_data = await client.get(key)

            if cached_data:
                # Update hit count
                await client.hincrby(f"{key}:meta", "hit_count", 1)

                data = json.loads(cached_data)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return data["results"]

            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set_retrieval_results(
        self, query: str, top_k: int, results: list[dict], filters: dict | None = None
    ) -> None:
        """Cache retrieval results."""
        client = await self._get_client()
        if not client:
            return

        try:
            key = self._generate_key(query, top_k, filters)

            cache_data = {
                "results": results,
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "corpus_version": self._corpus_version,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Set main cache entry
            await client.setex(key, self.ttl, json.dumps(cache_data, default=str))

            # Set metadata
            meta_data = {
                "hit_count": 0,
                "created_at": cache_data["timestamp"],
                "corpus_version": self._corpus_version,
            }

            await client.hmset(f"{key}:meta", meta_data)
            await client.expire(f"{key}:meta", self.ttl)

            logger.debug(f"Cached results for query: {query[:50]}...")

        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    async def invalidate_corpus_cache(self, new_version: str | None = None) -> None:
        """Invalidate all cache entries for corpus update."""
        client = await self._get_client()
        if not client:
            return

        try:
            # Update corpus version
            if new_version:
                self._corpus_version = new_version
            else:
                import time

                self._corpus_version = f"v{int(time.time())}"

            # Find and delete all retrieval cache keys
            pattern = "retrieval:*"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    await client.delete(*batch)

                    # Also delete metadata keys
                    meta_keys = [f"{key}:meta" for key in batch]
                    await client.delete(*meta_keys)

            logger.info(
                f"Invalidated {len(keys)} cache entries for corpus version {self._corpus_version}"
            )

        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        client = await self._get_client()
        if not client:
            return {"enabled": False, "available": False}

        try:
            info = await client.info("memory")
            keyspace = await client.info("keyspace")

            # Count retrieval cache keys
            retrieval_keys = 0
            async for _ in client.scan_iter(match="retrieval:*"):
                retrieval_keys += 1

            return {
                "enabled": True,
                "available": True,
                "corpus_version": self._corpus_version,
                "retrieval_keys": retrieval_keys,
                "memory_used": info.get("used_memory_human", "N/A"),
                "memory_peak": info.get("used_memory_peak_human", "N/A"),
                "total_keys": sum(db.get("keys", 0) for db in keyspace.values()),
                "ttl_seconds": self.ttl,
            }

        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"enabled": True, "available": False, "error": str(e)}

    async def clear_all_cache(self) -> None:
        """Clear all cache entries (use with caution)."""
        client = await self._get_client()
        if not client:
            return

        try:
            await client.flushdb()
            logger.warning("All cache entries cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Global cache manager instance
cache_manager = CacheManager()


async def get_cached_retrieval_results(
    query: str, top_k: int, filters: dict | None = None
) -> list[dict] | None:
    """Convenience function to get cached results."""
    return await cache_manager.get_retrieval_results(query, top_k, filters)


async def cache_retrieval_results(
    query: str, top_k: int, results: list[dict], filters: dict | None = None
) -> None:
    """Convenience function to cache results."""
    await cache_manager.set_retrieval_results(query, top_k, results, filters)


def cache_key_for_query(query: str, top_k: int = 5) -> str:
    """Generate cache key for debugging."""
    return cache_manager._generate_key(query, top_k)
