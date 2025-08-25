"""LLM provider implementations for RAG service."""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate answer based on query and contexts."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name for logging."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(
        self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str | None = None
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate answer using OpenAI."""
        try:
            client = self._get_client()

            # Prepare context
            context_text = "\n\n".join(contexts[:5])  # Use top 5 contexts

            # Create prompt
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def get_provider_name(self) -> str:
        return f"OpenAI ({self.model})"


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load Groq client."""
        if self._client is None:
            try:
                from groq import Groq

                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Groq package not installed. Run: pip install groq")
        return self._client

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate answer using Groq."""
        try:
            client = self._get_client()

            # Prepare context
            context_text = "\n\n".join(contexts[:5])

            # Create prompt
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def get_provider_name(self) -> str:
        return f"Groq ({self.model})"


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider implementation."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                import requests

                self._client = requests.Session()
                self._client.base_url = self.base_url
            except ImportError:
                raise ImportError(
                    "Requests package not installed. Run: pip install requests"
                )
        return self._client

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate answer using Ollama."""
        try:
            client = self._get_client()

            # Prepare context
            context_text = "\n\n".join(contexts[:5])

            # Create prompt
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""

            response = client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                },
            )

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def get_provider_name(self) -> str:
        return f"Ollama ({self.model})"


class MockProvider(LLMProvider):
    """Mock LLM provider for testing and development."""

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate mock answer."""
        context_text = "\n\n".join(contexts[:3])
        return f"Based on the provided context, here's what I found: {query}. Context: {context_text[:200]}..."

    def get_provider_name(self) -> str:
        return "Mock Provider"


def create_llm_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
    """Factory function to create LLM provider."""

    if provider_type.lower() == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        return OpenAIProvider(
            api_key=api_key,
            model=kwargs.get("model", "gpt-3.5-turbo"),
            base_url=kwargs.get("base_url"),
        )

    elif provider_type.lower() == "groq":
        api_key = kwargs.get("api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key required")
        return GroqProvider(
            api_key=api_key, model=kwargs.get("model", "llama3-8b-8192")
        )

    elif provider_type.lower() == "ollama":
        return OllamaProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama2"),
        )

    elif provider_type.lower() == "mock":
        return MockProvider()

    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")
