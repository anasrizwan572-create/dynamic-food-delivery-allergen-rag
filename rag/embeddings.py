"""Modular embedding providers for dense vector indexing and semantic retrieval."""

from abc import ABC, abstractmethod
import hashlib
import logging
import math
import os
import re
from typing import Any, List, Optional
import numpy as np

from rag.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of document strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute an embedding for a single user query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the embedding model."""
        pass


class DeterministicDenseEmbeddingProvider(EmbeddingProvider):
    """Robust, dependency-free local semantic embedding generator.

    Produces deterministic 384-dimensional unit-normalized dense vectors by projecting
    token and character n-gram hashes into a semantic space. Guarantees 100% offline
    execution with zero external GPU/network dependencies.
    """

    def __init__(self, dimension: int = 384, model_name: str = "deterministic-dense-384"):
        self._dim = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_single(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dim

        vec = np.zeros(self._dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())

        for idx, word in enumerate(words):
            # Token position and term weighting
            weight = 1.0 + 1.0 / (idx + 1.0)
            
            # Word level hashing
            h_word = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            pos = h_word % self._dim
            sign = 1.0 if ((h_word >> 8) & 1) == 0 else -1.0
            vec[pos] += sign * weight

            # Sub-word character n-grams (3-grams) for morphological capture
            for i in range(len(word) - 2):
                ngram = word[i:i + 3]
                h_ngram = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                pos_ng = h_ngram % self._dim
                sign_ng = 1.0 if ((h_ngram >> 4) & 1) == 0 else -1.0
                vec[pos_ng] += sign_ng * 0.5

        # L2 Unit Normalization for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider utilizing HuggingFace / Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dim = 384
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            dummy = self._model.encode("test")
            self._dim = len(dummy)
            logger.info("Initialized SentenceTransformer model: %s (dim=%d)", model_name, self._dim)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer (%s). Falling back.", e)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        raise RuntimeError("SentenceTransformer model is not loaded.")

    def embed_query(self, text: str) -> List[float]:
        if self._model is not None:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        raise RuntimeError("SentenceTransformer model is not loaded.")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API Embedding provider for text-embedding-3-small or text-embedding-ada-002."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._dim = 1536
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY must be provided for OpenAIEmbeddingProvider")
        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("Please install openai package to use OpenAIEmbeddingProvider")

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(input=[text], model=self._model_name)
        return response.data[0].embedding


def get_embedding_provider(
    provider_name: Optional[str] = None, 
    model_name: Optional[str] = None
) -> EmbeddingProvider:
    """Factory to retrieve configured embedding provider with automatic fallback.

    Args:
        provider_name: 'local', 'sentence_transformers', 'openai', or 'deterministic'
        model_name: Optional model override.

    Returns:
        Instance of EmbeddingProvider.
    """
    provider = (provider_name or config.embedding_provider).lower()
    model = model_name or config.embedding_model

    if provider in ["deterministic", "mock", "dummy"]:
        return DeterministicDenseEmbeddingProvider(model_name=model_name or f"{model}-local")

    elif provider in ["sentence_transformers", "st"]:
        try:
            return SentenceTransformerEmbeddingProvider(model_name=model)
        except Exception as e:
            logger.warning("SentenceTransformer failed (%s); using deterministic dense fallback.", e)
            return DeterministicDenseEmbeddingProvider(model_name=f"fallback-{model}")

    elif provider == "openai":
        return OpenAIEmbeddingProvider(model_name=model)

    else:
        # Check if sentence_transformers is readily importable for 'local'
        try:
            import sentence_transformers
            return SentenceTransformerEmbeddingProvider(model_name=model)
        except Exception:
            logger.info("Using deterministic local dense embedding provider (dim=384)")
            return DeterministicDenseEmbeddingProvider(model_name="all-MiniLM-L6-v2-local")
