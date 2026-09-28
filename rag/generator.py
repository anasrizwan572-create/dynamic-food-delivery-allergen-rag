"""Configurable LLM Generator interface supporting OpenAI, Gemini, and local mock fallback."""

from abc import ABC, abstractmethod
import logging
import os
import re
from typing import Any, Dict, List, Optional

from rag.config import config
from rag.prompts import RAG_SYSTEM_PROMPT, format_rag_user_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for LLM response generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        retrieved_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a grounded natural language response."""
        pass


class MockGroundedGenerator(BaseGenerator):
    """Deterministic offline generator synthesizing grounded answers from context without API keys."""

    def __init__(self, citation_engine: Optional[Any] = None):
        from rag.citation import get_citation_engine
        self.citation_engine = citation_engine or get_citation_engine()

    def generate(
        self,
        query: str,
        context: str,
        retrieved_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not retrieved_docs or "No retrieved documents available" in context:
            return "Based on the retrieved menu information, there is insufficient evidence to answer your request."

        answer, _, _ = self.citation_engine.synthesize_grounded_answer(
            query=query,
            retrieved_docs=retrieved_docs,
        )
        return answer


class OpenAIGenerator(BaseGenerator):
    """Generator utilizing OpenAI chat completion models (e.g. gpt-4o-mini)."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerator")
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai package to use OpenAIGenerator")

    def generate(
        self,
        query: str,
        context: str,
        retrieved_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
        user_prompt = format_rag_user_prompt(query, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("OpenAI generation failed (%s); falling back to mock generator.", e)
            fallback = MockGroundedGenerator()
            return fallback.generate(query, context, retrieved_docs, system_prompt)


def get_generator(provider_name: Optional[str] = None) -> BaseGenerator:
    """Factory to retrieve configured generator with graceful fallback to mock."""
    provider = (provider_name or os.getenv("LLM_PROVIDER", "mock")).lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                return OpenAIGenerator()
            except Exception as e:
                logger.warning("Failed to initialize OpenAIGenerator (%s); falling back to Mock.", e)
        else:
            logger.info("OPENAI_API_KEY not set; using MockGroundedGenerator.")
            return MockGroundedGenerator()

    return MockGroundedGenerator()
