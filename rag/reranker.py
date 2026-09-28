"""Cross-Encoder Reranker Module for deep query-document relevance scoring."""

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from rag.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseReranker:
    """Abstract interface for candidate document rerankers."""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Score candidates against the query and return sorted list."""
        raise NotImplementedError


class DeterministicFallbackReranker(BaseReranker):
    """Deterministic local cross-attention surrogate scorer.

    Computes query-document relevance using token alignment, phrase matching,
    position weighting, and field-importance heuristics without requiring
    external network downloads or heavy GPU dependencies.
    """

    def __init__(self, model_name: str = "deterministic-cross-scorer"):
        self.model_name = model_name
        logger.info("Initialized DeterministicFallbackReranker (%s).", self.model_name)

    def _score_pair(self, query: str, document_text: str, dish_name: str) -> float:
        """Compute cross-scoring interaction between query and document text."""
        q_clean = query.strip().lower()
        d_clean = document_text.strip().lower()
        name_clean = dish_name.strip().lower()

        if not q_clean or not d_clean:
            return 0.0

        q_tokens = [t for t in re.findall(r"\w+", q_clean) if t]
        d_tokens = [t for t in re.findall(r"\w+", d_clean) if t]

        if not q_tokens or not d_tokens:
            return 0.0

        # 1. Exact phrase match in dish name or document
        phrase_bonus = 0.0
        if q_clean in name_clean:
            phrase_bonus += 3.0
        elif name_clean in q_clean:
            phrase_bonus += 2.5
        elif q_clean in d_clean:
            phrase_bonus += 1.5

        # 2. Token overlap and positional alignment
        d_set = set(d_tokens)
        name_set = set(re.findall(r"\w+", name_clean))
        matched_tokens = 0
        name_matches = 0

        for t in q_tokens:
            if t in name_set:
                name_matches += 1
            if t in d_set:
                matched_tokens += 1

        overlap_ratio = matched_tokens / len(q_tokens)
        name_ratio = name_matches / len(q_tokens)

        # 3. Normalized cross-score
        raw_score = (overlap_ratio * 4.0) + (name_ratio * 3.0) + phrase_bonus
        return round(float(raw_score), 6)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using deterministic cross-scoring."""
        if not candidates:
            return []

        start_time = time.perf_counter()
        k = top_k or len(candidates)

        scored_candidates: List[Dict[str, Any]] = []
        for cand in candidates:
            doc_text = cand.get("document_text") or cand.get("dish_name", "")
            dish_name = cand.get("dish_name", "")
            score = self._score_pair(query, doc_text, dish_name)

            item = dict(cand)
            item["reranker_score"] = score
            item["score"] = score
            scored_candidates.append(item)

        # Deterministic sort: highest reranker score first; tie-break on dish ID ascending
        scored_candidates.sort(key=lambda x: (-x["reranker_score"], x["id"]))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "DeterministicFallbackReranker reranked %d candidates in %.2f ms",
            len(candidates),
            elapsed_ms,
        )
        return scored_candidates[:k]


class SentenceTransformerReranker(BaseReranker):
    """Local CrossEncoder reranker using Sentence-Transformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.cross_encoder_model
        self.model = None
        self._fallback: Optional[DeterministicFallbackReranker] = None

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load SentenceTransformers CrossEncoder model, falling back gracefully."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model: '%s'...", self.model_name)
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model '%s' loaded successfully.", self.model_name)
        except Exception as e:
            logger.warning(
                "Could not load CrossEncoder model '%s' (%s). "
                "Falling back to DeterministicFallbackReranker.",
                self.model_name,
                e,
            )
            self.model = None
            self._fallback = DeterministicFallbackReranker(model_name=f"fallback-{self.model_name}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using CrossEncoder predictions or fallback."""
        if not candidates:
            return []

        if self.model is None:
            if self._fallback is None:
                self._fallback = DeterministicFallbackReranker()
            return self._fallback.rerank(query=query, candidates=candidates, top_k=top_k)

        start_time = time.perf_counter()
        k = top_k or len(candidates)

        pairs = [
            (
                query.strip(),
                c.get("document_text") or f"{c.get('restaurant_name', '')} {c.get('dish_name', '')}",
            )
            for c in candidates
        ]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning("CrossEncoder prediction error: %s. Using deterministic fallback.", e)
            if self._fallback is None:
                self._fallback = DeterministicFallbackReranker()
            return self._fallback.rerank(query=query, candidates=candidates, top_k=top_k)

        scored_candidates: List[Dict[str, Any]] = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            val = float(score)
            item["reranker_score"] = round(val, 6)
            item["score"] = round(val, 6)
            scored_candidates.append(item)

        # Deterministic sort: highest reranker score first; tie-break on dish ID ascending
        scored_candidates.sort(key=lambda x: (-x["reranker_score"], x["id"]))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "SentenceTransformerReranker (%s) reranked %d candidates in %.2f ms",
            self.model_name,
            len(candidates),
            elapsed_ms,
        )
        return scored_candidates[:k]


# Singleton default reranker
_default_reranker: Optional[BaseReranker] = None


def get_reranker(
    reranker_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseReranker:
    """Factory returning configured reranker instance."""
    global _default_reranker
    rtype = (reranker_type or config.reranker_type).lower()

    if rtype in ["dummy", "mock", "deterministic"]:
        return DeterministicFallbackReranker(model_name=model_name or "deterministic-cross-scorer")

    if _default_reranker is None:
        _default_reranker = SentenceTransformerReranker(model_name=model_name or config.cross_encoder_model)
    return _default_reranker
