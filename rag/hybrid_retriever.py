"""Hybrid Retrieval Module: Combines Dense Vector Search and BM25 Lexical Search via Reciprocal Rank Fusion (RRF)."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.bm25 import BM25Index, get_default_bm25_index
from rag.config import config
from rag.metadata_filter import FilterResult, apply_metadata_filters
from rag.query_parser import HardConstraints
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    allowed_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Compute Reciprocal Rank Fusion (RRF) scores over dense and BM25 ranked lists.

    Formula:
        RRF_score(d) = sum_{m in {dense, bm25}} (1 / (rrf_k + rank_m(d)))

    where rank_m(d) is 1-based (i.e. rank 1 gets 1 / (rrf_k + 1)).

    Args:
        dense_results: Ranked list of candidates from dense vector search.
        bm25_results: Ranked list of candidates from BM25 lexical search.
        rrf_k: Configurable smoothing parameter k (default 60).
        allowed_ids: Optional set of allowed dish IDs. If specified, any item
                     outside this set is strictly ignored.

    Returns:
        Deduplicated list of merged candidates sorted by RRF score descending,
        with deterministic tie-breaking on dish ID.
    """
    if rrf_k < 1:
        raise ValueError(f"RRF smoothing parameter k must be >= 1, got {rrf_k}")

    fused_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}

    # 1. Process dense ranked results (1-based rank)
    for idx, item in enumerate(dense_results):
        doc_id = item["id"]
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        rank = idx + 1
        dense_ranks[doc_id] = rank
        dense_scores[doc_id] = float(item.get("score", item.get("similarity", 0.0)))
        contribution = 1.0 / (rrf_k + rank)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + contribution
        if doc_id not in doc_registry:
            doc_registry[doc_id] = dict(item)

    # 2. Process BM25 ranked results (1-based rank)
    for idx, item in enumerate(bm25_results):
        doc_id = item["id"]
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        rank = idx + 1
        bm25_ranks[doc_id] = rank
        bm25_scores[doc_id] = float(item.get("score", item.get("bm25_score", 0.0)))
        contribution = 1.0 / (rrf_k + rank)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + contribution
        if doc_id not in doc_registry:
            doc_registry[doc_id] = dict(item)

    # 3. Assemble fused candidate objects with complete telemetry
    merged_candidates: List[Dict[str, Any]] = []
    for doc_id, rrf_score in fused_scores.items():
        base_item = doc_registry[doc_id]
        fused_item = dict(base_item)
        fused_item["score"] = round(rrf_score, 8)
        fused_item["rrf_score"] = round(rrf_score, 8)
        fused_item["dense_rank"] = dense_ranks.get(doc_id)
        fused_item["bm25_rank"] = bm25_ranks.get(doc_id)
        fused_item["dense_score"] = dense_scores.get(doc_id)
        fused_item["bm25_score"] = bm25_scores.get(doc_id)
        merged_candidates.append(fused_item)

    # 4. Deterministic sorting: highest RRF score first; tie-break on dish ID ascending
    merged_candidates.sort(key=lambda x: (-x["score"], x["id"]))

    return merged_candidates


class HybridRetriever:
    """Hybrid Retriever combining dense vector similarity with BM25 lexical ranking via RRF."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: Optional[int] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or get_default_bm25_index()
        self.rrf_k = rrf_k if rrf_k is not None else config.rrf_k

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return all indexed menu items for deterministic pre-filtering."""
        return self.bm25_index.get_all_documents()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform unconstrained hybrid retrieval using Dense + BM25 + RRF.

        Args:
            query: Natural language query string.
            top_k: Number of fused results to return.

        Returns:
            List of fused document dictionaries.
        """
        k = top_k or config.top_k
        if not query or not query.strip():
            logger.warning("Empty query passed to HybridRetriever.retrieve()")
            return []

        start_time = time.perf_counter()
        pool_size = max(k * 2, 15)

        # 1. Dense retrieval
        dense_raw = self.vector_store.search(query=query.strip(), top_k=pool_size)
        dense_candidates = [
            {
                "id": d["id"],
                "dish_name": d["dish_name"],
                "restaurant_name": d["restaurant_name"],
                "score": float(d["similarity"]),
                "distance": float(d["distance"]),
                "price_pkr": float(d["price_pkr"]),
                "location": d["location"],
                "halal": d["halal"],
                "spice_level": d["spice_level"],
                "allergens": d["allergens"],
                "allergen_status": d["allergen_status"],
                "metadata": d["metadata"],
                "document_text": d["document"],
            }
            for d in dense_raw
        ]

        # 2. BM25 retrieval
        bm25_candidates = self.bm25_index.search(query=query.strip(), top_k=pool_size)

        # 3. RRF Fusion
        fused = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            rrf_k=self.rrf_k,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Hybrid retrieval for '%s': Dense=%d, BM25=%d, Fused=%d in %.2f ms (k=%d)",
            query,
            len(dense_candidates),
            len(bm25_candidates),
            len(fused),
            elapsed_ms,
            k,
        )
        return fused[:k]

    def retrieve_filtered(
        self,
        query: str,
        constraints: HardConstraints,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], FilterResult]:
        """Perform constraint-aware hybrid retrieval with hard metadata pre-filtering upstream.

        Strict Architecture Flow:
        User Query
            |
            v
        Query Parser
            |
            v
        Hard Metadata Pre-filter (apply_metadata_filters)
            |
            v
        Dense Retrieval + BM25 (restricted to passed candidate IDs only)
            |
            v
        RRF Hybrid Ranking
            |
            v
        Context Builder -> Generator

        Hybrid retrieval NEVER reintroduces an item that failed a hard constraint.

        Args:
            query: User search query text.
            constraints: Parsed HardConstraints.
            top_k: Number of final candidates to retrieve.

        Returns:
            Tuple of (fused_passed_candidates, filter_result).
        """
        k = top_k or config.top_k
        start_time = time.perf_counter()

        # Step 1: Upstream Hard Metadata Pre-filtering over entire menu universe
        all_menu_docs = self.get_all_documents()
        filter_result = apply_metadata_filters(all_menu_docs, constraints)

        # Step 2: Early termination on NO_MATCHING_ITEMS
        if filter_result.status == "NO_MATCHING_ITEMS" or not filter_result.passed_candidates:
            logger.info("Hard metadata pre-filtering yielded 0 passed candidates. Returning NO_MATCHING_ITEMS.")
            return [], filter_result

        # Index passed candidates by ID for fast lookup and preservation of allergen warnings
        passed_map: Dict[str, Dict[str, Any]] = {
            cand["id"]: cand for cand in filter_result.passed_candidates
        }
        allowed_ids: Set[str] = set(passed_map.keys())

        # Step 3: Candidate retrieval pool size
        pool_size = min(max(k * 4, 25), len(allowed_ids))

        # Step 4: Run Dense vector search on query (filtered strictly to allowed IDs)
        dense_raw = self.vector_store.search(query=query.strip(), top_k=pool_size * 2)
        dense_candidates: List[Dict[str, Any]] = []
        for d in dense_raw:
            doc_id = d["id"]
            if doc_id in allowed_ids:
                item = {
                    "id": doc_id,
                    "dish_name": d["dish_name"],
                    "restaurant_name": d["restaurant_name"],
                    "score": float(d["similarity"]),
                    "distance": float(d["distance"]),
                    "price_pkr": float(d["price_pkr"]),
                    "location": d["location"],
                    "halal": d["halal"],
                    "spice_level": d["spice_level"],
                    "allergens": d["allergens"],
                    "allergen_status": d["allergen_status"],
                    "metadata": d["metadata"],
                    "document_text": d["document"],
                }
                # Preserve allergen warning from metadata pre-filter
                if "allergen_warning" in passed_map[doc_id]:
                    item["allergen_warning"] = passed_map[doc_id]["allergen_warning"]
                dense_candidates.append(item)

        # Step 5: Run BM25 lexical search on query (restricted strictly to allowed IDs)
        bm25_raw = self.bm25_index.search(query=query.strip(), top_k=pool_size * 2, allowed_ids=allowed_ids)
        bm25_candidates: List[Dict[str, Any]] = []
        for d in bm25_raw:
            doc_id = d["id"]
            if doc_id in allowed_ids:
                item = dict(d)
                # Preserve allergen warning from metadata pre-filter
                if "allergen_warning" in passed_map[doc_id]:
                    item["allergen_warning"] = passed_map[doc_id]["allergen_warning"]
                bm25_candidates.append(item)

        # Step 6: Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            rrf_k=self.rrf_k,
            allowed_ids=allowed_ids,
        )

        # In case a candidate passed pre-filter but wasn't ranked in top dense or BM25,
        # ensure candidates are available if pool is smaller than k
        if len(fused_candidates) < k:
            seen_ids = {c["id"] for c in fused_candidates}
            for doc_id, cand in passed_map.items():
                if doc_id not in seen_ids:
                    fallback_item = dict(cand)
                    fallback_item["score"] = 0.0
                    fallback_item["rrf_score"] = 0.0
                    fused_candidates.append(fallback_item)
                if len(fused_candidates) >= k:
                    break

        # Step 7: Take top-k candidates and ensure allergen warnings are preserved
        final_candidates = fused_candidates[:k]
        for cand in final_candidates:
            doc_id = cand["id"]
            if doc_id in passed_map and "allergen_warning" in passed_map[doc_id]:
                cand["allergen_warning"] = passed_map[doc_id]["allergen_warning"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Hybrid retrieve_filtered for '%s': Passed Pre-filter=%d, Dense=%d, BM25=%d, Fused=%d in %.2f ms",
            query,
            len(allowed_ids),
            len(dense_candidates),
            len(bm25_candidates),
            len(final_candidates),
            elapsed_ms,
        )

        return final_candidates, filter_result


# Singleton default hybrid retriever
_default_hybrid_retriever: Optional[HybridRetriever] = None


def get_default_hybrid_retriever() -> HybridRetriever:
    """Return the cached default HybridRetriever instance."""
    global _default_hybrid_retriever
    if _default_hybrid_retriever is None:
        _default_hybrid_retriever = HybridRetriever()
    return _default_hybrid_retriever
