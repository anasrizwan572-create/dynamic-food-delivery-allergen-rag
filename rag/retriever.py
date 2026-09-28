"""Basic semantic retriever module with metadata pre-filtering."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from rag.config import config
from rag.metadata_filter import FilterResult, apply_metadata_filters
from rag.query_parser import HardConstraints
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicRetriever:
    """Semantic retriever operating over the pre-indexed vector store with metadata filtering."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-K menu items semantically matching the query.

        Args:
            query: Natural language user query.
            top_k: Number of candidates to retrieve (defaults to config.top_k).

        Returns:
            List of structured document dictionaries with similarity scores and metadata.
        """
        k = top_k or config.top_k
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        start_time = time.perf_counter()
        results = self.vector_store.search(query=query.strip(), top_k=k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        retrieved_docs: List[Dict[str, Any]] = []
        for item in results:
            doc_struct = {
                "id": item["id"],
                "dish_name": item["dish_name"],
                "restaurant_name": item["restaurant_name"],
                "score": float(item["similarity"]),
                "distance": float(item["distance"]),
                "price_pkr": float(item["price_pkr"]),
                "location": item["location"],
                "halal": item["halal"],
                "spice_level": item["spice_level"],
                "allergens": item["allergens"],
                "allergen_status": item["allergen_status"],
                "metadata": item["metadata"],
                "document_text": item["document"],
            }
            retrieved_docs.append(doc_struct)

        logger.info(
            "Retrieved %d documents for query '%s' in %.2f ms (top_k=%d)",
            len(retrieved_docs),
            query,
            elapsed_ms,
            k,
        )
        return retrieved_docs

    def retrieve_filtered(
        self,
        query: str,
        constraints: HardConstraints,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], FilterResult]:
        """Retrieve candidates and apply deterministic hard metadata filters.

        Ensures mathematical price inequalities, Halal certification, location,
        and excluded allergens are strictly enforced before final ranking.

        Args:
            query: User's search query text.
            constraints: Parsed HardConstraints.
            top_k: Target number of final filtered candidates.

        Returns:
            Tuple of (top_k_passed_candidates, filter_result).
        """
        k = top_k or config.top_k
        total_docs_in_store = self.vector_store.count()
        # Retrieve an ample candidate pool to filter
        pool_size = min(max(k * 4, 25), max(1, total_docs_in_store))

        # 1. Broad vector search on query
        raw_candidates = self.retrieve(query=query, top_k=pool_size)

        # 2. Deterministic metadata pre-filtering (Mathematical price bounds, Halal, location, allergens)
        filter_result = apply_metadata_filters(raw_candidates, constraints)

        # 3. Take top-k among passed candidates
        final_candidates = filter_result.passed_candidates[:k]

        return final_candidates, filter_result


# Global convenience function
_default_retriever: Optional[BasicRetriever] = None


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant menu documents for a given query."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = BasicRetriever()
    return _default_retriever.retrieve(query=query, top_k=top_k)
