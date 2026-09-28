"""Evaluation metrics computation for Food Delivery RAG system.

Implements mathematical evaluation metrics for:
- Retrieval Ranking Quality: MRR, NDCG@k, Recall@k
- Constraint Compliance: Price Precision, Dietary Recall, Allergen Safety Recall
- Grounded Generation: Citation Accuracy, Hallucination Rate
- Safety Invariants: Invariant Warning & Status Compliance
"""

import math
from typing import List, Dict, Any, Optional, Sequence


def compute_reciprocal_rank(retrieved_ids: Sequence[str], target_ids: Sequence[str]) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of the first relevant document in retrieved_ids (1-indexed).
    Returns 0.0 if no target_id is retrieved or if target_ids is empty.
    """
    if not retrieved_ids or not target_ids:
        return 0.0
    target_set = set(target_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in target_set:
            return 1.0 / rank
    return 0.0


def compute_mrr(all_retrieved_ids: Sequence[Sequence[str]], all_target_ids: Sequence[Sequence[str]]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries."""
    if not all_retrieved_ids or not all_target_ids:
        return 0.0
    reciprocal_ranks = [
        compute_reciprocal_rank(ret_ids, tgt_ids)
        for ret_ids, tgt_ids in zip(all_retrieved_ids, all_target_ids)
        if tgt_ids  # Only evaluate queries that have positive target dishes
    ]
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_recall_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int = 5) -> float:
    """Compute Recall@k for a single query.

    Recall@k = |retrieved_ids[:k] ∩ target_ids| / |target_ids|
    Returns 1.0 if target_ids is empty and retrieved_ids[:k] is empty (correct rejection).
    Returns 0.0 if target_ids is empty but items were retrieved.
    """
    if not target_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    top_k_ids = set(retrieved_ids[:k])
    target_set = set(target_ids)
    hits = len(top_k_ids.intersection(target_set))
    return hits / len(target_set)


def compute_ndcg_at_k(retrieved_ids: Sequence[str], target_ids: Sequence[str], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at k (NDCG@k) using binary relevance.

    DCG@k = sum(rel_i / log2(i + 1)) for i in 1..min(k, len(retrieved_ids))
    IDCG@k = sum(1 / log2(i + 1)) for i in 1..min(k, len(target_ids))
    """
    if not target_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    target_set = set(target_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in target_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    ideal_hits = min(k, len(target_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_forbidden_leakage(
    retrieved_ids: Sequence[str],
    forbidden_ids: Sequence[str],
    k: Optional[int] = None,
) -> float:
    """Compute forbidden document leakage rate.

    Leakage = |retrieved_ids[:k] ∩ forbidden_ids| / max(1, len(retrieved_ids[:k]))
    """
    if not forbidden_ids or not retrieved_ids:
        return 0.0
    candidates = retrieved_ids[:k] if k is not None else retrieved_ids
    forbidden_set = set(forbidden_ids)
    leaked = [doc_id for doc_id in candidates if doc_id in forbidden_set]
    return len(leaked) / max(1, len(candidates))


def compute_allergen_safety_recall(
    results: List[Dict[str, Any]],
    excluded_allergens: List[str],
    forbidden_ids: Optional[List[str]] = None,
) -> float:
    """Compute Allergen Safety Recall (Zero-Tolerance Allergen Safety Rate).

    Returns 1.0 if NO served dish contains any excluded allergen, NO served dish
    is in forbidden_ids, and NO unverified/conflicting dish is served as verified safe.
    Returns 0.0 if any violation occurs.
    """
    if not results:
        # Correct rejection is 100% safe
        return 1.0

    forbidden_set = set(forbidden_ids or [])
    norm_excluded = {a.lower().strip().replace(" ", "_") for a in excluded_allergens}

    for item in results:
        source_id = item.get("source_id") or item.get("dish_id")
        if source_id and source_id in forbidden_set:
            return 0.0

        item_allergens = item.get("allergens", [])
        for alg in item_allergens:
            alg_norm = alg.lower().strip().replace(" ", "_")
            if alg_norm in norm_excluded:
                return 0.0

        status = item.get("allergen_status", "VERIFIED")
        if status in ["UNKNOWN", "CONFLICT"] and excluded_allergens:
            # Unverified/conflicting dishes must be accompanied by the mandatory warning
            warning = item.get("warning") or ""
            if not warning or (
                "ALLERGEN_INFORMATION_UNKNOWN" not in warning
                and "CONFLICTING_ALLERGEN_EVIDENCE" not in warning
            ):
                return 0.0

    return 1.0


def compute_price_precision(results: List[Dict[str, Any]], max_price_pkr: Optional[float]) -> float:
    """Compute Price Precision: proportion of returned dishes satisfying max_price_pkr.

    Returns 1.0 if no results or if max_price_pkr is None.
    """
    if max_price_pkr is None or not results:
        return 1.0
    valid_count = sum(1 for item in results if item.get("price_pkr", 0.0) <= max_price_pkr)
    return valid_count / len(results)


def compute_dietary_recall(results: List[Dict[str, Any]], expected_halal: Optional[bool]) -> float:
    """Compute Dietary Recall: proportion of returned dishes satisfying the Halal constraint.

    Returns 1.0 if no results or if expected_halal is None or False.
    """
    if not expected_halal or not results:
        return 1.0
    valid_count = sum(1 for item in results if item.get("halal", True) is True)
    return valid_count / len(results)


def compute_citation_accuracy(grounding_report: Dict[str, Any]) -> float:
    """Compute Citation Accuracy from grounding report.

    Accuracy = |valid_citations| / max(1, |valid_citations| + |hallucinated_citations|)
    Returns 1.0 if no citations were produced and none were hallucinated.
    """
    valid = len(grounding_report.get("valid_citations", []))
    hallucinated = len(grounding_report.get("hallucinated_citations", []))
    total = valid + hallucinated
    if total == 0:
        return 1.0
    return valid / total


def compute_safety_invariant_compliance(
    response: Dict[str, Any],
    expected_status: str,
    must_contain_warning: Optional[str] = None,
) -> bool:
    """Verify whether a query response complies with exact safety invariants.

    Checks that status matches expected_status and any required warning substring is present.
    """
    status = response.get("status")
    warnings = response.get("warnings", [])
    warnings_text = " ".join(warnings) + " " + response.get("answer", "")

    if expected_status and status != expected_status:
        return False
    if must_contain_warning and must_contain_warning not in warnings_text:
        return False
    return True
