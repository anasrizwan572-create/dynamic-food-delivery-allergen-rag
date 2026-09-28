"""Tests for BM25 Lexical Retrieval, Hybrid Retrieval, and Reciprocal Rank Fusion (RRF)."""

import pytest
from rag.bm25 import BM25Index, get_default_bm25_index, tokenize
from rag.hybrid_retriever import (
    HybridRetriever,
    get_default_hybrid_retriever,
    reciprocal_rank_fusion,
)
from rag.metadata_filter import HardConstraints


@pytest.fixture(scope="module")
def bm25_index():
    return get_default_bm25_index()


@pytest.fixture(scope="module")
def hybrid_retriever():
    return get_default_hybrid_retriever()


def test_tokenizer():
    """Verify regex tokenization handles punctuation, casing, and spaces."""
    tokens = tokenize("Murgh Makhani (Butter Chicken) - PKR 1450!")
    assert "murgh" in tokens
    assert "makhani" in tokens
    assert "butter" in tokens
    assert "chicken" in tokens
    assert "pkr" in tokens
    assert "1450" in tokens


def test_exact_bm25_keyword_matching(bm25_index):
    """Test BM25 keyword matching returns matching items with positive score."""
    results = bm25_index.search("skewers", top_k=5)
    assert len(results) > 0
    assert results[0]["id"] == "dish_004"
    assert "Chicken Satay Skewers" in results[0]["dish_name"]
    assert results[0]["bm25_score"] > 0.0


def test_dish_name_matching(bm25_index):
    """Test exact dish name matching ranks target dish at #1."""
    results = bm25_index.search("Street Style Bun Kabab", top_k=3)
    assert len(results) > 0
    top_dish = results[0]
    assert top_dish["id"] == "dish_007"
    assert top_dish["dish_name"] == "Street Style Bun Kabab"
    assert top_dish["score"] > 5.0


def test_ingredient_term_matching(bm25_index):
    """Test specific ingredient search matches ingredient tokens."""
    results = bm25_index.search("basmati", top_k=5)
    matched_ids = [r["id"] for r in results]
    # dish_013 is Plain Steamed Basmati Rice, dish_012 is Mutton Dum Biryani
    assert "dish_013" in matched_ids
    assert "dish_012" in matched_ids
    assert any("basmati" in r["metadata"].get("ingredients", "").lower() for r in results)


def test_dense_and_bm25_result_merging():
    """Test merging disparate dense and BM25 candidate lists."""
    dense = [
        {"id": "dish_001", "dish_name": "Dish 1", "score": 0.9},
        {"id": "dish_002", "dish_name": "Dish 2", "score": 0.8},
    ]
    bm25 = [
        {"id": "dish_003", "dish_name": "Dish 3", "score": 4.5},
        {"id": "dish_001", "dish_name": "Dish 1", "score": 3.2},
    ]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)
    fused_ids = [d["id"] for d in fused]

    # All 3 unique dishes must be present
    assert len(fused) == 3
    assert set(fused_ids) == {"dish_001", "dish_002", "dish_003"}


def test_duplicate_removal_and_aggregation():
    """Verify duplicates across dense and BM25 are merged with combined RRF scores."""
    dense = [
        {"id": "dish_001", "dish_name": "Dish 1", "similarity": 0.95},
        {"id": "dish_002", "dish_name": "Dish 2", "similarity": 0.85},
    ]
    bm25 = [
        {"id": "dish_001", "dish_name": "Dish 1", "bm25_score": 6.0},
        {"id": "dish_003", "dish_name": "Dish 3", "bm25_score": 4.0},
    ]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)
    assert len(fused) == 3

    # dish_001 is rank 1 in dense (1/61) and rank 1 in BM25 (1/61)
    dish1 = next(d for d in fused if d["id"] == "dish_001")
    expected_score = round((1.0 / 61) + (1.0 / 61), 8)
    assert dish1["score"] == expected_score
    assert dish1["dense_rank"] == 1
    assert dish1["bm25_rank"] == 1


def test_rrf_score_calculation():
    """Verify exact formula: RRF_score = 1 / (k + rank)."""
    dense = [
        {"id": "dish_A", "score": 0.99},  # rank 1
        {"id": "dish_B", "score": 0.90},  # rank 2
    ]
    bm25 = [
        {"id": "dish_C", "score": 5.0},   # rank 1
        {"id": "dish_A", "score": 3.0},   # rank 2
    ]

    k = 60
    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=k)
    fused_map = {d["id"]: d for d in fused}

    # dish_A: 1/(60+1) + 1/(60+2)
    expected_A = round((1.0 / 61) + (1.0 / 62), 8)
    # dish_C: 1/(60+1)
    expected_C = round(1.0 / 61, 8)
    # dish_B: 1/(60+2)
    expected_B = round(1.0 / 62, 8)

    assert fused_map["dish_A"]["score"] == expected_A
    assert fused_map["dish_C"]["score"] == expected_C
    assert fused_map["dish_B"]["score"] == expected_B

    # dish_A has highest score because it appeared in both
    assert fused[0]["id"] == "dish_A"


def test_configurable_rrf_k():
    """Verify RRF score sensitivity to configurable k parameter."""
    dense = [{"id": "dish_A", "score": 0.99}]
    bm25 = [{"id": "dish_A", "score": 5.0}]

    fused_k10 = reciprocal_rank_fusion(dense, bm25, rrf_k=10)
    fused_k60 = reciprocal_rank_fusion(dense, bm25, rrf_k=60)

    # With k=10, rank 1 in both is 1/11 + 1/11 = 2/11 ≈ 0.18181818
    assert fused_k10[0]["score"] == round(2.0 / 11, 8)
    # With k=60, rank 1 in both is 1/61 + 1/61 = 2/61 ≈ 0.03278689
    assert fused_k60[0]["score"] == round(2.0 / 61, 8)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(dense, bm25, rrf_k=0)


def test_empty_bm25_results():
    """Verify behavior when BM25 finds zero lexical matches."""
    dense = [
        {"id": "dish_001", "dish_name": "Dish 1", "score": 0.8},
        {"id": "dish_002", "dish_name": "Dish 2", "score": 0.7},
    ]
    bm25 = []

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)
    assert len(fused) == 2
    assert fused[0]["id"] == "dish_001"
    assert fused[0]["score"] == round(1.0 / 61, 8)
    assert fused[0]["bm25_rank"] is None
    assert fused[0]["dense_rank"] == 1


def test_empty_dense_results():
    """Verify behavior when dense vector retrieval returns zero results."""
    dense = []
    bm25 = [
        {"id": "dish_007", "dish_name": "Bun Kabab", "score": 7.5},
    ]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)
    assert len(fused) == 1
    assert fused[0]["id"] == "dish_007"
    assert fused[0]["score"] == round(1.0 / 61, 8)
    assert fused[0]["dense_rank"] is None
    assert fused[0]["bm25_rank"] == 1


def test_hard_metadata_constraints_remaining_enforced(hybrid_retriever):
    """Verify upstream hard constraints eliminate matching items before hybrid ranking.

    Even if Murgh Makhani Butter Chicken (PKR 1450) is an exact BM25 match,
    a budget constraint of PKR 500 must strictly exclude it.
    """
    constraints = HardConstraints(max_price_pkr=500.0)
    candidates, filter_result = hybrid_retriever.retrieve_filtered(
        query="Murgh Makhani Butter Chicken",
        constraints=constraints,
        top_k=5,
    )

    # Murgh Makhani is dish_002 (PKR 1450)
    candidate_ids = [c["id"] for c in candidates]
    assert "dish_002" not in candidate_ids
    # All returned candidates must be <= 500 PKR
    assert all(c["price_pkr"] <= 500.0 for c in candidates)
    # The pre-filter must record rejection
    rejected_ids = [r["id"] for r in filter_result.rejected_candidates]
    assert "dish_002" in rejected_ids


def test_allergen_hard_exclusion_in_hybrid(hybrid_retriever):
    """Verify allergen exclusion cannot be bypassed by BM25 high score."""
    # dish_004 is Chicken Satay Skewers containing peanuts
    constraints = HardConstraints(excluded_allergens=["peanuts"])
    candidates, filter_result = hybrid_retriever.retrieve_filtered(
        query="Chicken Satay Skewers",
        constraints=constraints,
        top_k=5,
    )
    candidate_ids = [c["id"] for c in candidates]
    assert "dish_004" not in candidate_ids
    assert all("peanuts" not in c["allergens"] for c in candidates)


def test_unknown_and_conflict_allergen_safety_enforced(hybrid_retriever):
    """Verify UNKNOWN and CONFLICT allergen items retain status and trigger warnings."""
    # When allergens are excluded, dish_009 (UNKNOWN) and dish_033 (CONFLICT) must carry safety warnings
    constraints = HardConstraints(excluded_allergens=["dairy"])
    candidates, filter_result = hybrid_retriever.retrieve_filtered(
        query="Daal Karahi",
        constraints=constraints,
        top_k=10,
    )

    warnings = filter_result.warnings
    # dish_009 has allergen_status == 'UNKNOWN'
    # dish_033 has allergen_status == 'CONFLICT'
    assert any("ALLERGEN_INFORMATION_UNKNOWN" in w for w in warnings)
    assert any("CONFLICTING_ALLERGEN_EVIDENCE" in w for w in warnings)


def test_hybrid_retriever_empty_query(hybrid_retriever):
    """Verify empty query returns empty candidate list safely."""
    assert hybrid_retriever.retrieve("") == []
    assert hybrid_retriever.retrieve("   ") == []
