"""Tests for Cross-Encoder Reranking Module."""

import pytest
from rag.metadata_filter import HardConstraints
from rag.pipeline import BasicRAGPipeline, answer_query
from rag.reranker import (
    BaseReranker,
    DeterministicFallbackReranker,
    SentenceTransformerReranker,
    get_reranker,
)


@pytest.fixture(scope="module")
def reranker():
    return get_reranker()


def test_reranker_ordering(reranker):
    """Test that candidate with exact query match is promoted to top rank."""
    candidates = [
        {"id": "dish_001", "dish_name": "Plain Rice", "document_text": "Steamed white rice"},
        {"id": "dish_002", "dish_name": "Special Mutton Karahi", "document_text": "Tender mutton cooked in wok with ginger and green chilies"},
        {"id": "dish_003", "dish_name": "Chicken Biryani", "document_text": "Aromatic chicken biryani"},
    ]

    reranked = reranker.rerank("mutton karahi", candidates, top_k=3)
    assert len(reranked) == 3
    assert reranked[0]["id"] == "dish_002"
    assert reranked[0]["dish_name"] == "Special Mutton Karahi"
    assert reranked[0]["reranker_score"] > reranked[1]["reranker_score"]


def test_stable_dish_ids(reranker):
    """Verify that dish IDs remain identical after reranking."""
    candidates = [
        {"id": "dish_010", "dish_name": "Dish 10", "document_text": "Beef Nihari with lemon"},
        {"id": "dish_020", "dish_name": "Dish 20", "document_text": "Chicken Haleem with fried onions"},
    ]

    reranked = reranker.rerank("Nihari", candidates)
    reranked_ids = [d["id"] for d in reranked]
    assert set(reranked_ids) == {"dish_010", "dish_020"}
    assert reranked[0]["id"] == "dish_010"


def test_metadata_preservation(reranker):
    """Verify all rich metadata fields are preserved intact during reranking."""
    candidate = {
        "id": "dish_007",
        "dish_name": "Street Style Bun Kabab",
        "restaurant_name": "Karachi Bun Kabab Corner",
        "price_pkr": 350.0,
        "location": "Saddar",
        "halal": True,
        "spice_level": "Spicy",
        "allergens": ["eggs", "gluten"],
        "allergen_status": "KNOWN_ALLERGENS",
        "allergen_warning": None,
        "metadata": {"protein_g": 18.0, "cuisine": "Pakistani"},
        "document_text": "Bun Kabab with lentils, eggs, and mint chutney",
        "rrf_score": 0.032,
    }

    reranked = reranker.rerank("Bun Kabab", [candidate])
    assert len(reranked) == 1
    res = reranked[0]

    assert res["id"] == "dish_007"
    assert res["dish_name"] == "Street Style Bun Kabab"
    assert res["price_pkr"] == 350.0
    assert res["location"] == "Saddar"
    assert res["halal"] is True
    assert res["spice_level"] == "Spicy"
    assert res["allergens"] == ["eggs", "gluten"]
    assert res["allergen_status"] == "KNOWN_ALLERGENS"
    assert res["metadata"]["protein_g"] == 18.0
    assert "reranker_score" in res


def test_empty_candidate_list(reranker):
    """Test that empty candidate list returns empty list safely."""
    assert reranker.rerank("query", []) == []


def test_single_candidate(reranker):
    """Test reranking single candidate assigns score and returns it."""
    candidate = {"id": "dish_001", "dish_name": "Chicken Tikka", "document_text": "Barbecue chicken"}
    reranked = reranker.rerank("chicken", [candidate])
    assert len(reranked) == 1
    assert reranked[0]["id"] == "dish_001"
    assert "reranker_score" in reranked[0]


def test_deterministic_tie_handling(reranker):
    """Verify deterministic tie-breaking on dish ID ascending when scores are equal."""
    # Two identical documents with identical scores
    candidates = [
        {"id": "dish_009", "dish_name": "Daal", "document_text": "Yellow lentils"},
        {"id": "dish_002", "dish_name": "Daal", "document_text": "Yellow lentils"},
    ]

    reranked = reranker.rerank("daal", candidates)
    assert len(reranked) == 2
    # Equal score must be ordered ascending by ID: dish_002 before dish_009
    assert reranked[0]["id"] == "dish_002"
    assert reranked[1]["id"] == "dish_009"


def test_hard_constraints_remain_enforced():
    """Verify that an item rejected by upstream hard constraints cannot reach or be returned by reranking."""
    pipeline = BasicRAGPipeline()
    # Query with strict price ceiling of PKR 500 for Butter Chicken (which costs PKR 1450)
    result = pipeline.answer_query("Murgh Makhani Butter Chicken under PKR 500", top_k=5)

    retrieved_ids = [d["id"] for d in result["retrieved_documents"]]
    # dish_002 is Murgh Makhani Butter Chicken (PKR 1450)
    assert "dish_002" not in retrieved_ids
    assert all(d["price_pkr"] <= 500.0 for d in result["retrieved_documents"])


def test_unknown_allergen_preservation(reranker):
    """Verify UNKNOWN allergen status and warnings are preserved across reranking."""
    candidate = {
        "id": "dish_009",
        "dish_name": "Mystery Special Daily Daal",
        "allergen_status": "UNKNOWN",
        "allergen_warning": "ALLERGEN_INFORMATION_UNKNOWN",
        "document_text": "Chef special daily secret daal",
    }

    reranked = reranker.rerank("daal", [candidate])
    assert len(reranked) == 1
    assert reranked[0]["allergen_status"] == "UNKNOWN"
    assert reranked[0]["allergen_warning"] == "ALLERGEN_INFORMATION_UNKNOWN"


def test_conflict_allergen_preservation(reranker):
    """Verify CONFLICT allergen status and warnings are preserved across reranking."""
    candidate = {
        "id": "dish_033",
        "dish_name": "Chefs Secret Karahi",
        "allergen_status": "CONFLICT",
        "allergen_warning": "CONFLICTING_ALLERGEN_EVIDENCE",
        "document_text": "Conflicting allergen chicken karahi",
    }

    reranked = reranker.rerank("karahi", [candidate])
    assert len(reranked) == 1
    assert reranked[0]["allergen_status"] == "CONFLICT"
    assert reranked[0]["allergen_warning"] == "CONFLICTING_ALLERGEN_EVIDENCE"


def test_pipeline_end_to_end_with_reranker():
    """Verify pipeline integrates reranking with valid telemetry and citations."""
    result = answer_query("spicy chicken", top_k=3)
    assert result["status"] == "SUPPORTED"
    assert len(result["retrieved_documents"]) <= 3
    assert "rerank_ms" in result["latency"]
    assert result["latency"]["rerank_ms"] >= 0.0

    for doc in result["retrieved_documents"]:
        assert "reranker_score" in doc
        assert doc["id"] in result["sources"]
