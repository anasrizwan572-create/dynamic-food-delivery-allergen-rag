"""Focused integration tests for the FastAPI REST API (Milestone 10).

Covers:
1. Valid supported query via POST /query.
2. No matching items handling via POST /query.
3. Invalid request data (empty query, negative price) 422 errors.
4. Dietary and allergen constraints via explicit parameters and query text.
5. Budget and price constraints enforcement.
6. Citation and evidence propagation in API response.
7. UNKNOWN allergen safety behavior (dish_009) with exact warning.
8. CONFLICT allergen safety behavior (dish_033) with exact warning.
9. GET /health system status endpoint.
10. GET /menu catalog endpoint with filtering and pagination.
11. GET /metrics operational telemetry endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# =====================================================================
# 1. Health & Status Endpoint
# =====================================================================

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["vector_db_ready"] is True
    assert data["bm25_ready"] is True
    assert data["reranker_ready"] is True
    assert data["allergen_verifier_ready"] is True
    assert data["total_indexed_items"] > 0


# =====================================================================
# 2. Valid Supported Query
# =====================================================================

def test_query_valid_supported():
    payload = {
        "query": "Grilled Chicken Breast Bowl",
        "top_k": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
    assert len(data["results"]) > 0
    assert len(data["citations"]) > 0
    assert len(data["sources"]) > 0
    assert "answer" in data
    assert "latency" in data
    assert data["latency"]["total_ms"] >= 0


# =====================================================================
# 3. No Matching Items
# =====================================================================

def test_query_no_matching_items():
    payload = {
        "query": "Find pork in Saddar under PKR 30",
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NO_MATCHING_ITEMS"
    assert data["sources"] == []
    assert data["results"] == []
    assert data["citations"] == []
    assert "no menu item satisfies" in data["answer"].lower()


# =====================================================================
# 4. Invalid Request Data Validation (HTTP 422)
# =====================================================================

def test_query_invalid_empty_string():
    payload = {"query": "   "}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_invalid_negative_price():
    payload = {
        "query": "chicken",
        "max_price_pkr": -100.0,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_invalid_top_k_out_of_bounds():
    payload = {
        "query": "chicken",
        "top_k": 50,  # Max is 20
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422


# =====================================================================
# 5. Dietary & Allergen Constraints
# =====================================================================

def test_query_dietary_allergen_constraints():
    payload = {
        "query": "chicken in Gulberg",
        "excluded_allergens": ["dairy"],
        "max_price_pkr": 2000.0,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]

    for dish in data["results"]:
        allergens = [a.lower() for a in dish["allergens"]]
        assert "dairy" not in allergens, f"Dish {dish['dish_name']} contains dairy!"
        assert dish["price_pkr"] <= 2000.0


# =====================================================================
# 6. Price Constraints Enforcement
# =====================================================================

def test_query_price_constraints():
    payload = {
        "query": "food in Gulberg",
        "max_price_pkr": 1000.0,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    for dish in data["results"]:
        assert dish["price_pkr"] <= 1000.0, f"Dish price {dish['price_pkr']} exceeds PKR 1000!"


# =====================================================================
# 7. Citation & Evidence Propagation
# =====================================================================

def test_query_citations_evidence_propagation():
    payload = {
        "query": "Peshawari Chicken Karahi",
        "top_k": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) > 0

    citation = data["citations"][0]
    assert "source_id" in citation
    assert "dish_name" in citation
    assert "restaurant_name" in citation
    assert "price_pkr" in citation
    assert "source_menu" in citation
    assert "match_reason" in citation
    assert citation["source_id"] in data["sources"]


# =====================================================================
# 8. UNKNOWN Allergen Safety Invariant (dish_009)
# =====================================================================

def test_query_unknown_allergen_safety():
    payload = {
        "query": "Mystery Special Daily Daal",
        "top_k": 1,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        "Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in data["warnings"]
    )
    assert data["results"][0]["allergen_status"] == "UNKNOWN"
    assert data["results"][0]["verification_status"] == "INSUFFICIENT_EVIDENCE"


# =====================================================================
# 9. CONFLICT Allergen Safety Invariant (dish_033)
# =====================================================================

def test_query_conflict_allergen_safety():
    payload = {
        "query": "Chefs Secret Karahi",
        "top_k": 1,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "CONFLICTING_EVIDENCE"
    assert any(
        "Warning for 'Chefs Secret Karahi': CONFLICTING_ALLERGEN_EVIDENCE. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in data["warnings"]
    )
    assert data["results"][0]["allergen_status"] == "CONFLICT"
    assert data["results"][0]["verification_status"] == "CONFLICTING_EVIDENCE"


# =====================================================================
# 10. Menu Catalog & Metrics Endpoints
# =====================================================================

def test_menu_endpoint():
    response = client.get("/menu?location=Gulberg&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert len(data["items"]) <= 5
    for item in data["items"]:
        assert item["location"].lower() == "gulberg"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "collection_count" in data
    assert data["embedding_dimension"] == 384
    assert "retrieval_mode" in data
