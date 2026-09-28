"""Focused UI and Streamlit integration tests (Milestone 11).

Covers:
1. Query submission via UI client.
2. Supported results rendering.
3. No-matching-items response handling.
4. Citation and evidence propagation in UI payload.
5. UNKNOWN allergen safety warning rendering (dish_009).
6. CONFLICT allergen safety warning rendering (dish_033).
7. Price, dietary, and location constraints enforcement.
8. API error handling and input validation (HTTP 422 / INVALID_INPUT).
9. Health probe verification.
10. Headless Streamlit AppTest widget rendering and execution.
"""

import pytest
from streamlit.testing.v1 import AppTest

from ui.app import (
    fetch_health_status,
    query_backend_api,
    render_dish_card,
)


# =====================================================================
# 1. Query Submission & Supported Results
# =====================================================================

def test_ui_query_supported():
    payload = {
        "query": "Grilled Chicken Breast Bowl",
        "top_k": 2,
    }
    res = query_backend_api(payload)
    assert res["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
    assert len(res["results"]) > 0
    assert len(res["citations"]) > 0
    assert "answer" in res


# =====================================================================
# 2. No Matching Items Response
# =====================================================================

def test_ui_query_no_matching_items():
    payload = {
        "query": "Find pork in Saddar under PKR 30",
    }
    res = query_backend_api(payload)
    assert res["status"] == "NO_MATCHING_ITEMS"
    assert res["results"] == []
    assert res["sources"] == []


# =====================================================================
# 3. Citation and Evidence Propagation
# =====================================================================

def test_ui_citation_evidence_propagation():
    payload = {
        "query": "Peshawari Chicken Karahi",
        "top_k": 2,
    }
    res = query_backend_api(payload)
    assert len(res["citations"]) > 0
    first_cit = res["citations"][0]
    assert "source_id" in first_cit
    assert "dish_name" in first_cit
    assert "price_pkr" in first_cit
    assert "source_menu" in first_cit


# =====================================================================
# 4. UNKNOWN Allergen Safety Invariant (dish_009)
# =====================================================================

def test_ui_unknown_allergen_warning_rendering():
    payload = {
        "query": "Mystery Special Daily Daal",
        "top_k": 1,
    }
    res = query_backend_api(payload)
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        "Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in res["warnings"]
    )
    assert res["results"][0]["allergen_status"] == "UNKNOWN"


# =====================================================================
# 5. CONFLICT Allergen Safety Invariant (dish_033)
# =====================================================================

def test_ui_conflict_allergen_warning_rendering():
    payload = {
        "query": "Chefs Secret Karahi",
        "top_k": 1,
    }
    res = query_backend_api(payload)
    assert res["status"] == "CONFLICTING_EVIDENCE"
    assert any(
        "Warning for 'Chefs Secret Karahi': CONFLICTING_ALLERGEN_EVIDENCE. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in res["warnings"]
    )
    assert res["results"][0]["allergen_status"] == "CONFLICT"


# =====================================================================
# 6. Structured Constraints Overrides (Dietary, Price, Location)
# =====================================================================

def test_ui_constraints_filtering():
    payload = {
        "query": "chicken in Gulberg",
        "excluded_allergens": ["dairy"],
        "max_price_pkr": 1500.0,
        "location": "Gulberg",
    }
    res = query_backend_api(payload)
    assert res["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
    for dish in res["results"]:
        assert dish["price_pkr"] <= 1500.0
        assert "dairy" not in [a.lower() for a in dish["allergens"]]
        assert dish["location"].lower() == "gulberg"


# =====================================================================
# 7. Input Validation & Error Handling
# =====================================================================

def test_ui_invalid_empty_query():
    payload = {"query": "   "}
    res = query_backend_api(payload)
    assert res["status"] == "INVALID_INPUT"
    assert "Input validation error" in res["answer"]


# =====================================================================
# 8. Health Status Probe
# =====================================================================

def test_ui_fetch_health():
    health = fetch_health_status()
    assert health["status"] == "healthy"
    assert health["vector_db_ready"] is True


# =====================================================================
# 9. Headless Streamlit AppTest Execution
# =====================================================================

def test_streamlit_apptest_runs_cleanly():
    from pathlib import Path
    app_path = (Path(__file__).parent.parent / "ui" / "app.py").resolve()
    at = AppTest.from_file(str(app_path))
    at.run()
    assert len(at.exception) == 0
    # Verify title
    assert len(at.title) > 0
    assert "Dynamic Food Delivery Menu & Allergen RAG" in at.title[0].value
    # Verify input box exists
    assert len(at.text_input) > 0
    # Verify buttons exist
    assert len(at.button) > 0
