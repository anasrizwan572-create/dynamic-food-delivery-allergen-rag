"""Unit tests for the Milestone 8 Allergen Safety Verifier Module."""

import pytest
from rag.allergen_verifier import (
    AllergenSafetyStatus,
    AllergenSafetyVerifier,
    get_allergen_verifier,
)
from rag.pipeline import answer_query


@pytest.fixture(scope="module")
def verifier():
    return get_allergen_verifier()


def test_verified_pass_clean_dish(verifier):
    """Test that a dish with no excluded allergens receives VERIFIED_PASS."""
    candidate = {
        "id": "dish_001",
        "dish_name": "Grilled Chicken Breast Bowl",
        "allergens": [],
        "allergen_status": "KNOWN_NO_ALLERGENS",
        "ingredients": "chicken breast, brown rice, olive oil, lemon",
        "description": "Char-grilled chicken with brown rice",
    }

    res = verifier.verify_item(candidate, excluded_allergens=["peanuts", "dairy"])
    assert res.status == AllergenSafetyStatus.VERIFIED_PASS
    assert res.rejection_reason is None
    assert res.warning is None


def test_reject_declared_allergen(verifier):
    """Test that explicit match in declared allergens triggers REJECT."""
    candidate = {
        "id": "dish_004",
        "dish_name": "Chicken Satay Skewers",
        "allergens": ["peanuts", "soy"],
        "allergen_status": "KNOWN_ALLERGENS",
        "ingredients": "chicken, peanut sauce, soy sauce",
    }

    res = verifier.verify_item(candidate, excluded_allergens=["peanuts"])
    assert res.status == AllergenSafetyStatus.REJECT
    assert "Dish declared excluded allergen(s)" in res.rejection_reason
    assert "peanuts" in res.rejection_reason


def test_reject_hidden_ingredient_trigger(verifier):
    """Test that hidden allergen trigger in ingredient text triggers REJECT."""
    candidate = {
        "id": "dish_099",
        "dish_name": "Special Gravy Chicken",
        "allergens": [],  # Undeclared in metadata
        "allergen_status": "KNOWN_NO_ALLERGENS",
        "ingredients": "chicken, tomatoes, desi ghee, fenugreek",  # Ghee contains dairy
    }

    res = verifier.verify_item(candidate, excluded_allergens=["dairy"])
    assert res.status == AllergenSafetyStatus.REJECT
    assert "contain excluded allergen trigger(s)" in res.rejection_reason
    assert "dairy" in res.rejection_reason


def test_dish_009_unknown_allergen_invariant(verifier):
    """Safety Invariant: dish_009 must remain UNKNOWN and emit exact warning."""
    candidate = {
        "id": "dish_009",
        "dish_name": "Mystery Special Daily Daal",
        "allergens": ["unknown"],
        "allergen_status": "UNKNOWN",
        "ingredients": "Chef secret lentils blend and proprietary spices",
    }

    res = verifier.verify_item(candidate, excluded_allergens=["dairy"])
    assert res.status == AllergenSafetyStatus.INSUFFICIENT_EVIDENCE
    assert res.allergen_status == "UNKNOWN"

    exact_warning = (
        "Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. "
        "Cannot guarantee allergen-free status due to unverified allergen data."
    )
    assert res.warning == exact_warning


def test_dish_033_conflict_allergen_invariant(verifier):
    """Safety Invariant: dish_033 must remain CONFLICT and emit exact warning."""
    candidate = {
        "id": "dish_033",
        "dish_name": "Chefs Secret Karahi",
        "allergens": ["conflict"],
        "allergen_status": "CONFLICT",
        "ingredients": "Special chicken karahi with disputed recipe notes",
    }

    res = verifier.verify_item(candidate, excluded_allergens=["dairy"])
    assert res.status == AllergenSafetyStatus.CONFLICTING_EVIDENCE
    assert res.allergen_status == "CONFLICT"

    exact_warning = (
        "Warning for 'Chefs Secret Karahi': CONFLICTING_ALLERGEN_EVIDENCE. "
        "Cannot guarantee allergen-free status due to unverified allergen data."
    )
    assert res.warning == exact_warning


def test_missing_evidence_not_claimed_safe(verifier):
    """Verify missing/empty allergen records are never claimed allergen-free."""
    candidate = {
        "id": "dish_101",
        "dish_name": "Unverified Soup",
        "allergens": [],
        "allergen_status": "UNKNOWN",
        "ingredients": "",
    }

    res = verifier.verify_item(candidate, excluded_allergens=["gluten"])
    assert res.status == AllergenSafetyStatus.INSUFFICIENT_EVIDENCE
    assert "ALLERGEN_INFORMATION_UNKNOWN" in res.warning


def test_verify_candidates_report_structure(verifier):
    """Verify batch verification report separates passed, rejected, and warnings."""
    candidates = [
        {"id": "dish_001", "dish_name": "Clean Chicken", "allergens": [], "allergen_status": "KNOWN_NO_ALLERGENS", "ingredients": "chicken, salt"},
        {"id": "dish_004", "dish_name": "Satay", "allergens": ["peanuts"], "allergen_status": "KNOWN_ALLERGENS", "ingredients": "peanuts"},
        {"id": "dish_009", "dish_name": "Mystery Special Daily Daal", "allergens": ["unknown"], "allergen_status": "UNKNOWN"},
    ]

    report = verifier.verify_candidates(candidates, excluded_allergens=["peanuts"])
    verified_ids = [c["id"] for c in report.verified_candidates]
    rejected_ids = [c["id"] for c in report.rejected_candidates]

    assert "dish_001" in verified_ids
    assert "dish_009" in verified_ids  # Passed with warning
    assert "dish_004" in rejected_ids  # Hard rejected
    assert any("ALLERGEN_INFORMATION_UNKNOWN" in w for w in report.warnings)


def test_empty_candidates_handling(verifier):
    """Verify empty candidate list returns empty report safely."""
    report = verifier.verify_candidates([], excluded_allergens=["dairy"])
    assert report.verified_candidates == []
    assert report.rejected_candidates == []
    assert report.warnings == []


def test_no_excluded_allergens_query(verifier):
    """Verify queries without allergen exclusions still preserve UNKNOWN/CONFLICT warnings."""
    candidates = [
        {"id": "dish_009", "dish_name": "Mystery Special Daily Daal", "allergens": ["unknown"], "allergen_status": "UNKNOWN"},
        {"id": "dish_001", "dish_name": "Clean Chicken", "allergens": [], "allergen_status": "KNOWN_NO_ALLERGENS"},
    ]
    report = verifier.verify_candidates(candidates, excluded_allergens=[])
    assert len(report.verified_candidates) == 2
    assert any("ALLERGEN_INFORMATION_UNKNOWN" in w for w in report.warnings)


def test_pipeline_end_to_end_allergen_verification():
    """Verify end-to-end pipeline execution includes verify_ms and exact safety warnings."""
    res = answer_query("chicken without peanuts", top_k=3)
    assert res["status"] == "SUPPORTED"
    assert "verify_ms" in res["latency"]
    assert res["latency"]["verify_ms"] >= 0.0

    # Ensure no retrieved dish contains peanuts
    for doc in res["retrieved_documents"]:
        assert "peanuts" not in [a.lower() for a in doc.get("allergens", [])]
