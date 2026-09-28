"""Unit tests for the Citation Engine and Grounded Synthesis (Milestone 9).

Covers:
1. Correctly grounded answers with citations.
2. Multiple supporting evidence items.
3. Insufficient evidence handling.
4. Unsupported claims and hallucinated citation prevention.
5. Interaction with UNKNOWN allergen evidence (dish_009).
6. Interaction with CONFLICT allergen evidence (dish_033).
7. Unsupported allergen-free claim on declared allergen.
8. Deterministic grounded answer synthesis.
9. Pipeline end-to-end grounded citations and results.
10. Pipeline unknown allergen INSUFFICIENT_EVIDENCE status.
"""

import pytest
from rag.citation import (
    Citation,
    CitationEngine,
    GroundingStatus,
    GroundingVerificationReport,
    get_citation_engine,
)
from rag.pipeline import answer_query


SAMPLE_DOC_1 = {
    "id": "dish_001",
    "dish_name": "Grilled Chicken Breast Bowl",
    "restaurant_name": "Monal Express",
    "price_pkr": 1150.0,
    "location": "Gulberg",
    "halal": True,
    "spice_level": "Mild",
    "allergens": [],
    "allergen_status": "KNOWN_NO_ALLERGENS",
    "metadata": {
        "id": "dish_001",
        "dish_name": "Grilled Chicken Breast Bowl",
        "restaurant_name": "Monal Express",
        "price_pkr": 1150.0,
        "location": "Gulberg",
        "description": "Char-grilled chicken breast with steamed rice",
        "ingredients": "chicken breast, brown rice, olive oil, lemon juice",
        "source": "Monal Express Official Menu",
    },
}

SAMPLE_DOC_2 = {
    "id": "dish_002",
    "dish_name": "Special Chicken Makhani Karahi",
    "restaurant_name": "Salt'n Pepper",
    "price_pkr": 1850.0,
    "location": "Gulberg",
    "halal": True,
    "spice_level": "Medium",
    "allergens": ["dairy"],
    "allergen_status": "KNOWN_ALLERGENS",
    "metadata": {
        "id": "dish_002",
        "dish_name": "Special Chicken Makhani Karahi",
        "restaurant_name": "Salt'n Pepper",
        "price_pkr": 1850.0,
        "location": "Gulberg",
        "description": "Rich butter chicken karahi cooked in fresh desi makhan and cream",
        "ingredients": "chicken, butter, cream, tomatoes, ginger, spices",
        "source": "Salt'n Pepper Menu 2026",
    },
}

SAMPLE_DOC_UNKNOWN = {
    "id": "dish_009",
    "dish_name": "Mystery Special Daily Daal",
    "restaurant_name": "Anarkali Dhaba",
    "price_pkr": 400.0,
    "location": "Gulberg",
    "halal": True,
    "spice_level": "Medium",
    "allergens": ["unknown"],
    "allergen_status": "UNKNOWN",
    "verification_status": "INSUFFICIENT_EVIDENCE",
    "allergen_warning": "ALLERGEN_INFORMATION_UNKNOWN",
    "metadata": {
        "id": "dish_009",
        "dish_name": "Mystery Special Daily Daal",
        "restaurant_name": "Anarkali Dhaba",
        "price_pkr": 400.0,
        "location": "Gulberg",
        "description": "Daily chef special slow-cooked spiced lentils with secret tarka",
        "ingredients": "yellow lentils, unverified oil, garlic, cumin",
        "source": "Anarkali Dhaba Daily Board",
    },
}

SAMPLE_DOC_CONFLICT = {
    "id": "dish_033",
    "dish_name": "Chefs Secret Karahi",
    "restaurant_name": "Lahore Shinwari",
    "price_pkr": 2100.0,
    "location": "Gulberg",
    "halal": True,
    "spice_level": "High",
    "allergens": ["conflict"],
    "allergen_status": "CONFLICT",
    "verification_status": "CONFLICTING_EVIDENCE",
    "allergen_warning": "CONFLICTING_ALLERGEN_EVIDENCE",
    "metadata": {
        "id": "dish_033",
        "dish_name": "Chefs Secret Karahi",
        "restaurant_name": "Lahore Shinwari",
        "price_pkr": 2100.0,
        "location": "Gulberg",
        "description": "Contradictory allergen claims between menu board and kitchen bulletin",
        "ingredients": "mutton, unverified broth, animal fat, ginger",
        "source": "Shinwari Disputed Notice",
    },
}


@pytest.fixture
def engine():
    return CitationEngine()


# =====================================================================
# 1. Correctly Grounded Answers with Citations
# =====================================================================

def test_citation_extraction(engine):
    text = (
        "Here are recommendations: [dish_001] is grilled chicken. "
        "Also check [dish_002] and [dish_001] again. Ignore [DOCUMENT 1] and [99]."
    )
    extracted = engine.extract_citations(text)
    assert extracted == ["dish_001", "dish_002"]


def test_correctly_grounded_answer_with_single_citation(engine):
    answer = (
        "SUPPORTED\n\n"
        "1. Grilled Chicken Breast Bowl at Monal Express [dish_001]\n"
        "   - Price: PKR 1150.0\n"
        "   - Location: Gulberg\n"
        "   - Halal: Yes (Certified)\n"
        "   - Allergens noted: None listed (Verified)\n"
        "   - Source: Monal Express Official Menu [dish_001]"
    )
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_1],
        query="What chicken bowl is in Gulberg?",
    )
    assert report.is_grounded is True
    assert report.status == GroundingStatus.SUPPORTED
    assert len(report.valid_citations) == 1
    assert report.valid_citations[0].source_id == "dish_001"
    assert report.valid_citations[0].dish_name == "Grilled Chicken Breast Bowl"
    assert report.hallucinated_citations == []
    assert report.unsupported_claims == []


# =====================================================================
# 2. Multiple Supporting Evidence Items
# =====================================================================

def test_multiple_supporting_evidence_items(engine):
    answer = (
        "SUPPORTED\n\n"
        "1. Grilled Chicken Breast Bowl at Monal Express [dish_001] - PKR 1150.0\n"
        "2. Special Chicken Makhani Karahi at Salt'n Pepper [dish_002] - PKR 1850.0"
    )
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_1, SAMPLE_DOC_2],
        query="Chicken dishes in Gulberg",
    )
    assert report.is_grounded is True
    assert len(report.valid_citations) == 2
    assert [c.source_id for c in report.valid_citations] == ["dish_001", "dish_002"]
    assert report.evidence_count == 2
    assert report.hallucinated_citations == []


# =====================================================================
# 3. Insufficient Evidence Handling
# =====================================================================

def test_insufficient_evidence_empty_context(engine):
    answer = "Based on the retrieved menu information, there is insufficient evidence to answer your request."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[],
        query="Find kangaroo steak in Saddar",
    )
    assert report.is_grounded is True
    assert report.status == GroundingStatus.INSUFFICIENT_EVIDENCE
    assert report.valid_citations == []
    assert report.evidence_count == 0


# =====================================================================
# 4. Unsupported Claims & Hallucinated Citation Prevention
# =====================================================================

def test_hallucinated_citation_detection(engine):
    # Citing dish_999 which does not exist in retrieved docs
    answer = "We recommend the Phantom Curry at Ghost Diner [dish_999] for PKR 800."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_1],
        query="Recommend a dish",
    )
    assert report.is_grounded is False
    assert report.status == GroundingStatus.UNSUPPORTED
    assert "dish_999" in report.hallucinated_citations
    assert any("Hallucinated citation: [dish_999]" in c for c in report.unsupported_claims)


def test_unsupported_price_claim_detection(engine):
    # dish_001 is PKR 1150.0, but answer asserts PKR 250.0
    answer = "Grilled Chicken Breast Bowl [dish_001] is available for only PKR 250.0 at Monal Express."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_1],
        query="Price of dish 001",
    )
    assert report.is_grounded is False
    assert any("Price contradiction" in c for c in report.unsupported_claims)


# =====================================================================
# 5. Interaction with UNKNOWN Allergen Evidence (dish_009)
# =====================================================================

def test_interaction_with_unknown_allergen_unsupported_claim(engine):
    # False claim that dish_009 is allergen-free
    answer = "Mystery Special Daily Daal [dish_009] is completely allergen-free and guaranteed safe for allergy sufferers."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_UNKNOWN],
        query="Is daal safe from allergens?",
    )
    assert report.is_grounded is False
    assert any("UNKNOWN" in c for c in report.unsupported_claims)


def test_interaction_with_unknown_allergen_grounded_warning(engine):
    # Proper answer noting unknown allergen status with exact warning
    answer = (
        "INSUFFICIENT_EVIDENCE\n\n"
        "1. Mystery Special Daily Daal at Anarkali Dhaba [dish_009]\n"
        "   - Price: PKR 400.0\n"
        "   - Allergens noted: Unverified / Unknown (Missing data)\n"
        "   - Warning: ALLERGEN_INFORMATION_UNKNOWN\n"
        "   - Source: Anarkali Dhaba Daily Board [dish_009]"
    )
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_UNKNOWN],
        query="Mystery Special Daily Daal",
    )
    assert report.is_grounded is True
    assert report.status == GroundingStatus.INSUFFICIENT_EVIDENCE
    assert any(
        "Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in report.warnings
    )


# =====================================================================
# 6. Interaction with CONFLICT Allergen Evidence (dish_033)
# =====================================================================

def test_interaction_with_conflict_allergen_unsupported_claim(engine):
    # False claim that dish_033 is safe
    answer = "Chefs Secret Karahi [dish_033] is 100% safe from all allergens."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_CONFLICT],
        query="Is Chefs Secret Karahi safe?",
    )
    assert report.is_grounded is False
    assert any("CONFLICTING" in c for c in report.unsupported_claims)


def test_interaction_with_conflict_allergen_grounded_warning(engine):
    # Proper answer reporting conflicting evidence with exact warning
    answer = (
        "CONFLICTING_EVIDENCE\n\n"
        "1. Chefs Secret Karahi at Lahore Shinwari [dish_033]\n"
        "   - Price: PKR 2100.0\n"
        "   - Allergens noted: Conflicting Evidence (Contradictory source records)\n"
        "   - Source: Shinwari Disputed Notice [dish_033]"
    )
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_CONFLICT],
        query="Chefs Secret Karahi",
    )
    assert report.is_grounded is True
    assert report.status == GroundingStatus.CONFLICTING_EVIDENCE
    assert any(
        "Warning for 'Chefs Secret Karahi': CONFLICTING_ALLERGEN_EVIDENCE. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in report.warnings
    )


# =====================================================================
# 7. Unsupported Allergen-Free Claim on Declared Allergen
# =====================================================================

def test_unsupported_allergen_claim_on_declared_allergen(engine):
    # dish_002 has dairy in allergens, but answer claims it is dairy-free
    answer = "Special Chicken Makhani Karahi [dish_002] is a delicious dairy-free option cooked in olive oil."
    report = engine.verify_grounding(
        answer=answer,
        retrieved_docs=[SAMPLE_DOC_2],
        query="Dairy free chicken karahi",
    )
    assert report.is_grounded is False
    assert any("dairy-free" in c for c in report.unsupported_claims)


# =====================================================================
# 8. Deterministic Grounded Synthesis
# =====================================================================

def test_synthesize_grounded_answer_deterministic(engine):
    answer, citations, status = engine.synthesize_grounded_answer(
        query="chicken in Gulberg",
        retrieved_docs=[SAMPLE_DOC_1],
    )
    assert status == GroundingStatus.SUPPORTED
    assert "SUPPORTED" in answer
    assert "Grilled Chicken Breast Bowl" in answer
    assert "Monal Express" in answer
    assert "PKR 1150.0" in answer
    assert "[dish_001]" in answer
    assert len(citations) == 1
    assert citations[0].source_id == "dish_001"


# =====================================================================
# 9. Pipeline End-to-End Grounded Citations
# =====================================================================

def test_pipeline_end_to_end_grounded_citations():
    res = answer_query("Grilled Chicken Breast Bowl", top_k=2)
    assert res["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
    assert "citations" in res
    assert isinstance(res["citations"], list)
    assert len(res["citations"]) > 0
    assert "grounding" in res
    assert res["grounding"]["is_grounded"] is True
    assert "results" in res
    assert len(res["results"]) > 0

    first_citation = res["citations"][0]
    assert "source_id" in first_citation
    assert "dish_name" in first_citation
    assert "price_pkr" in first_citation
    assert "restaurant_name" in first_citation
    assert "match_reason" in first_citation


# =====================================================================
# 10. Pipeline Unknown Allergen INSUFFICIENT_EVIDENCE Status
# =====================================================================

def test_pipeline_unknown_allergen_insufficient_evidence_status():
    res = answer_query("Mystery Special Daily Daal", top_k=1)
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        "Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."
        in w for w in res["warnings"]
    )
