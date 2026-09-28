"""Unit tests for the context_builder module."""

import pytest
from rag.context_builder import build_context, format_single_document


SAMPLE_DOC = {
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
        "description": "Char-grilled chicken breast with steamed rice",
        "ingredients": "chicken breast, rice, olive oil",
        "source": "Monal Express Menu 2026",
    },
}

SAMPLE_DOC_UNKNOWN = {
    "id": "dish_009",
    "dish_name": "Mystery Daily Daal",
    "restaurant_name": "Anarkali Dhaba",
    "price_pkr": 400.0,
    "location": "Gulberg",
    "halal": True,
    "spice_level": "Medium",
    "allergens": ["unknown"],
    "allergen_status": "UNKNOWN",
    "metadata": {
        "id": "dish_009",
        "description": "Slow cooked lentils",
        "ingredients": "lentils, unverified oil",
        "source": "Anarkali Dhaba Board",
    },
}


def test_format_single_document_complete():
    formatted = format_single_document(SAMPLE_DOC, doc_index=1)
    assert "[DOCUMENT 1]" in formatted
    assert "Dish Name: Grilled Chicken Breast Bowl" in formatted
    assert "Restaurant: Monal Express" in formatted
    assert "Price: PKR 1150.0" in formatted
    assert "Location: Gulberg" in formatted
    assert "Halal Status: Yes (Certified)" in formatted
    assert "Spice Level: Mild" in formatted
    assert "Allergens: None listed (Verified)" in formatted
    assert "Source ID: dish_001" in formatted


def test_format_single_document_unknown_allergens():
    formatted = format_single_document(SAMPLE_DOC_UNKNOWN, doc_index=2)
    assert "[DOCUMENT 2]" in formatted
    assert "Allergens: Unverified / Unknown (Missing data)" in formatted
    assert "Source ID: dish_009" in formatted


def test_build_context_multiple_documents():
    context_str, source_ids = build_context([SAMPLE_DOC, SAMPLE_DOC_UNKNOWN])
    assert "[DOCUMENT 1]" in context_str
    assert "[DOCUMENT 2]" in context_str
    assert source_ids == ["dish_001", "dish_009"]


def test_build_context_empty():
    context_str, source_ids = build_context([])
    assert "No retrieved documents available" in context_str
    assert source_ids == []


def test_build_context_missing_metadata_handled():
    sparse_doc = {"id": "dish_sparse"}
    formatted = format_single_document(sparse_doc, doc_index=1)
    assert "[DOCUMENT 1]" in formatted
    assert "Source ID: dish_sparse" in formatted
