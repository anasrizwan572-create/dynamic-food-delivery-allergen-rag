"""Unit tests for the Generator and grounding synthesis."""

import pytest
from rag.generator import MockGroundedGenerator, get_generator


SAMPLE_DOCS = [
    {
        "id": "dish_001",
        "dish_name": "Grilled Chicken Breast Bowl",
        "restaurant_name": "Monal Express",
        "price_pkr": 1150.0,
        "location": "Gulberg",
        "spice_level": "Mild",
        "allergens": [],
    }
]


def test_mock_generator_grounded_response():
    gen = MockGroundedGenerator()
    context = "[DOCUMENT 1]\nDish: Grilled Chicken Breast Bowl\nPrice: PKR 1150.0\nSource ID: dish_001"
    response = gen.generate(
        query="What chicken is in Gulberg?",
        context=context,
        retrieved_docs=SAMPLE_DOCS,
    )
    assert "Grilled Chicken Breast Bowl" in response
    assert "Monal Express" in response
    assert "[dish_001]" in response
    assert "PKR 1150.0" in response


def test_mock_generator_empty_context():
    gen = MockGroundedGenerator()
    response = gen.generate(
        query="Non-existent dish",
        context="No retrieved documents available.",
        retrieved_docs=[],
    )
    assert "insufficient evidence" in response.lower()


def test_get_generator_factory_default():
    gen = get_generator("mock")
    assert isinstance(gen, MockGroundedGenerator)
