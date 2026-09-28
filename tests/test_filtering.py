"""Unit tests for query parsing, constraint extraction, and deterministic metadata filtering."""

import pytest
from rag.metadata_filter import apply_metadata_filters, evaluate_dish_constraints
from rag.pipeline import answer_query
from rag.query_parser import (
    HardConstraints,
    SoftPreferences,
    extract_allergen_exclusions,
    extract_halal,
    extract_location,
    extract_price_constraints,
    parse_query,
)

SAMPLE_CANDIDATES = [
    {
        "id": "dish_001",
        "dish_name": "Grilled Chicken Breast Bowl",
        "restaurant_name": "Monal Express",
        "price_pkr": 1150.0,
        "location": "Gulberg",
        "halal": True,
        "spice_level": "Mild",
        "allergens": [],
        "allergen_status": "KNOWN_NO_ALLERGENS",
        "availability": True,
        "metadata": {
            "ingredients": "chicken breast, brown rice, olive oil, lemon",
            "category": "Chicken",
        },
    },
    {
        "id": "dish_002",
        "dish_name": "Murgh Makhani Butter Chicken",
        "restaurant_name": "Haveli Restaurant",
        "price_pkr": 1450.0,
        "location": "Gulberg",
        "halal": True,
        "spice_level": "Medium",
        "allergens": ["dairy", "tree nuts"],
        "allergen_status": "KNOWN_ALLERGENS",
        "availability": True,
        "metadata": {
            "ingredients": "chicken, butter, dairy cream, cashew paste",
            "category": "Chicken",
        },
    },
    {
        "id": "dish_005",
        "dish_name": "Premium Wagyu Beef Steak",
        "restaurant_name": "Salt & Pepper Village",
        "price_pkr": 3200.0,
        "location": "Gulberg",
        "halal": True,
        "spice_level": "Mild",
        "allergens": ["dairy"],
        "allergen_status": "KNOWN_ALLERGENS",
        "availability": True,
        "metadata": {
            "ingredients": "wagyu beef, garlic butter, rosemary",
            "category": "Beef",
        },
    },
    {
        "id": "dish_006",
        "dish_name": "Imported Pork Pepperoni Pizza",
        "restaurant_name": "Expat Club Bistro",
        "price_pkr": 2400.0,
        "location": "DHA",
        "halal": False,
        "spice_level": "Mild",
        "allergens": ["gluten", "dairy"],
        "allergen_status": "KNOWN_ALLERGENS",
        "availability": True,
        "metadata": {
            "ingredients": "wheat dough, pork pepperoni, mozzarella",
            "category": "Fast Food",
        },
    },
    {
        "id": "dish_009",
        "dish_name": "Mystery Special Daily Daal",
        "restaurant_name": "Anarkali Dhaba",
        "price_pkr": 400.0,
        "location": "Gulberg",
        "halal": True,
        "spice_level": "Medium",
        "allergens": ["unknown"],
        "allergen_status": "UNKNOWN",
        "availability": True,
        "metadata": {
            "ingredients": "yellow lentils, unverified oil, garlic",
            "category": "Vegetarian",
        },
    },
    {
        "id": "dish_011",
        "dish_name": "Grilled Jumbo Tiger Prawns",
        "restaurant_name": "Lal Qila",
        "price_pkr": 2450.0,
        "location": "Saddar",
        "halal": True,
        "spice_level": "Medium",
        "allergens": ["shellfish"],
        "allergen_status": "KNOWN_ALLERGENS",
        "availability": True,
        "metadata": {
            "ingredients": "tiger prawns, garlic, lemon, chili flakes",
            "category": "Seafood",
        },
    },
]


# =====================================================================
# 1. Price Constraint Parsing & Mathematical Filtering
# =====================================================================

def test_under_price_parsing_and_filtering():
    parsed = parse_query("Find meals under PKR 1500")
    assert parsed.hard_constraints.max_price_pkr == 1500.0

    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    passed_ids = [c["id"] for c in res.passed_candidates]

    # Required invariant: dish with 2450 or 3200 MUST NOT appear
    assert "dish_011" not in passed_ids  # PKR 2450
    assert "dish_005" not in passed_ids  # PKR 3200
    assert "dish_001" in passed_ids      # PKR 1150
    assert "dish_002" in passed_ids      # PKR 1450


def test_above_price_parsing():
    max_p, min_p, _ = extract_price_constraints("meals above PKR 1000")
    assert min_p == 1000.0
    assert max_p is None


def test_between_price_parsing():
    max_p, min_p, _ = extract_price_constraints("meals between 500 and 1500")
    assert min_p == 500.0
    assert max_p == 1500.0


# =====================================================================
# 2. Halal Constraint Parsing & Filtering
# =====================================================================

def test_halal_constraint_filtering():
    parsed = parse_query("Halal food in DHA")
    assert parsed.hard_constraints.halal is True

    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    passed_ids = [c["id"] for c in res.passed_candidates]

    # Non-Halal dish_006 must be rejected
    assert "dish_006" not in passed_ids


# =====================================================================
# 3. Location Constraint Parsing & Capitalization
# =====================================================================

def test_location_normalization_and_filtering():
    for loc_query in ["meals in gulberg", "meals in GULBERG", "meals in Gulberg"]:
        parsed = parse_query(loc_query)
        assert parsed.hard_constraints.location == "Gulberg"
        res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
        for cand in res.passed_candidates:
            assert cand["location"] == "Gulberg"


# =====================================================================
# 4. Allergen Exclusion & Synonym Handling
# =====================================================================

def test_allergen_exclusion_dairy():
    # Required safety test: A dish containing dairy MUST NOT pass
    parsed = parse_query("food without dairy")
    assert "dairy" in parsed.hard_constraints.excluded_allergens

    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    passed_ids = [c["id"] for c in res.passed_candidates]

    assert "dish_002" not in passed_ids  # Contains dairy & nuts
    assert "dish_005" not in passed_ids  # Contains dairy (garlic butter)
    assert "dish_006" not in passed_ids  # Contains dairy & gluten
    assert "dish_001" in passed_ids      # Clean chicken bowl


def test_multiple_allergen_exclusions():
    parsed = parse_query("food without dairy or peanuts and no shellfish")
    assert set(parsed.hard_constraints.excluded_allergens) == {"dairy", "peanuts", "shellfish"}

    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    passed_ids = [c["id"] for c in res.passed_candidates]

    assert "dish_011" not in passed_ids  # Shellfish prawns
    assert "dish_002" not in passed_ids  # Dairy


def test_allergen_synonym_handling_in_query():
    # "without butter or paneer" -> normalized to dairy
    parsed = parse_query("curry without butter or paneer")
    assert "dairy" in parsed.hard_constraints.excluded_allergens


# =====================================================================
# 5. Missing Allergen Information Handling (UNKNOWN State)
# =====================================================================

def test_missing_allergen_information_warning():
    # When user excludes an allergen and dish has UNKNOWN status, it must have an explicit warning
    constraints = HardConstraints(excluded_allergens=["dairy"])
    res = apply_metadata_filters([SAMPLE_CANDIDATES[4]], constraints)  # dish_009 Daily Daal (UNKNOWN)

    assert len(res.passed_candidates) == 1
    passed_item = res.passed_candidates[0]
    assert passed_item["allergen_warning"] == "ALLERGEN_INFORMATION_UNKNOWN"
    assert any("ALLERGEN_INFORMATION_UNKNOWN" in w for w in res.warnings)


# =====================================================================
# 6. Combined Constraints and NO_MATCHING_ITEMS State
# =====================================================================

def test_combined_constraints():
    parsed = parse_query("Find Halal chicken under PKR 1500 without dairy in Gulberg")
    assert parsed.hard_constraints.max_price_pkr == 1500.0
    assert parsed.hard_constraints.halal is True
    assert parsed.hard_constraints.location == "Gulberg"
    assert "dairy" in parsed.hard_constraints.excluded_allergens
    assert parsed.soft_preferences.category == "Chicken"

    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    assert res.status == "SUPPORTED"
    passed_ids = [c["id"] for c in res.passed_candidates]
    assert "dish_001" in passed_ids
    assert "dish_002" not in passed_ids  # Rejected: contains dairy
    assert "dish_005" not in passed_ids  # Rejected: over budget & contains dairy
    assert "dish_006" not in passed_ids  # Rejected: DHA & non-Halal
    assert "dish_011" not in passed_ids  # Rejected: Saddar & over budget


def test_no_matching_items():
    # Extremely low budget: PKR 100
    parsed = parse_query("meal under PKR 100 in Gulberg")
    res = apply_metadata_filters(SAMPLE_CANDIDATES, parsed.hard_constraints)
    assert res.status == "NO_MATCHING_ITEMS"
    assert len(res.passed_candidates) == 0


# =====================================================================
# 7. Hard vs Soft Constraints Separation
# =====================================================================

def test_hard_vs_soft_constraints_separation():
    parsed = parse_query("Find high-protein Halal chicken under 1500 in Gulberg")
    # Hard
    assert parsed.hard_constraints.max_price_pkr == 1500.0
    assert parsed.hard_constraints.halal is True
    assert parsed.hard_constraints.location == "Gulberg"
    # Soft
    assert parsed.soft_preferences.protein_preference == "high"
    assert parsed.soft_preferences.category == "Chicken"


# =====================================================================
# 8. End-to-End Pipeline Constraint Invariants
# =====================================================================

def test_pipeline_under_1500_excludes_2450():
    res = answer_query("meal under PKR 1500", top_k=5)
    for doc in res["retrieved_documents"]:
        assert doc["price_pkr"] <= 1500.0, f"Dish {doc['id']} price {doc['price_pkr']} violated max budget!"


def test_pipeline_no_matching_items_response():
    res = answer_query("Find a meal under PKR 20 in Gulberg")
    assert res["status"] == "NO_MATCHING_ITEMS"
    assert res["sources"] == []
    assert "No menu item satisfies all of the requested constraints" in res["answer"]
