"""Unit tests for field normalization: allergens, halal, spice, price, location, and search text."""

import pytest
from ingestion.normalize_data import (
    build_search_text,
    detect_allergens_in_text,
    normalize_allergens,
    normalize_halal,
    normalize_location,
    normalize_price,
    normalize_record,
    normalize_spice_level,
)


# =====================================================================
# 1. Price Normalization Tests
# =====================================================================

def test_normalize_price_numeric():
    assert normalize_price(1250) == 1250.0
    assert normalize_price(990.5) == 990.5
    assert normalize_price(0) == 0.0


def test_normalize_price_string_formats():
    assert normalize_price("PKR 1,250") == 1250.0
    assert normalize_price("Rs. 1250") == 1250.0
    assert normalize_price("Rs 950") == 950.0
    assert normalize_price("1,450.50") == 1450.5
    assert normalize_price("  850  ") == 850.0


def test_normalize_price_invalid_and_negative():
    with pytest.raises(ValueError, match="Negative price is invalid"):
        normalize_price(-100)

    with pytest.raises(ValueError, match="Negative price is invalid"):
        normalize_price("PKR -500")

    with pytest.raises(ValueError, match="Price cannot be null"):
        normalize_price(None)

    with pytest.raises(ValueError, match="Cannot parse price string"):
        normalize_price("free")


# =====================================================================
# 2. Halal Normalization Tests
# =====================================================================

def test_normalize_halal_boolean_and_truthy():
    assert normalize_halal(True) is True
    assert normalize_halal("true") is True
    assert normalize_halal("1") is True
    assert normalize_halal("yes") is True
    assert normalize_halal("halal") is True


def test_normalize_halal_falsy():
    assert normalize_halal(False) is False
    assert normalize_halal("false") is False
    assert normalize_halal("0") is False
    assert normalize_halal("no") is False
    assert normalize_halal("non-halal") is False
    assert normalize_halal("haram") is False


def test_normalize_halal_missing_and_unknown():
    # Crucial requirement: Do not assume missing Halal means True!
    assert normalize_halal(None) is None
    assert normalize_halal("") is None
    assert normalize_halal("unknown") is None
    assert normalize_halal("nan") is None


# =====================================================================
# 3. Spice Level Normalization Tests
# =====================================================================

def test_normalize_spice_level():
    assert normalize_spice_level("Mild") == "Mild"
    assert normalize_spice_level("none") == "Mild"
    assert normalize_spice_level("No Spice") == "Mild"
    assert normalize_spice_level("medium") == "Medium"
    assert normalize_spice_level("spicy") == "Spicy"
    assert normalize_spice_level("extra spicy") == "Spicy"
    assert normalize_spice_level("HOT") == "Spicy"
    assert normalize_spice_level(None) == "Unknown"
    assert normalize_spice_level("") == "Unknown"
    assert normalize_spice_level("unknown") == "Unknown"


# =====================================================================
# 4. Location Normalization Tests
# =====================================================================

def test_normalize_location():
    assert normalize_location("gulberg") == "Gulberg"
    assert normalize_location("GULBERG") == "Gulberg"
    assert normalize_location("dha") == "DHA"
    assert normalize_location("f-7") == "F-7"
    assert normalize_location("f7") == "F-7"
    assert normalize_location("saddar") == "Saddar"
    assert normalize_location("johar town") == "Johar Town"
    assert normalize_location("rawalpindi") == "Rawalpindi"
    assert normalize_location("bahria") == "Bahria Town"
    assert normalize_location(None) == "Unknown"


# =====================================================================
# 5. Allergen Normalization & Ingredient Detection Tests
# =====================================================================

def test_allergen_normalization_known_allergens():
    # Dairy synonyms
    allergens, status = normalize_allergens(["butter", "paneer", "desi ghee"])
    assert status == "KNOWN_ALLERGENS"
    assert allergens == ["dairy"]

    # Peanut synonyms
    allergens, status = normalize_allergens(["groundnut", "peanut butter"])
    assert status == "KNOWN_ALLERGENS"
    assert allergens == ["peanuts"]

    # Tree nuts synonyms
    allergens, status = normalize_allergens(["almond", "cashew", "pistachios"])
    assert status == "KNOWN_ALLERGENS"
    assert allergens == ["tree nuts"]

    # Gluten synonyms
    allergens, status = normalize_allergens(["maida", "wheat flour", "breadcrumbs"])
    assert status == "KNOWN_ALLERGENS"
    assert allergens == ["gluten"]


def test_allergen_ingredient_scanning():
    # Dish declares no allergens in metadata, but ingredients have hidden butter & cream
    allergens, status = normalize_allergens([], ingredients_text="chicken, tomatoes, heavy dairy cream, butter, spices")
    assert status == "KNOWN_ALLERGENS"
    assert "dairy" in allergens

    # Prawns in ingredients
    allergens, status = normalize_allergens([], ingredients_text="jumbo tiger prawns, lemon juice, garlic")
    assert status == "KNOWN_ALLERGENS"
    assert "shellfish" in allergens


def test_allergen_false_positive_prevention():
    # Coconut milk should NOT trigger dairy
    allergens, status = normalize_allergens([], ingredients_text="tiger prawns, coconut milk, red chili paste")
    assert "dairy" not in allergens
    assert "shellfish" in allergens

    # Chickpea flour (besan) should NOT trigger gluten
    allergens, status = normalize_allergens([], ingredients_text="rahu fish, chickpea flour, carom seeds, mustard oil")
    assert "gluten" not in allergens

    # Peanut butter should trigger peanuts, NOT dairy
    allergens, status = normalize_allergens([], ingredients_text="chicken skewers, peanut butter, soy sauce")
    assert "peanuts" in allergens
    assert "dairy" not in allergens


def test_allergen_normalization_known_no_allergens():
    # Safe dish with explicit empty list and clean ingredients
    allergens, status = normalize_allergens([], ingredients_text="chicken breast, olive oil, lemon, rice")
    assert status == "KNOWN_NO_ALLERGENS"
    assert allergens == []


def test_allergen_missing_and_unknown_handling():
    # Crucial safety requirement: Never convert unknown to [] or assume safe!
    allergens, status = normalize_allergens(None)
    assert status == "UNKNOWN"
    assert allergens == ["unknown"]

    allergens, status = normalize_allergens("")
    assert status == "UNKNOWN"
    assert allergens == ["unknown"]

    allergens, status = normalize_allergens("unknown")
    assert status == "UNKNOWN"
    assert allergens == ["unknown"]

    allergens, status = normalize_allergens(["unknown"])
    assert status == "UNKNOWN"
    assert allergens == ["unknown"]


def test_allergen_conflict_handling():
    allergens, status = normalize_allergens(["conflict"])
    assert status == "CONFLICT"
    assert allergens == ["conflict"]


# =====================================================================
# 6. Searchable Text Generation Tests
# =====================================================================

def test_build_search_text():
    rec = {
        "restaurant_name": "Monal Express",
        "dish_name": "Grilled Chicken Breast Bowl",
        "description": "Char-grilled chicken breast served with steamed brown rice",
        "ingredients": "chicken breast, brown rice, olive oil, lemon",
        "category": "Chicken",
        "cuisine": "Pakistani",
        "spice_level": "Mild",
    }
    search_text = build_search_text(rec)
    assert "Monal Express" in search_text
    assert "Grilled Chicken Breast Bowl" in search_text
    assert "chicken breast" in search_text
    assert "Mild" in search_text
    # Ensure no double spaces
    assert "  " not in search_text


# =====================================================================
# 7. Complete Record Normalization
# =====================================================================

def test_normalize_record_full():
    raw = {
        "id": "dish_test_01",
        "restaurant_name": "Test Café",
        "dish_name": "Butter Chicken with Naan",
        "description": "Rich chicken in butter gravy",
        "ingredients": "chicken, butter, dairy cream, maida flour",
        "price_pkr": "PKR 1,350",
        "location": "gulberg",
        "halal": "yes",
        "spice_level": "medium",
        "allergens": "['milk']",
        "category": "Chicken",
        "protein_g": "32",
        "availability": "true",
        "cuisine": "pakistani",
    }
    item = normalize_record(raw)
    assert item.id == "dish_test_01"
    assert item.price_pkr == 1350.0
    assert item.location == "Gulberg"
    assert item.halal is True
    assert item.spice_level == "Medium"
    assert item.allergen_status == "KNOWN_ALLERGENS"
    assert "dairy" in item.allergens
    assert "gluten" in item.allergens  # Detected from maida flour
    assert item.protein_g == 32.0
    assert len(item.search_text) > 20
