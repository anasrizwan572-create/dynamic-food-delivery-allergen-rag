from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.query_parser import HardConstraints


@dataclass
class FilterResult:
    """Result of deterministic hard-constraint filtering."""

    passed_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "SUPPORTED"
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary_message: str = ""


def detect_allergens_in_text(text: str) -> Set[str]:
    """Detect common allergens mentioned in ingredient text."""

    text = str(text or "").lower()
    found: Set[str] = set()

    allergen_words = {
        "dairy": ["milk", "dairy", "cheese", "butter", "cream", "yogurt"],
        "gluten": ["gluten", "wheat", "flour", "bread", "bun"],
        "eggs": ["egg", "eggs", "mayonnaise"],
        "peanuts": ["peanut", "peanuts"],
        "tree nuts": [
            "almond",
            "almonds",
            "cashew",
            "cashews",
            "walnut",
            "walnuts",
            "pistachio",
            "pistachios",
        ],
        "soy": ["soy", "soya"],
        "shellfish": ["shrimp", "prawn", "prawns", "crab", "lobster"],
        "fish": ["fish", "tuna", "salmon"],
    }

    for allergen, words in allergen_words.items():
        if any(word in text for word in words):
            found.add(allergen)

    return found


def evaluate_dish_constraints(
    candidate: Dict[str, Any],
    constraints: HardConstraints,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check whether one dish satisfies all hard constraints.

    Returns:
        (is_valid, rejection_reason, allergen_warning)
    """

    meta = candidate.get("metadata", {})

    dish_name = str(
        candidate.get("dish_name") or meta.get("dish_name", "")
    ).strip()

    price = float(
        candidate.get("price_pkr") or meta.get("price_pkr", 0.0)
    )

    location = str(
        candidate.get("location") or meta.get("location", "")
    ).strip()

    halal = candidate.get("halal", meta.get("halal"))

    allergens = set(candidate.get("allergens", []) or [])

    allergen_status = str(
        candidate.get("allergen_status")
        or meta.get("allergen_status", "")
    ).strip()

    ingredients = str(
        meta.get("ingredients", "")
    ).strip()

    # ---------------------------------------------------------
    # 1. Maximum price
    # ---------------------------------------------------------
    if constraints.max_price_pkr is not None:
        if price > constraints.max_price_pkr:
            return (
                False,
                f"Price PKR {price:.1f} exceeds maximum budget "
                f"of PKR {constraints.max_price_pkr:.1f}",
                None,
            )

    # ---------------------------------------------------------
    # 2. Minimum price
    # ---------------------------------------------------------
    if constraints.min_price_pkr is not None:
        if price < constraints.min_price_pkr:
            return (
                False,
                f"Price PKR {price:.1f} is below minimum requested "
                f"price PKR {constraints.min_price_pkr:.1f}",
                None,
            )

    # ---------------------------------------------------------
    # 3. Location
    # ---------------------------------------------------------
    if constraints.location:
        requested_location = constraints.location.strip().lower()
        candidate_location = location.lower()

        if candidate_location != requested_location:
            return (
                False,
                f"Location '{location}' does not match requested "
                f"location '{constraints.location}'",
                None,
            )

    # ---------------------------------------------------------
    # 4. Category / dish type / protein
    # ---------------------------------------------------------
    requested_categories = getattr(constraints, "categories", []) or []

    dish_name_lower = dish_name.lower()

    for requested_category in requested_categories:
        category = requested_category.strip().lower()

        if category == "burger":
            if "burger" not in dish_name_lower:
                return False, "Dish is not a burger", None

        elif category == "beef":
            if "beef" not in dish_name_lower:
                return False, "Dish is not a beef dish", None

        elif category == "chicken":
            if "chicken" not in dish_name_lower:
                return False, "Dish is not a chicken dish", None

        elif category == "mutton":
            if "mutton" not in dish_name_lower:
                return False, "Dish is not a mutton dish", None

        elif category == "fish":
            if "fish" not in dish_name_lower:
                return False, "Dish is not a fish dish", None

        elif category == "prawns":
            if "prawn" not in dish_name_lower and "shrimp" not in dish_name_lower:
                return False, "Dish is not a prawns/shrimp dish", None

        elif category == "seafood":
            seafood_words = [
                "seafood",
                "fish",
                "prawn",
                "prawns",
                "shrimp",
                "crab",
                "lobster",
            ]

            if not any(word in dish_name_lower for word in seafood_words):
                return False, "Dish is not a seafood dish", None

    # ---------------------------------------------------------
    # 5. Halal
    # ---------------------------------------------------------
    if constraints.halal is True:
        if halal is False:
            return False, "Dish is non-Halal", None

        if halal is None:
            halal_raw = str(meta.get("halal_raw", "")).lower()

            if halal_raw == "unknown":
                return False, "Halal status is unverified", None

    elif constraints.halal is False:
        if halal is True:
            return False, "Dish is Halal but non-Halal was requested", None

    # ---------------------------------------------------------
    # 6. Availability
    # ---------------------------------------------------------
    if constraints.availability is True:
        availability = candidate.get(
            "availability",
            meta.get("availability", True),
        )

        if availability is False:
            return False, "Dish is currently out of stock", None

    # ---------------------------------------------------------
    # 7. Excluded allergens
    # ---------------------------------------------------------
    warning: Optional[str] = None

    excluded = set(constraints.excluded_allergens or [])

    if excluded:

        # Normalize direct allergen metadata
        normalized_allergens = {
            str(item).strip().lower()
            for item in allergens
        }

        direct_match = normalized_allergens.intersection(excluded)

        if direct_match:
            return (
                False,
                f"Dish contains excluded allergen(s): "
                f"{sorted(direct_match)}",
                None,
            )

        # Check ingredients
        if ingredients:
            ingredient_allergens = detect_allergens_in_text(ingredients)

            ingredient_match = ingredient_allergens.intersection(excluded)

            if ingredient_match:
                return (
                    False,
                    f"Ingredients contain excluded allergen(s): "
                    f"{sorted(ingredient_match)}",
                    None,
                )

        # Unknown allergen information
        if (
            allergen_status.upper() == "UNKNOWN"
            or "unknown" in normalized_allergens
        ):
            warning = "ALLERGEN_INFORMATION_UNKNOWN"

        # Conflicting allergen information
        if (
            allergen_status.upper() == "CONFLICT"
            or "conflict" in normalized_allergens
        ):
            warning = "CONFLICTING_ALLERGEN_EVIDENCE"

    return True, None, warning


def apply_metadata_filters(
    candidates: List[Dict[str, Any]],
    constraints: HardConstraints,
) -> FilterResult:
    """
    Apply all hard constraints deterministically.

    Only candidates that satisfy EVERY requested hard constraint
    are returned as passed candidates.
    """

    passed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: List[str] = []
    warnings: List[str] = []

    for candidate in candidates:

        is_valid, reason, warning = evaluate_dish_constraints(
            candidate,
            constraints,
        )

        if is_valid:
            candidate_copy = dict(candidate)

            if warning:
                candidate_copy["filter_warning"] = warning

            passed.append(candidate_copy)

        else:
            rejected.append(candidate)

            if reason:
                rejection_reasons.append(reason)

    if passed:
        status = "SUPPORTED"
    else:
        status = "NO_MATCHING_ITEMS"

    return FilterResult(
    passed_candidates=passed,
    rejected_candidates=rejected,
    status=status,
    rejection_reasons=rejection_reasons,
    warnings=warnings,
    summary_message=(
        f"{len(passed)} menu items satisfied all requested hard constraints."
        if passed
        else "No menu items satisfied all requested hard constraints."
    ),
)