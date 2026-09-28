"""Deterministic metadata filtering module enforcing hard mathematical and categorical constraints."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from ingestion.normalize_data import detect_allergens_in_text
from rag.query_parser import HardConstraints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of deterministic constraint filtering."""

    status: str = Field(..., description="SUPPORTED | NO_MATCHING_ITEMS")
    passed_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary_message: str = ""


def evaluate_dish_constraints(
    candidate: Dict[str, Any],
    constraints: HardConstraints,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluate whether a single candidate dish satisfies all hard constraints.

    Returns:
        (is_valid, rejection_reason, allergen_warning)
    """
    meta = candidate.get("metadata", {})
    price = float(candidate.get("price_pkr", meta.get("price_pkr", 0.0)))
    location = str(candidate.get("location") or meta.get("location", "")).strip()
    halal = candidate.get("halal", meta.get("halal"))
    allergens = set(candidate.get("allergens", []))
    allergen_status = str(candidate.get("allergen_status") or meta.get("allergen_status", "")).strip()
    ingredients = str(meta.get("ingredients", "")).strip()

    # 1. Maximum Price Check (Mathematical inequality)
    if constraints.max_price_pkr is not None:
        if price > constraints.max_price_pkr:
            return False, f"Price PKR {price:.1f} exceeds maximum budget of PKR {constraints.max_price_pkr:.1f}", None

    # 2. Minimum Price Check
    if constraints.min_price_pkr is not None:
        if price < constraints.min_price_pkr:
            return False, f"Price PKR {price:.1f} is below minimum requested price PKR {constraints.min_price_pkr:.1f}", None

    # 3. Location Check (Exact normalized match)
    if constraints.location:
        req_loc = constraints.location.strip().lower()
        cand_loc = location.lower()
        if cand_loc != req_loc:
            return False, f"Location '{location}' does not match requested location '{constraints.location}'", None

    # 4. Halal Check
    if constraints.halal is True:
        if halal is False:
            return False, "Dish is non-Halal", None
        if halal is None and str(meta.get("halal_raw", "")).lower() == "unknown":
            return False, "Halal status is unverified", None
    elif constraints.halal is False:
        if halal is True:
            return False, "Dish is Halal (user requested non-Halal)", None

    # 5. Availability Check
    if constraints.availability is True:
        avail = candidate.get("availability", meta.get("availability", True))
        if avail is False:
            return False, "Dish is currently out of stock", None

    # 6. Excluded Allergens Check (Hard constraint)
    warning: Optional[str] = None
    if constraints.excluded_allergens:
        excluded_set = set(constraints.excluded_allergens)

        # Direct metadata allergen intersection
        direct_intersect = allergens.intersection(excluded_set)
        if direct_intersect:
            return False, f"Dish contains excluded allergen(s): {sorted(list(direct_intersect))}", None

        # Hidden ingredients scan intersection
        if ingredients:
            ing_allergens = detect_allergens_in_text(ingredients)
            ing_intersect = ing_allergens.intersection(excluded_set)
            if ing_intersect:
                return False, f"Ingredients contain excluded allergen(s): {sorted(list(ing_intersect))}", None

        # UNKNOWN allergen handling
        if allergen_status == "UNKNOWN" or "unknown" in allergens:
            warning = "ALLERGEN_INFORMATION_UNKNOWN"

        # CONFLICT allergen handling
        if allergen_status == "CONFLICT" or "conflict" in allergens:
            warning = "CONFLICTING_ALLERGEN_EVIDENCE"

    return True, None, warning


def apply_metadata_filters(
    candidates: List[Dict[str, Any]],
    constraints: HardConstraints,
) -> FilterResult:
    """Filter a candidate document list using deterministic hard metadata constraints.

    Args:
        candidates: List of candidate document dictionaries.
        constraints: Parsed HardConstraints.

    Returns:
        FilterResult containing passed candidates, rejected candidates, and status.
    """
    passed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for cand in candidates:
        is_valid, reject_reason, allergen_warn = evaluate_dish_constraints(cand, constraints)
        cand_copy = dict(cand)

        if is_valid:
            if allergen_warn:
                cand_copy["allergen_warning"] = allergen_warn
                warning_msg = (
                    f"Warning for '{cand.get('dish_name')}': {allergen_warn}. "
                    "Cannot guarantee allergen-free status due to unverified allergen data."
                )
                if warning_msg not in warnings:
                    warnings.append(warning_msg)
            passed.append(cand_copy)
        else:
            cand_copy["rejection_reason"] = reject_reason
            rejected.append(cand_copy)

    if not passed:
        summary = "No menu item satisfies all of the requested hard constraints."
        logger.info("Metadata filtering: 0 candidates passed out of %d. Status: NO_MATCHING_ITEMS", len(candidates))
        return FilterResult(
            status="NO_MATCHING_ITEMS",
            passed_candidates=[],
            rejected_candidates=rejected,
            warnings=warnings,
            summary_message=summary,
        )

    logger.info("Metadata filtering: %d passed, %d rejected out of %d candidates.", len(passed), len(rejected), len(candidates))
    return FilterResult(
        status="SUPPORTED",
        passed_candidates=passed,
        rejected_candidates=rejected,
        warnings=warnings,
        summary_message=f"{len(passed)} menu items satisfied all requested hard constraints.",
    )
