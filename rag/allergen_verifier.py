"""Allergen Safety Verifier Module: Independent, deterministic safety verification gate.

Enforces four definitive states:
1. MATCH / REJECT:
   - Excluded allergen in declared allergens array, OR
   - Excluded allergen trigger detected via comprehensive ingredient text scan.
2. VERIFIED_PASS:
   - Populated allergen data does not contain excluded allergen, AND
   - Comprehensive ingredient scan reveals zero matching allergen triggers.
3. INSUFFICIENT_EVIDENCE:
   - Allergen information is missing (null, empty, or 'UNKNOWN').
   - NEVER assumed safe or allergen-free.
4. CONFLICTING_EVIDENCE:
   - Contradictory allergen evidence between sources/labels.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from ingestion.normalize_data import detect_allergens_in_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllergenSafetyStatus(str, Enum):
    """The four definitive allergen verification safety states."""

    VERIFIED_PASS = "VERIFIED_PASS"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class ItemVerificationResult(BaseModel):
    """Structured verification assessment for a single candidate dish."""

    dish_id: str
    dish_name: str
    status: AllergenSafetyStatus
    rejection_reason: Optional[str] = None
    warning: Optional[str] = None
    declared_allergens: List[str] = Field(default_factory=list)
    scanned_allergens: List[str] = Field(default_factory=list)
    allergen_status: str = "UNKNOWN"


class AllergenVerificationReport(BaseModel):
    """Comprehensive verification report for a batch of candidate dishes."""

    verified_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    item_results: List[ItemVerificationResult] = Field(default_factory=list)


class AllergenSafetyVerifier:
    """Independent deterministic allergen verification engine."""

    def verify_item(
        self,
        candidate: Dict[str, Any],
        excluded_allergens: List[str],
    ) -> ItemVerificationResult:
        """Deterministically assess a candidate dish against excluded allergens.

        Args:
            candidate: Candidate menu item dictionary.
            excluded_allergens: List of allergen names requested to be excluded.

        Returns:
            ItemVerificationResult containing safety status, reasons, and warnings.
        """
        meta = candidate.get("metadata", {})
        dish_id = str(candidate.get("id") or meta.get("id", "unknown"))
        dish_name = str(candidate.get("dish_name") or meta.get("dish_name", "Unknown Dish")).strip()

        declared_allergens: List[str] = list(candidate.get("allergens") or meta.get("allergens") or [])
        allergen_status: str = str(
            candidate.get("allergen_status") or meta.get("allergen_status") or "UNKNOWN"
        ).strip()

        ingredients = str(candidate.get("ingredients") or meta.get("ingredients", "")).strip()
        description = str(candidate.get("description") or meta.get("description", "")).strip()
        combined_text = f"{description} {ingredients}".strip()

        # Run context-aware regex scan across ingredients and description
        scanned_allergens: Set[str] = detect_allergens_in_text(combined_text) if combined_text else set()

        # If no allergen exclusions requested by query, verify integrity of safety metadata
        if not excluded_allergens:
            warning: Optional[str] = None
            if allergen_status == "UNKNOWN" or "unknown" in declared_allergens:
                status = AllergenSafetyStatus.INSUFFICIENT_EVIDENCE
                warning = (
                    f"Warning for '{dish_name}': ALLERGEN_INFORMATION_UNKNOWN. "
                    "Cannot guarantee allergen-free status due to unverified allergen data."
                )
            elif allergen_status == "CONFLICT" or "conflict" in declared_allergens:
                status = AllergenSafetyStatus.CONFLICTING_EVIDENCE
                warning = (
                    f"Warning for '{dish_name}': CONFLICTING_ALLERGEN_EVIDENCE. "
                    "Cannot guarantee allergen-free status due to unverified allergen data."
                )
            else:
                status = AllergenSafetyStatus.VERIFIED_PASS

            return ItemVerificationResult(
                dish_id=dish_id,
                dish_name=dish_name,
                status=status,
                warning=warning,
                declared_allergens=declared_allergens,
                scanned_allergens=sorted(list(scanned_allergens)),
                allergen_status=allergen_status,
            )

        excluded_set = {a.strip().lower() for a in excluded_allergens if a and a.strip()}

        # 1. MATCH / REJECT: Check declared allergens intersection
        declared_set = {a.strip().lower() for a in declared_allergens}
        declared_match = declared_set.intersection(excluded_set)
        if declared_match:
            return ItemVerificationResult(
                dish_id=dish_id,
                dish_name=dish_name,
                status=AllergenSafetyStatus.REJECT,
                rejection_reason=f"Dish declared excluded allergen(s): {sorted(list(declared_match))}",
                declared_allergens=declared_allergens,
                scanned_allergens=sorted(list(scanned_allergens)),
                allergen_status=allergen_status,
            )

        # 2. MATCH / REJECT: Check scanned ingredient triggers
        scanned_match = scanned_allergens.intersection(excluded_set)
        if scanned_match:
            return ItemVerificationResult(
                dish_id=dish_id,
                dish_name=dish_name,
                status=AllergenSafetyStatus.REJECT,
                rejection_reason=f"Ingredients/description contain excluded allergen trigger(s): {sorted(list(scanned_match))}",
                declared_allergens=declared_allergens,
                scanned_allergens=sorted(list(scanned_allergens)),
                allergen_status=allergen_status,
            )

        # 3. CONFLICTING_EVIDENCE: Discrepancy between labels or flagged conflict
        if allergen_status == "CONFLICT" or "conflict" in declared_set:
            warning_msg = (
                f"Warning for '{dish_name}': CONFLICTING_ALLERGEN_EVIDENCE. "
                "Cannot guarantee allergen-free status due to unverified allergen data."
            )
            return ItemVerificationResult(
                dish_id=dish_id,
                dish_name=dish_name,
                status=AllergenSafetyStatus.CONFLICTING_EVIDENCE,
                warning=warning_msg,
                declared_allergens=declared_allergens,
                scanned_allergens=sorted(list(scanned_allergens)),
                allergen_status=allergen_status,
            )

        # 4. INSUFFICIENT_EVIDENCE: Missing/null/unknown allergen record
        if (
            allergen_status == "UNKNOWN"
            or "unknown" in declared_set
            or (not declared_allergens and not ingredients and allergen_status != "KNOWN_NO_ALLERGENS")
        ):
            warning_msg = (
                f"Warning for '{dish_name}': ALLERGEN_INFORMATION_UNKNOWN. "
                "Cannot guarantee allergen-free status due to unverified allergen data."
            )
            return ItemVerificationResult(
                dish_id=dish_id,
                dish_name=dish_name,
                status=AllergenSafetyStatus.INSUFFICIENT_EVIDENCE,
                warning=warning_msg,
                declared_allergens=declared_allergens,
                scanned_allergens=sorted(list(scanned_allergens)),
                allergen_status=allergen_status,
            )

        # 5. VERIFIED_PASS: Clear evidence, zero allergen triggers
        return ItemVerificationResult(
            dish_id=dish_id,
            dish_name=dish_name,
            status=AllergenSafetyStatus.VERIFIED_PASS,
            declared_allergens=declared_allergens,
            scanned_allergens=sorted(list(scanned_allergens)),
            allergen_status=allergen_status,
        )

    def verify_candidates(
        self,
        candidates: List[Dict[str, Any]],
        excluded_allergens: List[str],
    ) -> AllergenVerificationReport:
        """Verify candidate documents and separate verified candidates from rejected items.

        Args:
            candidates: List of candidate document dictionaries.
            excluded_allergens: Excluded allergen names.

        Returns:
            AllergenVerificationReport with passed items, rejected items, warnings, and item results.
        """
        verified: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        warnings: List[str] = []
        item_results: List[ItemVerificationResult] = []

        for cand in candidates:
            res = self.verify_item(cand, excluded_allergens)
            item_results.append(res)
            cand_copy = dict(cand)

            if res.status == AllergenSafetyStatus.REJECT:
                cand_copy["rejection_reason"] = res.rejection_reason
                rejected.append(cand_copy)
                logger.info(
                    "AllergenSafetyVerifier REJECTED '%s' (%s): %s",
                    res.dish_name,
                    res.dish_id,
                    res.rejection_reason,
                )
            else:
                if res.warning:
                    cand_copy["allergen_warning"] = (
                        "ALLERGEN_INFORMATION_UNKNOWN"
                        if res.status == AllergenSafetyStatus.INSUFFICIENT_EVIDENCE
                        else "CONFLICTING_ALLERGEN_EVIDENCE"
                    )
                    if res.warning not in warnings:
                        warnings.append(res.warning)

                cand_copy["verification_status"] = res.status.value
                verified.append(cand_copy)

        return AllergenVerificationReport(
            verified_candidates=verified,
            rejected_candidates=rejected,
            warnings=warnings,
            item_results=item_results,
        )


# Global default singleton instance
_default_allergen_verifier: Optional[AllergenSafetyVerifier] = None


def get_allergen_verifier() -> AllergenSafetyVerifier:
    """Return the cached default AllergenSafetyVerifier instance."""
    global _default_allergen_verifier
    if _default_allergen_verifier is None:
        _default_allergen_verifier = AllergenSafetyVerifier()
    return _default_allergen_verifier
