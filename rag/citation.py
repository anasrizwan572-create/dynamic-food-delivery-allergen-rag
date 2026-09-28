"""Citation Engine and Grounded Synthesis Module.

Enforces:
1. Citation extraction: maps inline source citations [dish_xxx] to retrieved context evidence.
2. Hallucination prevention: rejects or flags unsupported dish IDs, phantom prices, and fabricated claims.
3. Allergen safety grounding: ensures UNKNOWN and CONFLICT allergen evidence is NEVER synthesized
   as "allergen-free" or "safe", preserving exact warnings.
4. Structured citations: produces rich Citation models linking factual claims directly to menu source data.
5. Evidence sufficiency evaluation: accurately classifies responses into typed states:
   SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, NO_MATCHING_ITEMS, UNSUPPORTED.
"""

from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Non-citation bracketed tokens to ignore during citation extraction
NON_CITATION_TOKENS = {
    "document",
    "context",
    "user question",
    "grounded answer",
    "supported",
    "partially_supported",
    "insufficient_evidence",
    "conflicting_evidence",
    "no_matching_items",
    "unsupported",
    "none listed",
    "verified",
}

# Regex detecting safety claims that require verified evidence
UNSUPPORTED_SAFETY_PATTERNS = [
    re.compile(r"\b(allergen[- ]free|safe from (?:all )?allergens|guaranteed safe|zero allergens|contains no allergens|100% safe)\b", re.I),
]


class GroundingStatus(str, Enum):
    """Grounded synthesis status reflecting evidence sufficiency and claim validity."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NO_MATCHING_ITEMS = "NO_MATCHING_ITEMS"
    UNSUPPORTED = "UNSUPPORTED"


class Citation(BaseModel):
    """Structured citation metadata linking a grounded claim to a retrieved menu document."""

    source_id: str = Field(..., description="Stable dish identifier e.g. dish_001")
    dish_name: str
    restaurant_name: str
    price_pkr: float
    location: str
    halal: bool = True
    spice_level: str = "Medium"
    allergens: List[str] = Field(default_factory=list)
    allergen_status: str = "UNKNOWN"
    source_menu: str = "Standard Menu"
    match_reason: str = ""
    reranker_score: Optional[float] = None
    verification_status: Optional[str] = None
    warning: Optional[str] = None


class GroundingVerificationReport(BaseModel):
    """Comprehensive grounding verification assessing answer validity against evidence."""

    is_grounded: bool = True
    status: GroundingStatus = GroundingStatus.SUPPORTED
    cited_ids: List[str] = Field(default_factory=list)
    valid_citations: List[Citation] = Field(default_factory=list)
    hallucinated_citations: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    verification_notes: List[str] = Field(default_factory=list)


class CitationEngine:
    """Citation Engine and Grounded Synthesis Verifier."""

    def extract_citations(self, text: str) -> List[str]:
        """Extract unique ordered source bracket citations (e.g. [dish_001]) from text.

        Args:
            text: Natural language response text containing bracket citations.

        Returns:
            List of unique source IDs in order of first appearance.
        """
        raw_matches = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", text)
        extracted: List[str] = []
        for match in raw_matches:
            match_clean = match.strip()
            match_lower = match_clean.lower()
            if (
                match_lower in NON_CITATION_TOKENS
                or match_lower.startswith("document")
                or match_clean.isdigit()
            ):
                continue
            if match_clean not in extracted:
                extracted.append(match_clean)
        return extracted

    def build_citation(
        self,
        doc: Dict[str, Any],
        match_reason: str = "",
        parsed_constraints: Optional[Dict[str, Any]] = None,
    ) -> Citation:
        """Construct a structured Citation model from a retrieved document dict."""
        meta = doc.get("metadata", {})
        source_id = str(doc.get("id") or meta.get("id", "unknown"))
        dish_name = str(doc.get("dish_name") or meta.get("dish_name", "Unknown Dish")).strip()
        restaurant_name = str(
            doc.get("restaurant_name") or meta.get("restaurant_name", "Unknown Restaurant")
        ).strip()
        price_val = doc.get("price_pkr", meta.get("price_pkr", 0.0))
        try:
            price_pkr = float(price_val)
        except (ValueError, TypeError):
            price_pkr = 0.0

        location = str(doc.get("location") or meta.get("location", "Unknown Location")).strip()
        halal_val = doc.get("halal") if doc.get("halal") is not None else meta.get("halal", True)
        halal = bool(halal_val)

        spice_level = str(doc.get("spice_level") or meta.get("spice_level", "Medium")).strip()
        allergens = list(doc.get("allergens") or meta.get("allergens") or [])
        allergen_status = str(
            doc.get("allergen_status") or meta.get("allergen_status", "UNKNOWN")
        ).strip()
        source_menu = str(
            meta.get("source") or doc.get("source", f"{restaurant_name} Menu")
        ).strip()

        reason = match_reason or self.build_match_reason(doc, parsed_constraints)
        reranker_score = doc.get("reranker_score")
        verification_status = doc.get("verification_status")
        warning = doc.get("allergen_warning")

        return Citation(
            source_id=source_id,
            dish_name=dish_name,
            restaurant_name=restaurant_name,
            price_pkr=price_pkr,
            location=location,
            halal=halal,
            spice_level=spice_level,
            allergens=allergens,
            allergen_status=allergen_status,
            source_menu=source_menu,
            match_reason=reason,
            reranker_score=reranker_score,
            verification_status=verification_status,
            warning=warning,
        )

    def build_match_reason(
        self,
        doc: Dict[str, Any],
        parsed_constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Deterministically formulate a factual explanation for why an item matches."""
        meta = doc.get("metadata", {})
        price = doc.get("price_pkr", meta.get("price_pkr", 0.0))
        loc = doc.get("location") or meta.get("location", "")
        halal = doc.get("halal") if doc.get("halal") is not None else meta.get("halal", True)
        allergen_status = doc.get("allergen_status") or meta.get("allergen_status", "UNKNOWN")
        allergens = doc.get("allergens", [])

        reasons = []
        if parsed_constraints and parsed_constraints.get("max_price_pkr"):
            reasons.append(f"Under budget (PKR {price})")
        else:
            reasons.append(f"PKR {price}")

        if halal:
            reasons.append("Halal certified")

        if loc:
            reasons.append(f"available in {loc}")

        if allergen_status == "KNOWN_NO_ALLERGENS":
            reasons.append("verified zero listed allergens")
        elif allergen_status == "UNKNOWN":
            reasons.append("allergen information unverified (caution advised)")
        elif allergen_status == "CONFLICT":
            reasons.append("conflicting allergen evidence noted")
        elif allergens:
            reasons.append(f"allergens noted: {', '.join(allergens)}")

        return ", ".join(reasons) + "."

    def verify_grounding(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        query: str = "",
        parsed_constraints: Optional[Dict[str, Any]] = None,
    ) -> GroundingVerificationReport:
        """Verify that the generated answer is strictly grounded in the retrieved documents.

        Performs:
        1. Citation identification and validation against retrieved document IDs.
        2. Detection of hallucinated citation IDs.
        3. Prevention of unsupported safety claims (especially on UNKNOWN and CONFLICT records).
        4. Detection of unsupported allergen-free claims on dishes declaring allergens.
        5. Verification of cited prices against document records.
        6. Classification of overall GroundingStatus.

        Args:
            answer: Generated response string.
            retrieved_docs: List of candidate menu item documents.
            query: Original user query.
            parsed_constraints: Parsed constraints dictionary.

        Returns:
            GroundingVerificationReport detailing grounding validity, citations, and notes.
        """
        cited_ids = self.extract_citations(answer)
        doc_map = {str(d.get("id") or d.get("metadata", {}).get("id")): d for d in retrieved_docs}

        valid_citations: List[Citation] = []
        hallucinated_citations: List[str] = []
        unsupported_claims: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []

        # 1. Validate extracted citations against retrieved documents
        for cid in cited_ids:
            if cid in doc_map:
                cit = self.build_citation(doc_map[cid], parsed_constraints=parsed_constraints)
                valid_citations.append(cit)
                if cit.warning and cit.warning not in warnings:
                    warnings.append(cit.warning)
            else:
                hallucinated_citations.append(cid)
                unsupported_claims.append(
                    f"Hallucinated citation: [{cid}] was cited but does not exist in retrieved evidence."
                )

        # 2. Check for missing documents / insufficient evidence
        if not retrieved_docs:
            is_grounded = "insufficient evidence" in answer.lower() or "no menu item" in answer.lower()
            return GroundingVerificationReport(
                is_grounded=is_grounded,
                status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                cited_ids=cited_ids,
                valid_citations=[],
                hallucinated_citations=hallucinated_citations,
                unsupported_claims=unsupported_claims,
                warnings=warnings,
                evidence_count=0,
                verification_notes=["No retrieved documents available for query."],
            )

        # If answer cited nothing, map all retrieved docs to citations for completeness
        if not cited_ids and retrieved_docs:
            for doc in retrieved_docs:
                cit = self.build_citation(doc, parsed_constraints=parsed_constraints)
                valid_citations.append(cit)
                if cit.warning and cit.warning not in warnings:
                    warnings.append(cit.warning)

        # 3. Check for unsupported safety claims in answer text
        answer_lower = answer.lower()

        # Check for unverified allergen claims across cited documents
        for cit in valid_citations:
            # Check UNKNOWN allergen evidence
            if cit.allergen_status == "UNKNOWN" or cit.verification_status == "INSUFFICIENT_EVIDENCE":
                # Ensure canonical warning is retained
                exact_unknown_warning = (
                    f"Warning for '{cit.dish_name}': ALLERGEN_INFORMATION_UNKNOWN. "
                    "Cannot guarantee allergen-free status due to unverified allergen data."
                )
                if exact_unknown_warning not in warnings:
                    warnings.append(exact_unknown_warning)

                # Check if answer falsely claims this dish is safe / allergen-free
                for pattern in UNSUPPORTED_SAFETY_PATTERNS:
                    if pattern.search(answer):
                        unsupported_claims.append(
                            f"Unsupported claim: Answer claims allergen-free safety for '{cit.dish_name}' [{cit.source_id}], "
                            "but its allergen status is UNKNOWN (insufficient evidence)."
                        )
                        break

            # Check CONFLICT allergen evidence
            elif cit.allergen_status == "CONFLICT" or cit.verification_status == "CONFLICTING_EVIDENCE":
                # Ensure canonical warning is retained
                exact_conflict_warning = (
                    f"Warning for '{cit.dish_name}': CONFLICTING_ALLERGEN_EVIDENCE. "
                    "Cannot guarantee allergen-free status due to unverified allergen data."
                )
                if exact_conflict_warning not in warnings:
                    warnings.append(exact_conflict_warning)

                # Check if answer falsely claims this dish is safe / allergen-free
                for pattern in UNSUPPORTED_SAFETY_PATTERNS:
                    if pattern.search(answer):
                        unsupported_claims.append(
                            f"Unsupported claim: Answer claims allergen-free safety for '{cit.dish_name}' [{cit.source_id}], "
                            "but its allergen evidence is CONFLICTING."
                        )
                        break

            # Check if answer claims dish is allergen-free for an allergen it declares
            for alg in cit.allergens:
                alg_lower = alg.strip().lower()
                if not alg_lower or alg_lower in ["none", "unknown", "conflict"]:
                    continue
                false_free_pattern = re.compile(rf"\b{re.escape(alg_lower)}[- ]free\b", re.I)
                if false_free_pattern.search(answer):
                    # Check if the dish explicitly contains it
                    unsupported_claims.append(
                        f"Unsupported claim: Answer claims '{cit.dish_name}' [{cit.source_id}] is {alg_lower}-free, "
                        f"but dish declares '{alg_lower}' in its allergen profile."
                    )

        # 4. Check for price fabrication in answer text
        for cit in valid_citations:
            cid = cit.source_id
            # Pattern: PKR <digits> anywhere in vicinity of citation
            price_matches = re.findall(rf"PKR\s*([0-9]+(?:\.[0-9]+)?)", answer, re.I)
            if price_matches and len(valid_citations) == 1:
                try:
                    claimed_price = float(price_matches[0])
                    if abs(claimed_price - cit.price_pkr) > 1.0:
                        unsupported_claims.append(
                            f"Price contradiction: Answer cited PKR {claimed_price} for [{cid}], "
                            f"but evidence record states PKR {cit.price_pkr}."
                        )
                except ValueError:
                    pass

        # 5. Determine overall GroundingStatus
        has_hallucinations = len(hallucinated_citations) > 0 or len(unsupported_claims) > 0
        if has_hallucinations:
            status = GroundingStatus.UNSUPPORTED
            is_grounded = False
            notes.append("Response contains hallucinated citations or unsupported factual claims.")
        else:
            is_grounded = True
            # Check allergen sufficiency of the evidence
            has_conflict = any(
                c.allergen_status == "CONFLICT" or c.verification_status == "CONFLICTING_EVIDENCE"
                for c in valid_citations
            )
            all_unknown = all(
                c.allergen_status == "UNKNOWN" or c.verification_status == "INSUFFICIENT_EVIDENCE"
                for c in valid_citations
            )
            some_unknown = any(
                c.allergen_status == "UNKNOWN" or c.verification_status == "INSUFFICIENT_EVIDENCE"
                for c in valid_citations
            )

            if has_conflict:
                status = GroundingStatus.CONFLICTING_EVIDENCE
                notes.append("Retrieved evidence contains conflicting allergen data.")
            elif all_unknown:
                status = GroundingStatus.INSUFFICIENT_EVIDENCE
                notes.append("All retrieved items lack reliable allergen evidence.")
            elif some_unknown:
                status = GroundingStatus.PARTIALLY_SUPPORTED
                notes.append("Some retrieved items contain unverified allergen information.")
            else:
                status = GroundingStatus.SUPPORTED
                notes.append("All claims are fully grounded in verified retrieved evidence.")

        return GroundingVerificationReport(
            is_grounded=is_grounded,
            status=status,
            cited_ids=cited_ids,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            unsupported_claims=unsupported_claims,
            warnings=warnings,
            evidence_count=len(retrieved_docs),
            verification_notes=notes,
        )

    def synthesize_grounded_answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        warnings: Optional[List[str]] = None,
        parsed_constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Citation], GroundingStatus]:
        """Deterministically synthesize a grounded, cited response from verified evidence.

        Args:
            query: User's search query.
            retrieved_docs: List of candidate menu item documents.
            warnings: Active safety warnings from pre-filter and allergen verifier.
            parsed_constraints: Extracted query constraints.

        Returns:
            Tuple of (formatted_answer_string, list_of_citations, grounding_status).
        """
        if not retrieved_docs:
            return (
                "Based on the retrieved menu information, there is insufficient evidence to answer your request.",
                [],
                GroundingStatus.INSUFFICIENT_EVIDENCE,
            )

        citations: List[Citation] = []
        doc_warnings = list(warnings or [])

        for doc in retrieved_docs:
            cit = self.build_citation(doc, parsed_constraints=parsed_constraints)
            citations.append(cit)
            if cit.warning and cit.warning not in doc_warnings:
                doc_warnings.append(cit.warning)

        # Determine overall synthesis status
        has_conflict = any(
            c.allergen_status == "CONFLICT" or c.verification_status == "CONFLICTING_EVIDENCE"
            for c in citations
        )
        all_unknown = all(
            c.allergen_status == "UNKNOWN" or c.verification_status == "INSUFFICIENT_EVIDENCE"
            for c in citations
        )
        some_unknown = any(
            c.allergen_status == "UNKNOWN" or c.verification_status == "INSUFFICIENT_EVIDENCE"
            for c in citations
        )

        if has_conflict:
            header_status = GroundingStatus.CONFLICTING_EVIDENCE
        elif all_unknown:
            header_status = GroundingStatus.INSUFFICIENT_EVIDENCE
        elif some_unknown:
            header_status = GroundingStatus.PARTIALLY_SUPPORTED
        else:
            header_status = GroundingStatus.SUPPORTED

        lines: List[str] = [
            f"{header_status.value}\n",
            f'Here are the menu options retrieved for your query: "{query}":\n',
        ]

        for idx, cit in enumerate(citations, start=1):
            halal_str = "Yes (Certified)" if cit.halal else "No (Non-Halal)"
            if cit.allergen_status == "KNOWN_NO_ALLERGENS":
                allergen_desc = "None listed (Verified)"
            elif cit.allergen_status == "UNKNOWN":
                allergen_desc = "Unverified / Unknown (Missing data)"
            elif cit.allergen_status == "CONFLICT":
                allergen_desc = "Conflicting Evidence (Contradictory source records)"
            elif cit.allergens:
                allergen_desc = ", ".join(cit.allergens)
            else:
                allergen_desc = "None listed"

            lines.append(
                f"{idx}. {cit.dish_name} at {cit.restaurant_name} [{cit.source_id}]\n"
                f"   - Restaurant: {cit.restaurant_name}\n"
                f"   - Price: PKR {cit.price_pkr}\n"
                f"   - Location: {cit.location}\n"
                f"   - Halal: {halal_str}\n"
                f"   - Spice: {cit.spice_level}\n"
                f"   - Allergens noted: {allergen_desc}\n"
                f"   - Why it matches: {cit.match_reason}\n"
                f"   - Source: {cit.source_menu} [{cit.source_id}]"
            )

        if doc_warnings:
            lines.append("\nWarnings:")
            for w in doc_warnings:
                lines.append(f"- {w}")

        lines.append(
            "\nNote: Semantic retrieval alone does not guarantee that all requested numeric, "
            "dietary, or allergen constraints are satisfied. Source IDs are indicated in brackets."
        )

        return "\n".join(lines), citations, header_status


# Global default singleton instance
_default_citation_engine: Optional[CitationEngine] = None


def get_citation_engine() -> CitationEngine:
    """Return the cached default CitationEngine instance."""
    global _default_citation_engine
    if _default_citation_engine is None:
        _default_citation_engine = CitationEngine()
    return _default_citation_engine
