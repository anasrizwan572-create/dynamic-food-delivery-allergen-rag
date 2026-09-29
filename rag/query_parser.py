"""Deterministic query and constraint parser for restaurant menu queries."""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ingestion.normalize_data import CANONICAL_LOCATIONS

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ALLERGEN_QUERY_TERMS: Dict[str, List[str]] = {
    "dairy": [
        "dairy", "milk", "butter", "cream", "cheese", "paneer",
        "yogurt", "yoghurt", "ghee", "desi ghee", "whey", "curd", "malai", "lactose",
    ],
    "peanuts": ["peanut", "peanuts", "groundnut", "groundnuts", "peanut butter"],
    "eggs": ["egg", "eggs", "mayo", "mayonnaise"],
    "gluten": [
        "gluten", "wheat", "flour", "maida", "atta", "bread", "breadcrumbs",
        "naan", "roti", "bun", "pastry",
    ],
    "soy": ["soy", "soya", "soybean", "edamame", "tofu"],
    "tree nuts": [
        "tree nut", "tree nuts", "nut", "nuts", "almond", "almonds", "badam",
        "cashew", "cashews", "kaju", "pistachio", "pistachios", "pista",
        "walnut", "walnuts", "akhrot",
    ],
    "shellfish": ["shellfish", "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster"],
}

# term -> allergens it implies. A generic "nut" is ambiguous, so exclude both
# (over-excluding is the safe failure mode for allergen filtering).
_TERM_TO_ALLERGENS: Dict[str, List[str]] = {}
for _allergen, _terms in ALLERGEN_QUERY_TERMS.items():
    for _t in _terms:
        _TERM_TO_ALLERGENS.setdefault(_t, []).append(_allergen)
for _t in ("nut", "nuts"):
    _TERM_TO_ALLERGENS[_t] = ["tree nuts", "peanuts"]

_TERMS_ALT = "|".join(re.escape(t) for t in sorted(_TERM_TO_ALLERGENS, key=len, reverse=True))
_TERM_RE = re.compile(rf"\b(?:{_TERMS_ALT})\b", re.I)
_LIST = rf"(?:{_TERMS_ALT})(?:\s*(?:,|/|&|\band\b|\bor\b)\s*(?:{_TERMS_ALT}))*"
_EXCLUDE_RE = re.compile(
    rf"\b(?:without|no|free\s+from|exclude|excluding|avoid|allergic\s+to|allergy\s+to)\s+(?:any\s+)?({_LIST})\b",
    re.I,
)
_FREE_RE = re.compile(rf"\b({_TERMS_ALT})[-\s]free\b", re.I)

SPICE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:mild|non-spicy|not\s+spicy)\b", re.I), "Mild"),
    (re.compile(r"\b(?:extra\s+spicy|very\s+spicy|hot|spicy)\b", re.I), "Spicy"),
    (re.compile(r"\b(?:medium\s+spice|medium)\b", re.I), "Medium"),
]
PROTEIN_RE = re.compile(r"\b(?:high[\s-]protein|protein[\s-]rich|rich\s+in\s+protein)\b", re.I)

# A dish type ("burger") is WHAT the user wants; a protein ("beef") only
# describes it. Keeping them apart stops "beef burger" from matching any beef dish.
DISH_TYPES = ["burger", "biryani", "dessert", "rice", "bbq"]
HARD_DISH_TYPES = {"burger"}  # dish types strict enough to filter on
PROTEINS = ["chicken", "beef", "mutton", "seafood", "fish", "prawns", "vegetarian"]
_SEAFOOD_ALIASES = {"fish", "prawns"}
CUISINES = ["pakistani", "continental", "chinese", "thai", "italian", "middle eastern"]

_LOCATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf"\b(?:(?:in|at|near|around)\s+)?{re.escape(raw)}\b", re.I), canonical)
    for raw, canonical in sorted(CANONICAL_LOCATIONS.items(), key=lambda kv: len(kv[0]), reverse=True)
]

_CUR = r"(?:pkr|rs\.?|rupees)?\s*"
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_RANGE_RE = re.compile(rf"\b(?:between|from)\s+{_CUR}{_NUM}\s*(?:and|to|-)\s*{_CUR}{_NUM}", re.I)
_MAX_RE = re.compile(
    rf"\b(?:under|below|less\s+than|up\s+to|within|max(?:imum)?(?:\s+price)?|at\s+most)\s*(?:of\s+)?{_CUR}{_NUM}",
    re.I,
)
_MIN_RE = re.compile(
    rf"\b(?:above|over|more\s+than|greater\s+than|at\s+least|min(?:imum)?(?:\s+price)?)\s*(?:of\s+)?{_CUR}{_NUM}",
    re.I,
)
_OR_LESS_RE = re.compile(rf"\b{_CUR}{_NUM}\s*(?:or\s+less|or\s+under|budget)\b", re.I)
_FILLER_RE = re.compile(r"\b(?:find|show|give|me|meals?|food|dishes?|options?|available|please)\b", re.I)
_NON_HALAL_RE = re.compile(r"\bnon[-\s]?halal\b", re.I)
_HALAL_RE = re.compile(r"\bhalal\b", re.I)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class HardConstraints(BaseModel):
    """Hard constraints that candidate dishes MUST satisfy."""

    max_price_pkr: Optional[float] = None
    min_price_pkr: Optional[float] = None
    location: Optional[str] = None
    halal: Optional[bool] = None
    excluded_allergens: List[str] = Field(default_factory=list)
    availability: Optional[bool] = True
    categories: List[str] = Field(default_factory=list)  # dish type if present, else protein
    proteins: List[str] = Field(default_factory=list)    # must ALSO match (AND) in the backend


class SoftPreferences(BaseModel):
    """Soft preferences that influence ranking or descriptive matching."""

    category: Optional[str] = None
    cuisine: Optional[str] = None
    spice_level: Optional[str] = None
    protein_preference: Optional[str] = None
    restaurant: Optional[str] = None


class ParsedQuery(BaseModel):
    """Structured representation of a parsed natural-language query."""

    raw_query: str
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    cleaned_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "hard_constraints": self.hard_constraints.model_dump(exclude_none=True),
            "soft_preferences": self.soft_preferences.model_dump(exclude_none=True),
            "cleaned_query": self.cleaned_query,
        }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _cut(text: str, m: re.Match) -> str:
    return text[: m.start()] + " " + text[m.end():]


def extract_price_constraints(query: str) -> Tuple[Optional[float], Optional[float], str]:
    """Extract (max_price, min_price, cleaned_query).

    Handles "under/below/up to 1500", "above/over 500",
    "between 500 and 1500", and "1500 or less".
    """
    cleaned = query

    m = _RANGE_RE.search(cleaned)
    if m:
        a, b = _to_float(m.group(1)), _to_float(m.group(2))
        return max(a, b), min(a, b), _cut(cleaned, m)

    max_price = min_price = None

    m = _MAX_RE.search(cleaned)
    if m:
        max_price = _to_float(m.group(1))
        cleaned = _cut(cleaned, m)

    m = _MIN_RE.search(cleaned)
    if m:
        min_price = _to_float(m.group(1))
        cleaned = _cut(cleaned, m)

    if max_price is None:
        m = _OR_LESS_RE.search(cleaned)
        if m:
            max_price = _to_float(m.group(1))
            cleaned = _cut(cleaned, m)

    return max_price, min_price, cleaned


def extract_location(query: str) -> Tuple[Optional[str], str]:
    """Extract and normalize a location zone (longest names match first)."""
    for pattern, canonical in _LOCATION_PATTERNS:
        m = pattern.search(query)
        if m:
            return canonical, _cut(query, m)
    return None, query


def extract_halal(query: str) -> Tuple[Optional[bool], str]:
    """Extract Halal requirement (checks 'non-halal' first)."""
    if _NON_HALAL_RE.search(query):
        return False, _NON_HALAL_RE.sub(" ", query)
    if _HALAL_RE.search(query):
        return True, _HALAL_RE.sub(" ", query)
    return None, query


def extract_allergen_exclusions(query: str) -> Tuple[List[str], str]:
    """Extract excluded allergens.

    Matches "without dairy or peanuts", "no gluten", "free from soy",
    "exclude nuts", "allergic to shellfish", "dairy-free", "gluten free".

    Only the allergen terms are consumed, so "without dairy chicken biryani"
    keeps "chicken biryani" for category extraction and search.
    """
    excluded: set = set()

    def _collect(text: str) -> None:
        for term in _TERM_RE.findall(text):
            excluded.update(_TERM_TO_ALLERGENS[term.lower()])

    def _exclude_repl(m: re.Match) -> str:
        _collect(m.group(1))
        return " "

    def _free_repl(m: re.Match) -> str:
        _collect(m.group(1))
        return " "

    cleaned = _EXCLUDE_RE.sub(_exclude_repl, query)
    cleaned = _FREE_RE.sub(_free_repl, cleaned)
    return sorted(excluded), cleaned


def extract_soft_preferences(query: str) -> Tuple[SoftPreferences, str]:
    """Extract soft preferences: protein, spice level, category, cuisine."""
    cleaned = query
    prefs = SoftPreferences()

    if PROTEIN_RE.search(cleaned):
        prefs.protein_preference = "high"
        cleaned = PROTEIN_RE.sub(" ", cleaned)

    for pattern, level in SPICE_RULES:
        if pattern.search(cleaned):
            prefs.spice_level = level
            cleaned = pattern.sub(" ", cleaned)
            break

    dishes, proteins = extract_dish_and_protein(cleaned)
    prefs.category = (dishes or proteins or [None])[0]

    for cuisine in CUISINES:
        if re.search(rf"\b{cuisine}\b", cleaned, re.I):
            prefs.cuisine = cuisine.title()
            break

    return prefs, cleaned


def _label(term: str) -> str:
    return "Seafood" if term in _SEAFOOD_ALIASES else term.capitalize()


def _find_terms(terms: List[str], text: str) -> List[str]:
    found: List[str] = []
    for t in terms:
        if re.search(rf"\b{t}\b", text, re.I) and _label(t) not in found:
            found.append(_label(t))
    return found


def extract_dish_and_protein(text: str) -> Tuple[List[str], List[str]]:
    """Return (dish_types, proteins) mentioned in the text, e.g.
    "beef burger" -> (["Burger"], ["Beef"])."""
    return _find_terms(DISH_TYPES, text), _find_terms(PROTEINS, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(raw_query: str) -> ParsedQuery:
    """Parse a natural-language query into hard constraints and soft preferences.

    Example:
        "Find high-protein Halal chicken meals under PKR 1500 without dairy or peanuts in Gulberg."
    Yields:
        HardConstraints(max_price_pkr=1500.0, halal=True, location='Gulberg',
                        excluded_allergens=['dairy', 'peanuts'], categories=['Chicken'])
        SoftPreferences(category='Chicken', protein_preference='high')
    """
    text = raw_query.strip()

    max_price, min_price, text = extract_price_constraints(text)
    location, text = extract_location(text)
    halal, text = extract_halal(text)
    excluded_allergens, text = extract_allergen_exclusions(text)
    soft_prefs, text = extract_soft_preferences(text)

    dishes, proteins = extract_dish_and_protein(text)
    hard_dishes = [d for d in dishes if d.lower() in HARD_DISH_TYPES]

    hard = HardConstraints(
        max_price_pkr=max_price,
        min_price_pkr=min_price,
        location=location,
        halal=halal,
        excluded_allergens=excluded_allergens,
        categories=hard_dishes or proteins,
        proteins=proteins,
        availability=True,
    )

    cleaned_query = _FILLER_RE.sub(" ", text)
    cleaned_query = re.sub(r"[\s.,]+", " ", cleaned_query).strip() or raw_query.strip()

    return ParsedQuery(
        raw_query=raw_query,
        hard_constraints=hard,
        soft_preferences=soft_prefs,
        cleaned_query=cleaned_query,
    )