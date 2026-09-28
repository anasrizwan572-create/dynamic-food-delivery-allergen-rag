"""Deterministic query and constraint parser for restaurant menu queries."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from ingestion.normalize_data import ALLERGEN_PATTERNS, CANONICAL_LOCATIONS, mask_allergen_exceptions

# Common Allergen Synonyms for Query Extraction
ALLERGEN_QUERY_TERMS: Dict[str, List[str]] = {
    "dairy": [
        "dairy", "milk", "butter", "cream", "cheese", "paneer", 
        "yogurt", "yoghurt", "ghee", "desi ghee", "whey", "curd", "malai", "lactose"
    ],
    "peanuts": [
        "peanut", "peanuts", "groundnut", "groundnuts", "peanut butter"
    ],
    "eggs": [
        "egg", "eggs", "mayo", "mayonnaise"
    ],
    "gluten": [
        "gluten", "wheat", "flour", "maida", "atta", "bread", "breadcrumbs", 
        "naan", "roti", "bun", "pastry"
    ],
    "soy": [
        "soy", "soya", "soybean", "edamame", "tofu"
    ],
    "tree nuts": [
        "tree nut", "tree nuts", "nut", "nuts", "almond", "almonds", "badam", 
        "cashew", "cashews", "kaju", "pistachio", "pistachios", "pista", 
        "walnut", "walnuts", "akhrot"
    ],
    "shellfish": [
        "shellfish", "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster"
    ],
}


class HardConstraints(BaseModel):
    """Hard constraints that candidate dishes MUST satisfy."""

    max_price_pkr: Optional[float] = None
    min_price_pkr: Optional[float] = None
    location: Optional[str] = None
    halal: Optional[bool] = None
    excluded_allergens: List[str] = Field(default_factory=list)
    availability: Optional[bool] = True


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


def extract_price_constraints(query: str) -> Tuple[Optional[float], Optional[float], str]:
    """Extract maximum and minimum price constraints from natural language query.

    Handles:
        - "under 1500", "below 1500", "less than 1500", "up to 1500", "within 1500"
        - "above 500", "over 500", "more than 500", "starting from 500"
        - "between 500 and 1500", "500 to 1500"
    """
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    cleaned = query

    # 1. Between MIN and MAX
    range_match = re.search(
        r"(?i)\b(?:between|from)\s+(?:pkr|rs\.?)?\s*(\d+(?:,\d+)?)\s*(?:and|to|-)\s*(?:pkr|rs\.?)?\s*(\d+(?:,\d+)?)",
        cleaned,
    )
    if range_match:
        val1 = float(range_match.group(1).replace(",", ""))
        val2 = float(range_match.group(2).replace(",", ""))
        min_price = min(val1, val2)
        max_price = max(val1, val2)
        cleaned = cleaned[:range_match.start()] + cleaned[range_match.end():]
        return max_price, min_price, cleaned

    # 2. Maximum price (under, below, less than, up to, max)
    max_match = re.search(
        r"(?i)\b(?:under|below|less\s+than|up\s+to|within|max(?:imum)?(?:\s+price)?|at\s+most)\s*(?:of)?\s*(?:pkr|rs\.?|rupees)?\s*(\d+(?:,\d+)?)",
        cleaned,
    )
    if max_match:
        max_price = float(max_match.group(1).replace(",", ""))
        cleaned = cleaned[:max_match.start()] + cleaned[max_match.end():]

    # 3. Minimum price (above, over, more than, at least, min)
    min_match = re.search(
        r"(?i)\b(?:above|over|more\s+than|greater\s+than|at\s+least|min(?:imum)?(?:\s+price)?)\s*(?:of)?\s*(?:pkr|rs\.?|rupees)?\s*(\d+(?:,\d+)?)",
        cleaned,
    )
    if min_match:
        min_price = float(min_match.group(1).replace(",", ""))
        cleaned = cleaned[:min_match.start()] + cleaned[min_match.end():]

    # 4. Trailing "under PKR 1500" or "PKR 1500 or less"
    or_less_match = re.search(
        r"(?i)(?:pkr|rs\.?|rupees)?\s*(\d+(?:,\d+)?)\s*(?:or\s+less|or\s+under|budget)",
        cleaned,
    )
    if or_less_match and max_price is None:
        max_price = float(or_less_match.group(1).replace(",", ""))
        cleaned = cleaned[:or_less_match.start()] + cleaned[or_less_match.end():]

    return max_price, min_price, cleaned


def extract_location(query: str) -> Tuple[Optional[str], str]:
    """Extract and normalize location zone from query."""
    cleaned = query
    for raw_loc, canonical in CANONICAL_LOCATIONS.items():
        pattern = rf"(?i)\b(?:in|at|near|around)?\s*\b{re.escape(raw_loc)}\b"
        match = re.search(pattern, cleaned)
        if match:
            # Remove matched location phrase from cleaned search text
            cleaned = re.sub(pattern, "", cleaned, count=1)
            return canonical, cleaned
    return None, cleaned


def extract_halal(query: str) -> Tuple[Optional[bool], str]:
    """Extract Halal requirement from query."""
    cleaned = query
    if re.search(r"(?i)\bnon-halal\b", cleaned):
        cleaned = re.sub(r"(?i)\bnon-halal\b", "", cleaned)
        return False, cleaned
    if re.search(r"(?i)\bhalal\b", cleaned):
        cleaned = re.sub(r"(?i)\bhalal\b", "", cleaned)
        return True, cleaned
    return None, cleaned


def extract_allergen_exclusions(query: str) -> Tuple[List[str], str]:
    """Extract explicitly excluded allergens from query.

    Matches patterns like:
        - "without dairy"
        - "no peanuts"
        - "dairy-free"
        - "free from gluten"
        - "exclude peanuts and dairy"
        - "allergic to shellfish"
    """
    excluded: Set[str] = set()
    cleaned = query

    exclusion_prefixes = [
        r"(?i)\bwithout\s+([^,\.]+)",
        r"(?i)\bno\s+([^,\.]+)",
        r"(?i)\bfree\s+from\s+([^,\.]+)",
        r"(?i)\bexclude\s+([^,\.]+)",
        r"(?i)\bavoid\s+([^,\.]+)",
        r"(?i)\ballergic\s+to\s+([^,\.]+)",
    ]

    for pat in exclusion_prefixes:
        for match in re.finditer(pat, cleaned):
            phrase = match.group(1).lower()
            # Match allergen terms inside phrase
            for standard_allergen, synonyms in ALLERGEN_QUERY_TERMS.items():
                for syn in synonyms:
                    if re.search(rf"\b{re.escape(syn)}\b", phrase):
                        excluded.add(standard_allergen)
            # Remove the exclusion phrase from cleaned query text
            cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]

    # Hyphenated patterns: e.g. "dairy-free", "peanut-free", "gluten-free"
    hyphen_pattern = r"(?i)\b([\w\s]+)-(?:free)\b"
    for match in re.finditer(hyphen_pattern, cleaned):
        item = match.group(1).strip().lower()
        for standard_allergen, synonyms in ALLERGEN_QUERY_TERMS.items():
            if item == standard_allergen or item in synonyms:
                excluded.add(standard_allergen)
        cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]

    return sorted(list(excluded)), cleaned


def extract_soft_preferences(query: str) -> Tuple[SoftPreferences, str]:
    """Extract soft preferences: category, spice level, protein preference, cuisine."""
    cleaned = query
    prefs = SoftPreferences()

    # Protein preference
    if re.search(r"(?i)\b(?:high\s+protein|high-protein|protein\s+rich|rich\s+in\s+protein)\b", cleaned):
        prefs.protein_preference = "high"
        cleaned = re.sub(r"(?i)\b(?:high\s+protein|high-protein|protein\s+rich|rich\s+in\s+protein)\b", "", cleaned)

    # Spice level
    if re.search(r"(?i)\b(?:extra\s+spicy|very\s+spicy|hot)\b", cleaned):
        prefs.spice_level = "Spicy"
        cleaned = re.sub(r"(?i)\b(?:extra\s+spicy|very\s+spicy|hot)\b", "", cleaned)
    elif re.search(r"(?i)\b(?:spicy)\b", cleaned):
        prefs.spice_level = "Spicy"
        cleaned = re.sub(r"(?i)\b(?:spicy)\b", "", cleaned)
    elif re.search(r"(?i)\b(?:medium\s+spice|medium)\b", cleaned):
        prefs.spice_level = "Medium"
        cleaned = re.sub(r"(?i)\b(?:medium\s+spice|medium)\b", "", cleaned)
    elif re.search(r"(?i)\b(?:mild|non-spicy|not\s+spicy)\b", cleaned):
        prefs.spice_level = "Mild"
        cleaned = re.sub(r"(?i)\b(?:mild|non-spicy|not\s+spicy)\b", "", cleaned)

    # Common categories
    categories = ["chicken", "beef", "mutton", "seafood", "fish", "prawns", "vegetarian", "rice", "biryani", "burger", "dessert", "bbq"]
    for cat in categories:
        if re.search(rf"(?i)\b{cat}\b", cleaned):
            prefs.category = "Seafood" if cat in ["fish", "prawns"] else cat.capitalize()
            break

    # Cuisines
    cuisines = ["pakistani", "continental", "chinese", "thai", "italian", "middle eastern"]
    for cuis in cuisines:
        if re.search(rf"(?i)\b{cuis}\b", cleaned):
            prefs.cuisine = cuis.title()
            break

    return prefs, cleaned


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse natural-language query into structured hard constraints and soft preferences.

    Example:
        "Find high-protein Halal chicken meals under PKR 1500 without dairy or peanuts in Gulberg."
    Yields:
        HardConstraints(max_price_pkr=1500.0, halal=True, location='Gulberg', excluded_allergens=['dairy', 'peanuts'])
        SoftPreferences(category='Chicken', protein_preference='high')
    """
    working_text = raw_query.strip()

    # 1. Price constraints (Hard)
    max_price, min_price, working_text = extract_price_constraints(working_text)

    # 2. Location (Hard)
    location, working_text = extract_location(working_text)

    # 3. Halal requirement (Hard)
    halal, working_text = extract_halal(working_text)

    # 4. Excluded allergens (Hard)
    excluded_allergens, working_text = extract_allergen_exclusions(working_text)

    # 5. Soft preferences
    soft_prefs, working_text = extract_soft_preferences(working_text)

    hard_constraints = HardConstraints(
        max_price_pkr=max_price,
        min_price_pkr=min_price,
        location=location,
        halal=halal,
        excluded_allergens=excluded_allergens,
        availability=True,
    )

    # Clean residual filler words from search query
    cleaned_query = re.sub(r"(?i)\b(find|show|give|me|meals?|food|dishes?|options?|available|please)\b", "", working_text)
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
    if not cleaned_query:
        cleaned_query = raw_query.strip()

    return ParsedQuery(
        raw_query=raw_query,
        hard_constraints=hard_constraints,
        soft_preferences=soft_prefs,
        cleaned_query=cleaned_query,
    )
