"""Data normalization module for allergen, dietary, price, location, and text fields."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLERGEN_PATTERNS: Dict[str, List[re.Pattern]] = {
    "dairy": [
        re.compile(r"(?i)\b(?:milk|butter|cream|cheese|paneer|yogurt|yoghurt|desi ghee|ghee|whey|curd|malai|dairy)\b"),
    ],
    "peanuts": [
        re.compile(r"(?i)\b(?:peanut|peanuts|groundnut|groundnuts|peanut butter|satay)\b"),
    ],
    "eggs": [
        re.compile(r"(?i)\b(?:egg|eggs|egg wash|mayonnaise|mayo)\b"),
    ],
    "gluten": [
        re.compile(r"(?i)\b(?:gluten|wheat|flour|maida|atta|suji|semolina|barley|rye|breadcrumb|breadcrumbs|bread|naan|roti|bun|brioche|filo|pastry)\b"),
    ],
    "soy": [
        re.compile(r"(?i)\b(?:soy|soya|soybean|edamame|soy sauce|tofu)\b"),
    ],
    "tree nuts": [
        re.compile(r"(?i)\b(?:tree nut|tree nuts|almond|almonds|badam|cashew|cashews|kaju|pistachio|pistachios|pista|walnut|walnuts|akhrot|hazelnut|pecan)\b"),
    ],
    "shellfish": [
        re.compile(r"(?i)\b(?:shellfish|prawn|prawns|shrimp|shrimps|crab|crabs|lobster)\b"),
    ],
}


def mask_allergen_exceptions(allergen: str, text: str) -> str:
    """Mask terms that contain keyword substrings but are medically/factually unrelated."""
    t = text
    if allergen == "dairy":
        # Plant milks are dairy-free
        t = re.sub(r"(?i)\b(?:coconut|almond|soy|soya|oat|rice|cashew|hemp)\s+milk\b", " [plant_milk] ", t)
        # Non-dairy butters (e.g. peanut butter, cocoa butter, shea butter)
        t = re.sub(r"(?i)\b(?:peanut|cocoa|shea|apple)\s+butter\b", " [non_dairy_butter] ", t)
        # Non-dairy creams (e.g. coconut cream, cream of tartar)
        t = re.sub(r"(?i)\b(?:coconut\s+cream|cream\s+of\s+tartar)\b", " [non_dairy_cream] ", t)
    elif allergen == "gluten":
        # Naturally gluten-free flours commonly used in Pakistani & global cooking
        t = re.sub(r"(?i)\b(?:chickpea|besan|rice|almond|coconut|corn|potato)\s+flour\b", " [gf_flour] ", t)
    return t

# Canonical Pakistani Location Mappings
CANONICAL_LOCATIONS: Dict[str, str] = {
    "gulberg": "Gulberg",
    "dha": "DHA",
    "f-7": "F-7",
    "f7": "F-7",
    "f 7": "F-7",
    "saddar": "Saddar",
    "johar town": "Johar Town",
    "johartown": "Johar Town",
    "rawalpindi": "Rawalpindi",
    "bahria town": "Bahria Town",
    "bahria": "Bahria Town",
}


class ProcessedMenuItem(BaseModel):
    """Pydantic model representing a cleaned and normalized menu record."""

    id: str
    restaurant_name: str
    dish_name: str
    description: str = ""
    ingredients: str = ""
    price_pkr: float = Field(..., ge=0)
    location: str
    halal: Optional[bool] = None
    spice_level: str = "Unknown"
    allergens: List[str] = Field(default_factory=list)
    allergen_status: str = Field(..., description="KNOWN_NO_ALLERGENS | KNOWN_ALLERGENS | UNKNOWN | CONFLICT")
    category: str = "General"
    protein_g: float = Field(default=0.0, ge=0)
    availability: bool = True
    cuisine: str = "Pakistani"
    source: str = "Restaurant Menu"
    source_url: str = ""
    timestamp: str = ""
    search_text: str = ""

    @field_validator("description", "ingredients", "cuisine", "source", "source_url", "timestamp", "search_text", mode="before")
    @classmethod
    def parse_string_fields(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    @field_validator("halal", mode="before")
    @classmethod
    def parse_halal_field(cls, v: Any) -> Optional[bool]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ["true", "1", "yes"]:
            return True
        elif s in ["false", "0", "no"]:
            return False
        return None

    @field_validator("allergens", mode="before")
    @classmethod
    def parse_allergens(cls, v: Any) -> List[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, list):
            return [str(item).strip().lower() for item in v if item]
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean or v_clean.lower() in ["none", "nan", "null", "[]"]:
                return []
            if v_clean.startswith("[") and v_clean.endswith("]"):
                try:
                    parsed = json.loads(v_clean)
                    if isinstance(parsed, list):
                        return [str(x).strip().lower() for x in parsed if x]
                except json.JSONDecodeError:
                    pass
            items = [item.strip().strip("'\"").lower() for item in v_clean.split(",")]
            return [i for i in items if i]
        return []


def normalize_price(val: Any) -> float:
    """Normalize price input to float PKR. Reject negative values.

    Handles string formats like 'PKR 1,250', 'Rs. 1250', '1,250', 1250.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        raise ValueError("Price cannot be null or empty")

    if isinstance(val, (int, float)):
        num = float(val)
        if num < 0:
            raise ValueError(f"Negative price is invalid: {num}")
        return num

    val_str = str(val).strip()
    # Strip common currency symbols and labels (e.g. PKR, Rs., Rs, $, Rupees)
    cleaned = re.sub(r"(?i)\b(?:pkr|rupees)\b|rs\.?|\$", "", val_str)
    cleaned = cleaned.replace(",", "").strip()

    try:
        num = float(cleaned)
    except ValueError:
        raise ValueError(f"Cannot parse price string: '{val}'")

    if num < 0:
        raise ValueError(f"Negative price is invalid: {num}")
    return num


def normalize_halal(val: Any) -> Optional[bool]:
    """Normalize Halal certification status.

    Returns:
        True, False, or None (unknown). Missing values are NOT assumed to be True.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    if isinstance(val, bool):
        return val

    s = str(val).strip().lower()
    if s in ["true", "1", "yes", "halal"]:
        return True
    elif s in ["false", "0", "no", "non-halal", "haram"]:
        return False
    elif s in ["unknown", "none", "nan", "null", ""]:
        return None
    else:
        logger.warning("Unrecognized Halal value '%s'; defaulting to None (unknown)", val)
        return None


def normalize_spice_level(val: Any) -> str:
    """Normalize spice level to one of: Mild, Medium, Spicy, Unknown."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"

    s = str(val).strip().lower()
    if s in ["none", "no spice", "mild", "low"]:
        return "Mild"
    elif s in ["medium", "med", "moderate"]:
        return "Medium"
    elif s in ["spicy", "hot", "extra spicy", "very spicy"]:
        return "Spicy"
    elif s in ["unknown", "nan", "null", ""]:
        return "Unknown"
    return "Unknown"


def normalize_location(val: Any) -> str:
    """Normalize location to standard canonical casing and names."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"

    s = str(val).strip()
    key = s.lower()
    if key in CANONICAL_LOCATIONS:
        return CANONICAL_LOCATIONS[key]
    return s.title()


def detect_allergens_in_text(text: str) -> Set[str]:
    """Scan ingredient or description text for specific allergen synonyms using context-aware patterns."""
    if not text:
        return set()

    detected: Set[str] = set()
    text_clean = f" {text.strip()} "

    for allergen, patterns in ALLERGEN_PATTERNS.items():
        masked_text = mask_allergen_exceptions(allergen, text_clean)
        for pat in patterns:
            if pat.search(masked_text):
                detected.add(allergen)
                break

    return detected


def normalize_allergens(
    raw_allergens_val: Any, 
    ingredients_text: str = ""
) -> Tuple[List[str], str]:
    """Deterministically normalize allergen data and assign safety status.

    Returns:
        (normalized_allergens_list, allergen_status)
        where allergen_status is one of:
        - KNOWN_ALLERGENS
        - KNOWN_NO_ALLERGENS
        - UNKNOWN
        - CONFLICT
    """
    # 1. Parse raw allergen representation
    raw_list: List[str] = []
    is_missing = False
    is_conflict = False

    if raw_allergens_val is None or (isinstance(raw_allergens_val, float) and pd.isna(raw_allergens_val)):
        is_missing = True
    elif isinstance(raw_allergens_val, list):
        raw_list = [str(x).strip().lower() for x in raw_allergens_val if str(x).strip()]
    elif isinstance(raw_allergens_val, str):
        cleaned_str = raw_allergens_val.strip()
        if not cleaned_str or cleaned_str.lower() in ["nan", "null", "not available", "unverified"]:
            is_missing = True
        elif cleaned_str.lower() in ["unknown", "['unknown']", '["unknown"]']:
            is_missing = True
        elif cleaned_str.lower() in ["conflict", "['conflict']", '["conflict"]']:
            is_conflict = True
        elif cleaned_str == "[]" or cleaned_str.lower() == "none":
            raw_list = []
        elif cleaned_str.startswith("[") and cleaned_str.endswith("]"):
            try:
                parsed = json.loads(cleaned_str)
                if isinstance(parsed, list):
                    raw_list = [str(x).strip().lower() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                raw_list = [x.strip().strip("'\"").lower() for x in cleaned_str[1:-1].split(",") if x.strip()]
        else:
            raw_list = [x.strip().strip("'\"").lower() for x in cleaned_str.split(",") if x.strip()]

    # 2. Check for explicit conflict flag
    if is_conflict or "conflict" in raw_list:
        return ["conflict"], "CONFLICT"

    # 3. Check for explicit missing/unknown flag
    if is_missing or "unknown" in raw_list:
        return ["unknown"], "UNKNOWN"

    # 4. Standardize listed allergens against known patterns
    mapped_allergens: Set[str] = set()
    for item in raw_list:
        detected = detect_allergens_in_text(item)
        if detected:
            mapped_allergens.update(detected)
        elif item:
            # Preserve recognized custom allergen if not in standard dictionary
            mapped_allergens.add(item)

    # 5. Scan ingredients text for hidden allergen indicators
    hidden_detected = detect_allergens_in_text(ingredients_text)
    mapped_allergens.update(hidden_detected)

    # 6. Assign final status
    if mapped_allergens:
        return sorted(list(mapped_allergens)), "KNOWN_ALLERGENS"
    else:
        return [], "KNOWN_NO_ALLERGENS"


def build_search_text(rec: Dict[str, Any]) -> str:
    """Construct searchable text concatenation without hidden safety assumptions."""
    parts = [
        str(rec.get("restaurant_name", "")),
        str(rec.get("dish_name", "")),
        str(rec.get("description", "")),
        str(rec.get("ingredients", "")),
        str(rec.get("category", "")),
        str(rec.get("cuisine", "")),
        str(rec.get("spice_level", "")),
    ]
    # Filter empty parts and collapse whitespace
    text = " ".join(p.strip() for p in parts if p.strip())
    return re.sub(r"\s+", " ", text).strip()


def normalize_record(raw_rec: Dict[str, Any]) -> ProcessedMenuItem:
    """Normalize all attributes of a single menu item record."""
    price_val = normalize_price(raw_rec.get("price_pkr"))
    halal_val = normalize_halal(raw_rec.get("halal"))
    spice_val = normalize_spice_level(raw_rec.get("spice_level"))
    location_val = normalize_location(raw_rec.get("location"))
    ingredients_val = str(raw_rec.get("ingredients", "")).strip()

    allergens_list, allergen_status = normalize_allergens(
        raw_rec.get("allergens"), 
        ingredients_text=ingredients_val
    )

    protein_val = 0.0
    try:
        raw_protein = raw_rec.get("protein_g", 0)
        protein_val = float(raw_protein) if raw_protein is not None and not pd.isna(raw_protein) else 0.0
    except (ValueError, TypeError):
        protein_val = 0.0

    availability_val = True
    raw_avail = raw_rec.get("availability", True)
    if isinstance(raw_avail, str):
        availability_val = raw_avail.strip().lower() in ["true", "1", "yes"]
    elif isinstance(raw_avail, bool):
        availability_val = raw_avail

    search_text = build_search_text({
        "restaurant_name": raw_rec.get("restaurant_name", ""),
        "dish_name": raw_rec.get("dish_name", ""),
        "description": raw_rec.get("description", ""),
        "ingredients": ingredients_val,
        "category": raw_rec.get("category", ""),
        "cuisine": raw_rec.get("cuisine", ""),
        "spice_level": spice_val,
    })

    return ProcessedMenuItem(
        id=str(raw_rec.get("id")).strip(),
        restaurant_name=str(raw_rec.get("restaurant_name")).strip(),
        dish_name=str(raw_rec.get("dish_name")).strip(),
        description=str(raw_rec.get("description", "")).strip(),
        ingredients=ingredients_val,
        price_pkr=price_val,
        location=location_val,
        halal=halal_val,
        spice_level=spice_val,
        allergens=allergens_list,
        allergen_status=allergen_status,
        category=str(raw_rec.get("category", "General")).strip(),
        protein_g=protein_val,
        availability=availability_val,
        cuisine=str(raw_rec.get("cuisine", "Pakistani")).strip(),
        source=str(raw_rec.get("source", "Restaurant Menu")).strip(),
        source_url=str(raw_rec.get("source_url", "")).strip(),
        timestamp=str(raw_rec.get("timestamp", "")).strip(),
        search_text=search_text,
    )
