"""Context builder module formatting retrieved menu items into grounded prompt context."""

from typing import Any, Dict, List, Tuple


def format_single_document(doc: Dict[str, Any], doc_index: int) -> str:
    """Format a single retrieved menu document dictionary into a standardized text block."""
    meta = doc.get("metadata", {})
    dish = doc.get("dish_name") or meta.get("dish_name", "Unknown Dish")
    restaurant = doc.get("restaurant_name") or meta.get("restaurant_name", "Unknown Restaurant")
    price = doc.get("price_pkr", meta.get("price_pkr", "N/A"))
    location = doc.get("location") or meta.get("location", "Unknown Location")
    
    # Halal representation
    halal_val = doc.get("halal") if doc.get("halal") is not None else meta.get("halal")
    halal_raw = meta.get("halal_raw")
    if halal_val is True:
        halal_str = "Yes (Certified)"
    elif halal_val is False and halal_raw != "unknown":
        halal_str = "No (Non-Halal)"
    else:
        halal_str = "Unverified / Unknown"

    spice = doc.get("spice_level") or meta.get("spice_level", "Unknown")
    
    # Allergens
    allergens = doc.get("allergens", [])
    allergen_status = doc.get("allergen_status") or meta.get("allergen_status", "UNKNOWN")
    if allergen_status == "KNOWN_NO_ALLERGENS":
        allergen_str = "None listed (Verified)"
    elif allergen_status == "UNKNOWN":
        allergen_str = "Unverified / Unknown (Missing data)"
    elif allergen_status == "CONFLICT":
        allergen_str = "Conflicting Evidence (Contradictory source records)"
    elif allergens:
        allergen_str = ", ".join(allergens)
    else:
        allergen_str = "Unspecified"

    # Description and ingredients
    desc = meta.get("description", "")
    ingredients = meta.get("ingredients", "")
    source_menu = meta.get("source", "Standard Menu")
    source_id = doc.get("id") or meta.get("id", f"dish_{doc_index:03d}")

    lines = [
        f"[DOCUMENT {doc_index}]",
        f"Dish Name: {dish}",
        f"Restaurant: {restaurant}",
        f"Price: PKR {price}",
        f"Location: {location}",
        f"Halal Status: {halal_str}",
        f"Spice Level: {spice}",
        f"Allergens: {allergen_str} [{allergen_status}]",
    ]
    if desc:
        lines.append(f"Description: {desc}")
    if ingredients:
        lines.append(f"Ingredients: {ingredients}")
    lines.append(f"Source ID: {source_id}")
    lines.append(f"Source Document: {source_menu}")

    return "\n".join(lines)


def build_context(retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Convert a list of retrieved documents into a consolidated prompt context string.

    Args:
        retrieved_docs: List of document dictionaries returned by retriever.

    Returns:
        Tuple of (formatted_context_string, list_of_source_ids).
    """
    if not retrieved_docs:
        return "No retrieved documents available.", []

    doc_blocks: List[str] = []
    source_ids: List[str] = []

    for idx, doc in enumerate(retrieved_docs, start=1):
        block = format_single_document(doc, idx)
        doc_blocks.append(block)
        doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
        if doc_id and doc_id not in source_ids:
            source_ids.append(doc_id)

    full_context = "\n\n".join(doc_blocks)
    return full_context, source_ids
