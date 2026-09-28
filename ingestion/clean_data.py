"""Data cleaning and structural validation module for menu datasets."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["id", "restaurant_name", "dish_name"]


class DataCleaningReport:
    """Report detailing the results of the data cleaning and validation phase."""

    def __init__(self, total_records: int = 0):
        self.total_records: int = total_records
        self.valid_records: int = 0
        self.invalid_records: int = 0
        self.duplicate_ids: List[str] = []
        self.validation_errors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "duplicate_ids": self.duplicate_ids,
            "validation_errors_count": len(self.validation_errors),
            "validation_errors": self.validation_errors,
        }

    def summary(self) -> str:
        return (
            f"--- Data Cleaning Summary ---\n"
            f"Total Records: {self.total_records}\n"
            f"Valid Records: {self.valid_records}\n"
            f"Invalid Records: {self.invalid_records}\n"
            f"Duplicate IDs: {len(self.duplicate_ids)} ({self.duplicate_ids})\n"
            f"Validation Errors: {len(self.validation_errors)}\n"
        )


def clean_string(val: Any) -> str:
    """Trim and clean string values, replacing NaNs and nulls with empty string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    return "" if text.lower() in ["nan", "null", "none"] else text


def detect_duplicate_ids(records: List[Dict[str, Any]]) -> List[str]:
    """Identify duplicate IDs in a record list."""
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for rec in records:
        rec_id = str(rec.get("id", "")).strip()
        if not rec_id:
            continue
        if rec_id in seen:
            duplicates.add(rec_id)
        else:
            seen.add(rec_id)
    return sorted(list(duplicates))


def validate_raw_record(rec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a single raw record against basic structural constraints.

    Returns:
        (is_valid, error_reason)
    """
    rec_id = clean_string(rec.get("id"))
    if not rec_id:
        return False, "Missing or empty 'id' field"

    dish_name = clean_string(rec.get("dish_name"))
    if not dish_name:
        return False, f"Dish ID '{rec_id}' has missing or empty 'dish_name'"

    restaurant_name = clean_string(rec.get("restaurant_name"))
    if not restaurant_name:
        return False, f"Dish ID '{rec_id}' has missing or empty 'restaurant_name'"

    return True, None


def clean_and_validate_records(
    records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], DataCleaningReport]:
    """Clean missing values, check for duplicate IDs, and validate records.

    Args:
        records: List of raw record dictionaries.

    Returns:
        Tuple of (clean_records, report).
    """
    report = DataCleaningReport(total_records=len(records))
    
    # 1. Check for duplicate IDs
    duplicate_ids = detect_duplicate_ids(records)
    report.duplicate_ids = duplicate_ids
    if duplicate_ids:
        logger.warning("Duplicate dish IDs found: %s", duplicate_ids)

    seen_ids: Set[str] = set()
    valid_records: List[Dict[str, Any]] = []

    for idx, raw_rec in enumerate(records):
        cleaned_rec = dict(raw_rec)

        # Clean string fields
        for k in ["id", "restaurant_name", "dish_name", "description", "ingredients", "category", "cuisine", "source"]:
            if k in cleaned_rec:
                cleaned_rec[k] = clean_string(cleaned_rec[k])

        is_valid, error = validate_raw_record(cleaned_rec)
        if not is_valid:
            report.invalid_records += 1
            report.validation_errors.append({"index": idx, "id": cleaned_rec.get("id"), "error": error})
            logger.warning("Record %d dropped: %s", idx, error)
            continue

        rec_id = cleaned_rec["id"]
        if rec_id in seen_ids:
            report.invalid_records += 1
            report.validation_errors.append(
                {"index": idx, "id": rec_id, "error": f"Duplicate ID '{rec_id}' discarded"}
            )
            continue

        seen_ids.add(rec_id)
        valid_records.append(cleaned_rec)
        report.valid_records += 1

    logger.info("Cleaning complete: %d valid, %d invalid", report.valid_records, report.invalid_records)
    return valid_records, report
