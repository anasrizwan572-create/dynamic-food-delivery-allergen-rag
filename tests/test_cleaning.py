"""Unit tests for data cleaning and duplicate detection."""

import pytest
from ingestion.clean_data import (
    clean_and_validate_records,
    clean_string,
    detect_duplicate_ids,
    validate_raw_record,
)


def test_clean_string():
    assert clean_string("  Gulberg  ") == "Gulberg"
    assert clean_string(None) == ""
    assert clean_string("nan") == ""
    assert clean_string("NULL") == ""
    assert clean_string("None") == ""


def test_detect_duplicate_ids():
    records = [
        {"id": "dish_001", "dish_name": "Dish 1"},
        {"id": "dish_002", "dish_name": "Dish 2"},
        {"id": "dish_001", "dish_name": "Dish 1 Duplicate"},
        {"id": "dish_003", "dish_name": "Dish 3"},
        {"id": "dish_002", "dish_name": "Dish 2 Duplicate"},
    ]
    duplicates = detect_duplicate_ids(records)
    assert duplicates == ["dish_001", "dish_002"]


def test_validate_raw_record_missing_id():
    rec = {"dish_name": "Chicken Karahi", "restaurant_name": "Shinwari"}
    is_valid, err = validate_raw_record(rec)
    assert not is_valid
    assert "Missing or empty 'id'" in err


def test_validate_raw_record_missing_dish_name():
    rec = {"id": "dish_999", "restaurant_name": "Shinwari", "dish_name": " "}
    is_valid, err = validate_raw_record(rec)
    assert not is_valid
    assert "dish_name" in err


def test_validate_raw_record_missing_restaurant():
    rec = {"id": "dish_999", "restaurant_name": "", "dish_name": "Chicken Karahi"}
    is_valid, err = validate_raw_record(rec)
    assert not is_valid
    assert "restaurant_name" in err


def test_clean_and_validate_records_pipeline():
    records = [
        {"id": "dish_101", "restaurant_name": "Test Rest 1", "dish_name": "Item 1"},
        {"id": "dish_102", "restaurant_name": "Test Rest 2", "dish_name": "Item 2"},
        {"id": "dish_101", "restaurant_name": "Test Rest 1", "dish_name": "Item 1 Duplicate"},
        {"id": "dish_103", "restaurant_name": "", "dish_name": "Item 3 No Restaurant"},
    ]
    cleaned, report = clean_and_validate_records(records)
    assert report.total_records == 4
    assert report.valid_records == 2
    assert report.invalid_records == 2
    assert report.duplicate_ids == ["dish_101"]
    assert len(cleaned) == 2
    assert cleaned[0]["id"] == "dish_101"
    assert cleaned[1]["id"] == "dish_102"
