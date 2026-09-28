"""Tests for data loading, parsing, and schema validation."""

from pathlib import Path
import pytest
from ingestion.load_data import MenuItem, load_menu_items, load_raw_menu_df

DATA_PATH = Path("data/menu.csv")


def test_menu_file_exists():
    assert DATA_PATH.exists(), f"Menu dataset must exist at {DATA_PATH}"


def test_load_raw_menu_df():
    df = load_raw_menu_df(DATA_PATH)
    assert not df.empty
    assert len(df) >= 30, "Expected at least 30 menu items in demo dataset"
    assert "dish_name" in df.columns
    assert "price_pkr" in df.columns
    assert "allergens" in df.columns
    assert "halal" in df.columns


def test_load_menu_items_pydantic_validation():
    items = load_menu_items(DATA_PATH)
    assert len(items) >= 30
    
    # Check first item
    first = items[0]
    assert isinstance(first, MenuItem)
    assert first.id == "dish_001"
    assert first.price_pkr > 0
    assert first.halal is True
    assert isinstance(first.allergens, list)


def test_allergens_parsing():
    items = load_menu_items(DATA_PATH)
    items_by_id = {item.id: item for item in items}
    
    # dish_001 has no allergens
    assert items_by_id["dish_001"].allergens == []
    
    # dish_002 has dairy and tree nuts
    assert "dairy" in items_by_id["dish_002"].allergens
    assert "tree nuts" in items_by_id["dish_002"].allergens
    
    # dish_004 has peanuts and soy
    assert "peanuts" in items_by_id["dish_004"].allergens
    assert "soy" in items_by_id["dish_004"].allergens


def test_halal_parsing():
    items = load_menu_items(DATA_PATH)
    items_by_id = {item.id: item for item in items}
    
    # dish_006 is non-Halal pepperoni pizza
    assert items_by_id["dish_006"].halal is False
    
    # Normal Pakistani dishes should be Halal
    assert items_by_id["dish_001"].halal is True


def test_full_ingestion_pipeline_run(tmp_path):
    from ingestion.build_index import run_ingestion_pipeline
    import pandas as pd

    test_out = tmp_path / "menu_cleaned_test.csv"
    items, report = run_ingestion_pipeline(raw_path=DATA_PATH, output_path=test_out)

    assert len(items) == 35
    assert report.total_raw_records == 35
    assert report.valid_cleaned_records == 35
    assert report.invalid_records == 0
    assert report.duplicate_ids_count == 0
    assert test_out.exists()

    # Verify CSV columns in output file
    df_out = pd.read_csv(test_out)
    expected_cols = [
        "id", "restaurant_name", "dish_name", "description", "ingredients",
        "price_pkr", "location", "halal", "spice_level", "allergens",
        "allergen_status", "category", "protein_g", "availability",
        "cuisine", "source", "source_url", "timestamp", "search_text"
    ]
    for col in expected_cols:
        assert col in df_out.columns, f"Expected column {col} in processed output"

    # Check that search_text is populated
    assert df_out["search_text"].str.len().min() > 10

