"""Data loading module for the Food Delivery Menu dataset."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "id",
    "restaurant_name",
    "dish_name",
    "description",
    "ingredients",
    "price_pkr",
    "location",
    "halal",
    "spice_level",
    "allergens",
    "category",
    "protein_g",
    "availability",
    "cuisine",
    "source",
    "source_url",
    "timestamp",
]


class MenuItem(BaseModel):
    """Pydantic model representing a normalized menu item."""

    id: str = Field(..., description="Unique dish identifier")
    restaurant_name: str
    dish_name: str
    description: str = ""
    ingredients: str = ""
    price_pkr: float = Field(..., ge=0, description="Price in Pakistani Rupees")
    location: str
    halal: bool = True
    spice_level: str = "Medium"
    allergens: List[str] = Field(default_factory=list)
    category: str = "General"
    protein_g: float = Field(default=0.0, ge=0)
    availability: bool = True
    cuisine: str = "Pakistani"
    source: str = "Restaurant Menu"
    source_url: str = ""
    timestamp: str = ""

    @field_validator("description", "ingredients", "cuisine", "source", "source_url", "timestamp", mode="before")
    @classmethod
    def parse_string_fields(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    @field_validator("spice_level", mode="before")
    @classmethod
    def parse_spice_level(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip():
            return "None"
        val = str(v).strip()
        if val.lower() == "none":
            return "None"
        return val

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
            # Split comma separated strings
            items = [item.strip().strip("'\"").lower() for item in v_clean.split(",")]
            return [i for i in items if i]
        return []


def load_raw_menu_df(file_path: Path | str = "data/menu.csv") -> pd.DataFrame:
    """Load menu dataset into a pandas DataFrame and validate required columns.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame containing the raw menu data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Menu data file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    logger.info("Loaded %d records successfully from %s", len(df), path)
    return df


def load_menu_items(file_path: Path | str = "data/menu.csv") -> List[MenuItem]:
    """Load menu data and parse records into validated MenuItem Pydantic objects.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of validated MenuItem instances.
    """
    df = load_raw_menu_df(file_path)
    records: List[MenuItem] = []

    for idx, row in df.iterrows():
        try:
            item_dict = row.to_dict()
            item = MenuItem(**item_dict)
            records.append(item)
        except Exception as e:
            logger.error("Failed to parse row %d: %s (Error: %s)", idx, row.get("id"), e)
            raise e

    logger.info("Successfully validated %d MenuItem objects.", len(records))
    return records


if __name__ == "__main__":
    items = load_menu_items()
    print(f"Loaded {len(items)} menu items. First item: {items[0].dish_name} (PKR {items[0].price_pkr})")
