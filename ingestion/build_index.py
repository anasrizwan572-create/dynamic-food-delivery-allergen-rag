"""Ingestion and Vector Indexing orchestrator.

Loads processed menu records, creates searchable document representations,
computes dense vector embeddings, and populates the persistent ChromaDB index.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ingestion.clean_data import clean_and_validate_records, DataCleaningReport
from ingestion.load_data import load_raw_menu_df
from ingestion.normalize_data import ProcessedMenuItem, normalize_record
from rag.config import config
from rag.embeddings import EmbeddingProvider, get_embedding_provider
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/menu.csv")
PROCESSED_DATA_PATH = Path("data/processed/menu_cleaned.csv")


class IngestionReport:
    """Detailed summary of the complete ingestion and vector indexing pipeline execution."""

    def __init__(self):
        self.total_raw_records: int = 0
        self.valid_cleaned_records: int = 0
        self.invalid_records: int = 0
        self.duplicate_ids_count: int = 0
        self.duplicate_ids: List[str] = []
        self.known_allergens_count: int = 0
        self.known_no_allergens_count: int = 0
        self.unknown_allergens_count: int = 0
        self.conflict_allergens_count: int = 0
        self.halal_true_count: int = 0
        self.halal_false_count: int = 0
        self.halal_unknown_count: int = 0
        self.min_price: float = 0.0
        self.max_price: float = 0.0
        self.avg_price: float = 0.0
        self.output_file: str = ""
        # Vector Index stats
        self.indexed_vector_count: int = 0
        self.embedding_model: str = ""
        self.embedding_dimension: int = 0
        self.indexing_duration_seconds: float = 0.0
        self.vector_db_path: str = ""

    def summary(self) -> str:
        return (
            "==================================================\n"
            "       INGESTION & VECTOR INDEXING REPORT         \n"
            "==================================================\n"
            f"Total Raw Records:           {self.total_raw_records}\n"
            f"Valid Processed Records:     {self.valid_cleaned_records}\n"
            f"Invalid Records:             {self.invalid_records}\n"
            f"Duplicate IDs:               {self.duplicate_ids_count} {self.duplicate_ids}\n"
            "--------------------------------------------------\n"
            "Allergen Verification Status Breakdown:\n"
            f"  - KNOWN_ALLERGENS:         {self.known_allergens_count}\n"
            f"  - KNOWN_NO_ALLERGENS:      {self.known_no_allergens_count}\n"
            f"  - UNKNOWN:                 {self.unknown_allergens_count}\n"
            f"  - CONFLICT:                {self.conflict_allergens_count}\n"
            "--------------------------------------------------\n"
            "Halal Certification Breakdown:\n"
            f"  - Halal (True):            {self.halal_true_count}\n"
            f"  - Non-Halal (False):       {self.halal_false_count}\n"
            f"  - Unverified (Unknown):    {self.halal_unknown_count}\n"
            "--------------------------------------------------\n"
            f"Price Distribution (PKR):    Min: {self.min_price:.1f} | Max: {self.max_price:.1f} | Avg: {self.avg_price:.1f}\n"
            f"Processed Dataset File:      {self.output_file}\n"
            "--------------------------------------------------\n"
            "Vector Database Indexing:\n"
            f"  - Vector DB Path:          {self.vector_db_path}\n"
            f"  - Embedding Model:         {self.embedding_model}\n"
            f"  - Embedding Dimension:     {self.embedding_dimension}\n"
            f"  - Documents Indexed:       {self.indexed_vector_count}\n"
            f"  - Indexing Time:           {self.indexing_duration_seconds:.3f} s\n"
            "=================================================="
        )


def run_ingestion_pipeline(
    raw_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = PROCESSED_DATA_PATH,
    index_vectors: bool = True,
    rebuild_index: bool = False,
    vector_store: Optional[VectorStore] = None,
) -> Tuple[List[ProcessedMenuItem], IngestionReport]:
    """Execute complete ingestion pipeline:
    RAW CSV -> Load -> Clean & Validate -> Normalize -> Save Processed Dataset
            -> (Optional) Generate Embeddings & Index in Vector DB.

    Args:
        raw_path: Path to the raw CSV dataset.
        output_path: Target path for the processed output CSV.
        index_vectors: If True, index records into the vector database.
        rebuild_index: If True, reset and rebuild the vector collection.
        vector_store: Optional VectorStore instance to use.

    Returns:
        Tuple of (list_of_processed_items, ingestion_report).
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    report = IngestionReport()

    logger.info("Starting ingestion pipeline from: %s", raw_path.resolve())

    # 1. Load raw data
    raw_df = load_raw_menu_df(raw_path)
    raw_records = raw_df.to_dict(orient="records")
    report.total_raw_records = len(raw_records)

    # 2. Clean and validate records
    cleaned_records, cleaning_report = clean_and_validate_records(raw_records)
    report.invalid_records = cleaning_report.invalid_records
    report.duplicate_ids = cleaning_report.duplicate_ids
    report.duplicate_ids_count = len(cleaning_report.duplicate_ids)

    # 3. Normalize records
    processed_items: List[ProcessedMenuItem] = []
    prices: List[float] = []

    for rec in cleaned_records:
        try:
            item = normalize_record(rec)
            processed_items.append(item)
            prices.append(item.price_pkr)

            if item.allergen_status == "KNOWN_ALLERGENS":
                report.known_allergens_count += 1
            elif item.allergen_status == "KNOWN_NO_ALLERGENS":
                report.known_no_allergens_count += 1
            elif item.allergen_status == "UNKNOWN":
                report.unknown_allergens_count += 1
            elif item.allergen_status == "CONFLICT":
                report.conflict_allergens_count += 1

            if item.halal is True:
                report.halal_true_count += 1
            elif item.halal is False:
                report.halal_false_count += 1
            else:
                report.halal_unknown_count += 1

        except Exception as e:
            logger.error("Failed to normalize record %s: %s", rec.get("id"), e)
            report.invalid_records += 1

    report.valid_cleaned_records = len(processed_items)

    if prices:
        report.min_price = min(prices)
        report.max_price = max(prices)
        report.avg_price = sum(prices) / len(prices)

    # 4. Save processed dataset to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_save = []
    for item in processed_items:
        d = item.model_dump()
        d["allergens"] = json.dumps(d["allergens"])
        rows_to_save.append(d)

    fieldnames = list(rows_to_save[0].keys()) if rows_to_save else []
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows_to_save:
            writer.writerow(row)

    report.output_file = str(output_path.resolve())
    logger.info("Saved %d normalized records to %s", len(rows_to_save), report.output_file)

    # 5. Build Vector Index (Milestone 3)
    if index_vectors:
        vstore = vector_store or VectorStore()
        if rebuild_index:
            vstore.reset()

        idx_start = time.perf_counter()
        vstore.upsert_menu_items(processed_items, batch_size=config.batch_size)
        idx_duration = time.perf_counter() - idx_start

        report.indexed_vector_count = vstore.count()
        report.embedding_model = vstore.embedding_provider.model_name
        report.embedding_dimension = vstore.embedding_provider.dimension
        report.indexing_duration_seconds = idx_duration
        report.vector_db_path = str(vstore.db_path)

    return processed_items, report


def load_cleaned_records(csv_path: Path | str = PROCESSED_DATA_PATH) -> List[ProcessedMenuItem]:
    """Load pre-processed records directly from the cleaned CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)
    items: List[ProcessedMenuItem] = []
    for _, row in df.iterrows():
        items.append(ProcessedMenuItem(**row.to_dict()))
    return items


def main():
    parser = argparse.ArgumentParser(description="Ingest menu dataset and build ChromaDB vector index.")
    parser.add_argument("--rebuild", action="store_true", help="Reset and rebuild the vector collection from scratch.")
    parser.add_argument("--no-index", action="store_true", help="Skip vector indexing (ingest & clean only).")
    args = parser.parse_args()

    _, report = run_ingestion_pipeline(
        index_vectors=not args.no_index,
        rebuild_index=args.rebuild,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
