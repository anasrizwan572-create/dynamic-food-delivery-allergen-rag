"""BM25 Lexical Retrieval Module for Fast Keyword and Exact-Match Search."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from rank_bm25 import BM25Okapi

from ingestion.load_data import MenuItem
from rag.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Tokenize a string into lowercased alphanumeric tokens for BM25.

    Splits on punctuation, whitespace, and non-alphanumeric characters.
    """
    if not text:
        return []
    return [token.lower() for token in re.findall(r"\w+", text) if token]


class BM25Index:
    """Deterministic BM25 lexical search index over processed menu items."""

    def __init__(
        self,
        csv_path: Optional[Path | str] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ):
        self.csv_path = Path(csv_path or config.data_processed_csv)
        self.k1 = k1 if k1 is not None else config.bm25_k1
        self.b = b if b is not None else config.bm25_b

        self.documents: List[Dict[str, Any]] = []
        self.id_to_doc: Dict[str, Dict[str, Any]] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self._build_index()

    def _build_index(self) -> None:
        """Load processed menu items and build the Okapi BM25 index."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Processed menu CSV not found at: {self.csv_path.resolve()}. "
                "Run ingestion first: python -m ingestion.build_index"
            )

        df = pd.read_csv(self.csv_path)
        logger.info("Building BM25 index from %s (%d records)...", self.csv_path, len(df))

        docs: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        for _, row in df.iterrows():
            doc_id = str(row["id"]).strip()

            # Parse allergens safely
            allergens_val = row.get("allergens", [])
            allergens_list = MenuItem.parse_allergens(allergens_val)

            # Parse Halal boolean safely
            halal_val = row.get("halal")
            if pd.isna(halal_val):
                halal_bool = None
            elif isinstance(halal_val, bool):
                halal_bool = halal_val
            elif str(halal_val).strip().lower() in ["true", "1", "yes"]:
                halal_bool = True
            elif str(halal_val).strip().lower() in ["false", "0", "no"]:
                halal_bool = False
            else:
                halal_bool = None

            metadata: Dict[str, Any] = {
                "id": doc_id,
                "restaurant_name": str(row.get("restaurant_name", "")).strip(),
                "dish_name": str(row.get("dish_name", "")).strip(),
                "description": str(row.get("description", "")).strip() if pd.notna(row.get("description")) else "",
                "ingredients": str(row.get("ingredients", "")).strip() if pd.notna(row.get("ingredients")) else "",
                "price_pkr": float(row.get("price_pkr", 0.0)),
                "location": str(row.get("location", "")).strip(),
                "halal": halal_bool,
                "halal_raw": "unknown" if halal_bool is None else ("true" if halal_bool else "false"),
                "spice_level": str(row.get("spice_level", "Medium")).strip(),
                "allergens": allergens_list,
                "allergens_json": json.dumps(allergens_list),
                "allergen_status": str(row.get("allergen_status", "UNKNOWN")).strip(),
                "category": str(row.get("category", "")).strip(),
                "protein_g": float(row.get("protein_g", 0.0)),
                "availability": bool(row.get("availability", True)),
                "cuisine": str(row.get("cuisine", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "source_url": str(row.get("source_url", "")).strip(),
            }

            # Rich text combining dish name, restaurant, description, ingredients, category, cuisine, and spice
            doc_text = (
                f"Restaurant: {metadata['restaurant_name']} | "
                f"Dish: {metadata['dish_name']} | "
                f"Description: {metadata['description']} | "
                f"Ingredients: {metadata['ingredients']} | "
                f"Category: {metadata['category']} | "
                f"Cuisine: {metadata['cuisine']} | "
                f"Spice Level: {metadata['spice_level']} | "
                f"Location: {metadata['location']}"
            )

            # Tokens for lexical indexing: combine doc_text and original search_text if available
            combined_text = doc_text + " " + str(row.get("search_text", ""))
            tokens = tokenize(combined_text)

            doc_struct: Dict[str, Any] = {
                "id": doc_id,
                "dish_name": metadata["dish_name"],
                "restaurant_name": metadata["restaurant_name"],
                "price_pkr": metadata["price_pkr"],
                "location": metadata["location"],
                "halal": metadata["halal"],
                "spice_level": metadata["spice_level"],
                "allergens": allergens_list,
                "allergen_status": metadata["allergen_status"],
                "metadata": metadata,
                "document_text": doc_text,
            }

            docs.append(doc_struct)
            tokenized_corpus.append(tokens)
            self.id_to_doc[doc_id] = doc_struct

        self.documents = docs
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(
            "BM25 index built successfully with %d documents (k1=%.2f, b=%.2f).",
            len(self.documents),
            self.k1,
            self.b,
        )

    def count(self) -> int:
        """Return the number of indexed documents."""
        return len(self.documents)

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full document struct by dish ID."""
        return self.id_to_doc.get(doc_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return a copy of all indexed documents."""
        return [dict(d) for d in self.documents]

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the BM25 index for exact token matches.

        Args:
            query: Query string.
            top_k: Number of results to return.
            allowed_ids: Optional set of allowed dish IDs (from upstream pre-filtering).

        Returns:
            List of matching documents sorted by BM25 score descending,
            with deterministic tie-breaking on dish ID.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to BM25 search.")
            return []

        tokens = tokenize(query)
        if not tokens or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokens)

        # Collect matching candidates with score > 0
        candidates: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue

            doc = self.documents[idx]
            doc_id = doc["id"]

            # Enforce upstream allowed IDs if provided
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue

            item = dict(doc)
            item["score"] = float(score)
            item["bm25_score"] = float(score)
            candidates.append(item)

        # Sort by score descending, tie-break by ID ascending for deterministic output
        candidates.sort(key=lambda x: (-x["score"], x["id"]))

        k = top_k or config.top_k
        return candidates[:k]


# Singleton default BM25 index
_default_bm25_index: Optional[BM25Index] = None


def get_default_bm25_index() -> BM25Index:
    """Return the cached default BM25 index instance."""
    global _default_bm25_index
    if _default_bm25_index is None:
        _default_bm25_index = BM25Index()
    return _default_bm25_index
