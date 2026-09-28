"""ChromaDB Vector Store wrapper for indexing, storing, and semantic retrieval of menu items."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from ingestion.normalize_data import ProcessedMenuItem
from rag.config import config
from rag.embeddings import EmbeddingProvider, get_embedding_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_document_text(item: ProcessedMenuItem) -> str:
    """Construct searchable document text for embedding generation.

    Includes restaurant name, dish name, description, ingredients, category, cuisine,
    spice level, and location. Safety assumptions are strictly avoided.
    """
    parts = [
        f"Restaurant: {item.restaurant_name}",
        f"Dish: {item.dish_name}",
        f"Description: {item.description}" if item.description else "",
        f"Ingredients: {item.ingredients}" if item.ingredients else "",
        f"Category: {item.category}",
        f"Cuisine: {item.cuisine}",
        f"Spice Level: {item.spice_level}",
        f"Location: {item.location}",
    ]
    return " | ".join(p for p in parts if p)


def format_metadata_for_chroma(item: ProcessedMenuItem) -> Dict[str, Any]:
    """Format structured metadata for ChromaDB storage (primitive types only)."""
    return {
        "id": item.id,
        "restaurant_name": item.restaurant_name,
        "dish_name": item.dish_name,
        "price_pkr": float(item.price_pkr),
        "location": item.location,
        "halal": bool(item.halal) if item.halal is not None else False,
        "halal_raw": "unknown" if item.halal is None else ("true" if item.halal else "false"),
        "spice_level": item.spice_level,
        "allergens_json": json.dumps(item.allergens),
        "allergen_status": item.allergen_status,
        "category": item.category,
        "protein_g": float(item.protein_g),
        "availability": bool(item.availability),
        "cuisine": item.cuisine,
        "source": item.source,
        "source_url": item.source_url,
    }


# Process-wide shared persistent clients to prevent ChromaDB System refcount drops
# from triggering 'RustBindingsAPI' object has no attribute 'bindings'.
_shared_persistent_clients: Dict[str, chromadb.ClientAPI] = {}


def get_shared_chroma_client(db_path: Path | str) -> chromadb.ClientAPI:
    """Return or create a process-wide shared PersistentClient to prevent refcount drops."""
    resolved = str(Path(db_path).resolve())
    if resolved not in _shared_persistent_clients:
        _shared_persistent_clients[resolved] = chromadb.PersistentClient(path=resolved)
    return _shared_persistent_clients[resolved]


class VectorStore:
    """Persistent ChromaDB vector database client for semantic menu retrieval."""

    def __init__(
        self,
        db_path: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.db_path = Path(db_path or config.vector_db_path)
        self.collection_name = collection_name or config.collection_name
        self.embedding_provider = embedding_provider or get_embedding_provider()

        # Ensure storage directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing ChromaDB PersistentClient at: %s (Collection: %s)",
            self.db_path.resolve(),
            self.collection_name,
        )

        self.client = get_shared_chroma_client(self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_collection(self) -> Any:
        """Ensure underlying ChromaDB client and collection have active bindings."""
        try:
            client = getattr(self.collection, "_client", None)
            server = getattr(client, "_server", None)
            if server is not None and not hasattr(server, "bindings"):
                logger.warning(
                    "Detected stale ChromaDB bindings in VectorStore. Re-acquiring fresh client and collection..."
                )
                resolved = str(self.db_path.resolve())
                _shared_persistent_clients.pop(resolved, None)
                self.client = get_shared_chroma_client(self.db_path)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception as e:
            logger.warning("Collection validation failed (%s). Re-initializing collection...", e)
            resolved = str(self.db_path.resolve())
            _shared_persistent_clients.pop(resolved, None)
            self.client = get_shared_chroma_client(self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    def count(self) -> int:
        """Return the current document count in the collection."""
        self._ensure_collection()
        try:
            return self.collection.count()
        except AttributeError as e:
            if "bindings" in str(e):
                logger.warning("Caught bindings attribute error in count(). Healing and retrying...")
                resolved = str(self.db_path.resolve())
                _shared_persistent_clients.pop(resolved, None)
                self.client = get_shared_chroma_client(self.db_path)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return self.collection.count()
            raise

    def reset(self) -> None:
        """Delete and recreate the current collection."""
        logger.warning("Resetting collection: %s", self.collection_name)
        self._ensure_collection()
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_menu_items(
        self,
        items: List[ProcessedMenuItem],
        batch_size: Optional[int] = None,
    ) -> int:
        """Index or update menu items into the vector database in batches.

        Args:
            items: List of ProcessedMenuItem objects.
            batch_size: Batch size for embedding and upsert.

        Returns:
            Number of successfully indexed items.
        """
        if not items:
            return 0

        batch_sz = batch_size or config.batch_size
        total_items = len(items)
        num_batches = (total_items + batch_sz - 1) // batch_sz
        start_time = time.perf_counter()

        logger.info(
            "Beginning batch indexing of %d records across %d batches (Batch Size: %d, Model: %s, Dim: %d)",
            total_items,
            num_batches,
            batch_sz,
            self.embedding_provider.model_name,
            self.embedding_provider.dimension,
        )

        for i in range(0, total_items, batch_sz):
            batch = items[i : i + batch_sz]
            batch_ids = [item.id for item in batch]
            batch_docs = [build_document_text(item) for item in batch]
            batch_metadatas = [format_metadata_for_chroma(item) for item in batch]

            # Generate dense embeddings
            batch_embeddings = self.embedding_provider.embed_documents(batch_docs)

            # Stable Upsert into ChromaDB
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_metadatas,
            )
            batch_idx = (i // batch_sz) + 1
            logger.info("Indexed batch %d/%d (%d items)", batch_idx, num_batches, len(batch))

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Completed indexing %d documents in %.3f seconds (Total in DB: %d)",
            total_items,
            elapsed,
            self.count(),
        )
        return total_items

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search against the indexed menu collection.

        Args:
            query: Natural language query string.
            top_k: Number of candidates to retrieve.
            where: Optional ChromaDB metadata filter dictionary.

        Returns:
            List of candidate dictionaries ordered by relevance.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to vector search; returning empty results.")
            return []

        k = top_k or config.top_k
        k = min(k, max(1, self.count()))

        start_time = time.perf_counter()
        query_vec = self.embedding_provider.embed_query(query.strip())

        query_args: Dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where

        self._ensure_collection()
        try:
            results = self.collection.query(**query_args)
        except AttributeError as e:
            if "bindings" in str(e):
                logger.warning("Caught bindings attribute error in search(). Healing collection and retrying...")
                resolved = str(self.db_path.resolve())
                _shared_persistent_clients.pop(resolved, None)
                self.client = get_shared_chroma_client(self.db_path)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                results = self.collection.query(**query_args)
            else:
                raise
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        candidates: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            logger.info("Search returned 0 results in %.2f ms", elapsed_ms)
            return candidates

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)

        for doc_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
            
            # Reconstruct allergens list from JSON
            allergens_list = []
            if "allergens_json" in meta:
                try:
                    allergens_list = json.loads(meta["allergens_json"])
                except Exception:
                    allergens_list = []

            candidates.append({
                "id": doc_id,
                "dish_name": meta.get("dish_name", ""),
                "restaurant_name": meta.get("restaurant_name", ""),
                "price_pkr": meta.get("price_pkr", 0.0),
                "location": meta.get("location", ""),
                "halal": meta.get("halal", True),
                "spice_level": meta.get("spice_level", "Unknown"),
                "allergens": allergens_list,
                "allergen_status": meta.get("allergen_status", "UNKNOWN"),
                "category": meta.get("category", ""),
                "protein_g": meta.get("protein_g", 0.0),
                "distance": float(dist) if dist is not None else 0.0,
                "similarity": similarity,
                "metadata": meta,
                "document": doc,
            })

        logger.info(
            "Semantic search for '%s' returned %d results in %.2f ms",
            query,
            len(candidates),
            elapsed_ms,
        )
        return candidates

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics for the vector database."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "vector_db_path": str(self.db_path),
            "embedding_dimension": self.embedding_provider.dimension,
            "embedding_model": self.embedding_provider.model_name,
        }


# Alias for explicit class referencing
ChromaVectorStore = VectorStore

