"""Unit and integration tests for VectorStore, Embeddings, and Semantic Retrieval."""

from pathlib import Path
import pytest
from ingestion.build_index import load_cleaned_records
from ingestion.normalize_data import ProcessedMenuItem
from rag.embeddings import (
    DeterministicDenseEmbeddingProvider,
    get_embedding_provider,
)
from rag.vector_store import (
    VectorStore,
    build_document_text,
    format_metadata_for_chroma,
)

SAMPLE_ITEM_1 = ProcessedMenuItem(
    id="test_001",
    restaurant_name="Monal Express",
    dish_name="Grilled Chicken Breast Bowl",
    description="Char-grilled chicken breast with steamed rice",
    ingredients="chicken breast, olive oil, lemon, cumin, rice",
    price_pkr=1150.0,
    location="Gulberg",
    halal=True,
    spice_level="Mild",
    allergens=[],
    allergen_status="KNOWN_NO_ALLERGENS",
    category="Chicken",
    protein_g=42.0,
    availability=True,
    cuisine="Pakistani",
    source="Test Menu",
    source_url="https://test.pk/001",
    timestamp="2026-09-28T00:00:00Z",
    search_text="Monal Express Grilled Chicken Breast Bowl",
)

SAMPLE_ITEM_2 = ProcessedMenuItem(
    id="test_002",
    restaurant_name="Johnny & Jugnu",
    dish_name="Gourmet Smash Beef Burger",
    description="Double beef patties with melted cheddar cheese",
    ingredients="beef, cheese, brioche bun, mayo",
    price_pkr=1050.0,
    location="Johar Town",
    halal=True,
    spice_level="Medium",
    allergens=["gluten", "dairy", "eggs"],
    allergen_status="KNOWN_ALLERGENS",
    category="Fast Food",
    protein_g=31.0,
    availability=True,
    cuisine="Fast Food",
    source="Test Menu",
    source_url="https://test.pk/002",
    timestamp="2026-09-28T00:00:00Z",
    search_text="Johnny & Jugnu Gourmet Smash Beef Burger",
)


# =====================================================================
# 1. Embedding Provider Initialization Tests
# =====================================================================

def test_embedding_provider_initialization():
    provider = get_embedding_provider(provider_name="deterministic")
    assert provider.dimension == 384
    assert "dense" in provider.model_name.lower() or "local" in provider.model_name.lower()

    vec = provider.embed_query("Chicken Karahi")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert any(v != 0.0 for v in vec)

    batch_vecs = provider.embed_documents(["Dish 1", "Dish 2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384


# =====================================================================
# 2. Document Text Creation Tests
# =====================================================================

def test_document_creation():
    doc_text = build_document_text(SAMPLE_ITEM_1)
    assert "Restaurant: Monal Express" in doc_text
    assert "Dish: Grilled Chicken Breast Bowl" in doc_text
    assert "Ingredients: chicken breast, olive oil, lemon, cumin, rice" in doc_text
    assert "Location: Gulberg" in doc_text
    assert "Spice Level: Mild" in doc_text
    # Invariant: No safety assumptions injected
    assert "safe" not in doc_text.lower()
    assert "allergen-free" not in doc_text.lower()


# =====================================================================
# 3. Metadata Preservation Tests
# =====================================================================

def test_metadata_preservation():
    meta = format_metadata_for_chroma(SAMPLE_ITEM_1)
    assert meta["id"] == "test_001"
    assert meta["restaurant_name"] == "Monal Express"
    assert meta["price_pkr"] == 1150.0
    assert meta["location"] == "Gulberg"
    assert meta["halal"] is True
    assert meta["allergen_status"] == "KNOWN_NO_ALLERGENS"
    assert meta["allergens_json"] == "[]"
    assert meta["protein_g"] == 42.0

    meta_2 = format_metadata_for_chroma(SAMPLE_ITEM_2)
    assert meta_2["allergen_status"] == "KNOWN_ALLERGENS"
    assert "dairy" in meta_2["allergens_json"]


# =====================================================================
# 4. Vector Database Initialization Tests (Isolated Test DB)
# =====================================================================

def test_vector_database_initialization(tmp_path):
    test_db_dir = tmp_path / "chroma_test"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_food_menu",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    assert vstore.count() == 0
    stats = vstore.get_stats()
    assert stats["collection_name"] == "test_food_menu"
    assert stats["document_count"] == 0
    assert stats["embedding_dimension"] == 384


# =====================================================================
# 5. Document Insertion & Upsert Tests
# =====================================================================

def test_document_insertion(tmp_path):
    test_db_dir = tmp_path / "chroma_test_insert"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_insert",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    inserted = vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2], batch_size=2)
    assert inserted == 2
    assert vstore.count() == 2


# =====================================================================
# 6. Stable IDs and Re-index Idempotency Tests
# =====================================================================

def test_stable_ids_and_upsert_idempotency(tmp_path):
    test_db_dir = tmp_path / "chroma_test_upsert"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_upsert",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2])
    assert vstore.count() == 2

    # Re-indexing the exact same items must NOT create duplicates
    vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2])
    assert vstore.count() == 2


# =====================================================================
# 7. Semantic Search Tests
# =====================================================================

def test_semantic_search(tmp_path):
    test_db_dir = tmp_path / "chroma_test_search"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_search",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2])

    results = vstore.search("grilled chicken in Gulberg", top_k=2)
    assert len(results) > 0
    first = results[0]
    assert first["id"] == "test_001"
    assert first["dish_name"] == "Grilled Chicken Breast Bowl"
    assert first["restaurant_name"] == "Monal Express"
    assert first["similarity"] > 0.0
    assert "metadata" in first
    assert "document" in first


# =====================================================================
# 8. Empty Query Handling
# =====================================================================

def test_empty_query_handling(tmp_path):
    test_db_dir = tmp_path / "chroma_test_empty"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_empty",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1])
    assert vstore.search("") == []
    assert vstore.search("   ") == []


# =====================================================================
# 9. Collection Statistics Tests
# =====================================================================

def test_collection_statistics(tmp_path):
    test_db_dir = tmp_path / "chroma_test_stats"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_stats",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1])
    stats = vstore.get_stats()
    assert stats["document_count"] == 1
    assert stats["collection_name"] == "test_stats"
    assert stats["embedding_dimension"] == 384
    assert "chroma_test_stats" in stats["vector_db_path"]


# =====================================================================
# 10. Rebuild / Reset Behavior Tests
# =====================================================================

def test_rebuild_behavior(tmp_path):
    test_db_dir = tmp_path / "chroma_test_rebuild"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_rebuild",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2])
    assert vstore.count() == 2

    # Reset clears the collection
    vstore.reset()
    assert vstore.count() == 0

    # Can re-insert cleanly
    vstore.upsert_menu_items([SAMPLE_ITEM_1])
    assert vstore.count() == 1


# =====================================================================
# 11. Metadata Filtering Tests
# =====================================================================

def test_metadata_filtering(tmp_path):
    test_db_dir = tmp_path / "chroma_test_filter"
    vstore = VectorStore(
        db_path=test_db_dir,
        collection_name="test_filter",
        embedding_provider=DeterministicDenseEmbeddingProvider(),
    )
    vstore.upsert_menu_items([SAMPLE_ITEM_1, SAMPLE_ITEM_2])

    # Filter for Gulberg only
    gulberg_results = vstore.search("food", top_k=5, where={"location": "Gulberg"})
    assert len(gulberg_results) == 1
    assert gulberg_results[0]["id"] == "test_001"
    assert gulberg_results[0]["location"] == "Gulberg"

    # Filter for Johar Town only
    johar_results = vstore.search("food", top_k=5, where={"location": "Johar Town"})
    assert len(johar_results) == 1
    assert johar_results[0]["id"] == "test_002"
    assert johar_results[0]["location"] == "Johar Town"
