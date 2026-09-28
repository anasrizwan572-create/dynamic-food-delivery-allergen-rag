"""Configuration settings for RAG components: Vector DB, Embeddings, and Retrieval."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if available
load_dotenv()


class RAGConfig(BaseModel):
    """Runtime configuration for embeddings, vector store, and search."""

    # Embeddings
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local").lower()
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dimension: int = Field(default=384)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", None)
    )

    # Vector Database
    vector_db_path: Path = Field(
        default_factory=lambda: Path(os.getenv("VECTOR_DB_PATH", "data/vector_db"))
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "food_menu_rag")
    )
    similarity_metric: str = Field(default="cosine")
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "16"))
    )

    # Retrieval & Hybrid Settings
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K", "10"))
    )
    retrieval_mode: str = Field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    )
    bm25_k1: float = Field(
        default_factory=lambda: float(os.getenv("BM25_K1", "1.5"))
    )
    bm25_b: float = Field(
        default_factory=lambda: float(os.getenv("BM25_B", "0.75"))
    )
    rrf_k: int = Field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    data_processed_csv: Path = Field(
        default_factory=lambda: Path(os.getenv("DATA_PROCESSED_CSV", "data/processed/menu_cleaned.csv"))
    )

    # Reranking Settings (Milestone 7)
    reranker_type: str = Field(
        default_factory=lambda: os.getenv("RERANKER_TYPE", "cross_encoder").lower()
    )
    cross_encoder_model: str = Field(
        default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    reranker_candidate_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_CANDIDATE_K", "15"))
    )
    reranker_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "5"))
    )

    # Logging
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


# Global default configuration instance
config = RAGConfig()
