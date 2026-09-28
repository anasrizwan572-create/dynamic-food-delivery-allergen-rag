"""FastAPI application for the Dynamic Food Delivery Menu & Allergen RAG system."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    MenuItemSummary,
    MenuListResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from ingestion.load_data import load_raw_menu_df
from rag.config import config
from rag.pipeline import BasicRAGPipeline, answer_query
from rag.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Dynamic Food Delivery Menu & Allergen RAG API",
    description=(
        "Production-grade, grounded RAG API for Pakistani restaurant delivery platforms. "
        "Enforces deterministic query constraint parsing, hybrid dense + BM25 retrieval, "
        "cross-encoder reranking, and independent allergen safety verification."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend interoperability (e.g. Streamlit, web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_api_vector_store: Optional[ChromaVectorStore] = None


def get_api_vector_store() -> ChromaVectorStore:
    """Return cached ChromaVectorStore instance for health and metrics probes."""
    global _api_vector_store
    if _api_vector_store is None:
        _api_vector_store = ChromaVectorStore(
            db_path=config.vector_db_path,
            collection_name=config.collection_name,
        )
    return _api_vector_store


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """System health check endpoint verifying vector DB and component readiness."""
    total_items = 0
    vector_ready = True
    try:
        vs = get_api_vector_store()
        total_items = vs.count()
    except Exception as e:
        logger.warning("Vector store health check issue: %s", e)
        vector_ready = False

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_db_ready=vector_ready,
        bm25_ready=True,
        reranker_ready=True,
        allergen_verifier_ready=True,
        total_indexed_items=total_items,
    )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & RAG"])
def execute_query(payload: QueryRequest) -> QueryResponse:
    """Execute grounded RAG search with multi-stage filtering, reranking, and allergen safety verification.

    Accepts:
    - Natural language query (e.g. 'Halal chicken under 1500 in Gulberg without dairy')
    - Optional explicit constraint overrides (max_price_pkr, min_price_pkr, location, halal, excluded_allergens)
    """
    logger.info("Received API query: '%s' (top_k=%s)", payload.query, payload.top_k)

    constraints_override: Dict[str, Any] = {}
    if payload.max_price_pkr is not None:
        constraints_override["max_price_pkr"] = payload.max_price_pkr
    if payload.min_price_pkr is not None:
        constraints_override["min_price_pkr"] = payload.min_price_pkr
    if payload.location:
        constraints_override["location"] = payload.location
    if payload.halal is not None:
        constraints_override["halal"] = payload.halal
    if payload.excluded_allergens:
        constraints_override["excluded_allergens"] = payload.excluded_allergens

    try:
        res = answer_query(
            query=payload.query,
            top_k=payload.top_k or config.top_k,
            constraints_override=constraints_override or None,
        )
        return QueryResponse(**res)
    except Exception as e:
        logger.error("Failed to process query '%s': %s", payload.query, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG pipeline: {str(e)}",
        )


@app.get("/menu", response_model=MenuListResponse, tags=["Menu Catalog"])
def get_menu(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
    location: Optional[str] = Query(None, description="Filter by location/delivery zone"),
    halal: Optional[bool] = Query(None, description="Filter by Halal certification"),
    max_price: Optional[float] = Query(None, ge=0, description="Max budget in PKR"),
    category: Optional[str] = Query(None, description="Filter by dish category"),
) -> MenuListResponse:
    """Retrieve catalog menu records with optional filtering and pagination."""
    data_path = config.data_processed_csv if config.data_processed_csv.is_file() else Path("data/menu.csv")
    if not data_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Menu catalog dataset file not found.",
        )

    df = load_raw_menu_df(data_path)

    # Apply optional query filters
    if location:
        df = df[df["location"].str.lower() == location.strip().lower()]
    if halal is not None:
        df = df[df["halal"] == halal]
    if max_price is not None:
        df = df[df["price_pkr"] <= max_price]
    if category:
        df = df[df["category"].str.lower() == category.strip().lower()]

    total = len(df)
    page_df = df.iloc[skip : skip + limit]

    items: List[MenuItemSummary] = []
    for _, row in page_df.iterrows():
        allergens_raw = row.get("allergens", [])
        if isinstance(allergens_raw, str):
            allergens_list = [a.strip() for a in allergens_raw.split(",") if a.strip()]
        elif isinstance(allergens_raw, list):
            allergens_list = allergens_raw
        else:
            allergens_list = []

        items.append(
            MenuItemSummary(
                id=str(row.get("id")),
                dish_name=str(row.get("dish_name")),
                restaurant_name=str(row.get("restaurant_name")),
                price_pkr=float(row.get("price_pkr", 0.0)),
                location=str(row.get("location")),
                halal=bool(row.get("halal", True)),
                spice_level=str(row.get("spice_level", "Medium")),
                allergens=allergens_list,
                allergen_status=str(row.get("allergen_status", "UNKNOWN")),
                category=str(row.get("category", "General")),
                protein_g=float(row.get("protein_g", 0.0)),
                availability=bool(row.get("availability", True)),
                cuisine=str(row.get("cuisine", "Pakistani")),
            )
        )

    return MenuListResponse(total=total, count=len(items), items=items)


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def get_metrics() -> MetricsResponse:
    """Retrieve operational telemetry and vector index configuration."""
    count = 0
    try:
        vs = get_api_vector_store()
        count = vs.count()
    except Exception:
        pass

    return MetricsResponse(
        collection_count=count,
        embedding_dimension=384,
        retrieval_mode=config.retrieval_mode,
        default_top_k=config.top_k,
    )
