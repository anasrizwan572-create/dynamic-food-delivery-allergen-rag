"""Pydantic schemas for the FastAPI REST application."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request payload for POST /query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language search or recommendation query.",
        examples=["Find Halal chicken under PKR 1500 without dairy in Gulberg"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of candidate dishes to retrieve and return.",
    )
    max_price_pkr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional explicit maximum budget ceiling in Pakistani Rupees.",
    )
    min_price_pkr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional explicit minimum budget in Pakistani Rupees.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Optional explicit location or delivery zone (e.g. Gulberg, DHA).",
    )
    halal: Optional[bool] = Field(
        default=None,
        description="Optional explicit Halal compliance requirement.",
    )
    excluded_allergens: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit list of allergens to exclude.",
    )
    spice_level: Optional[str] = Field(
        default=None,
        description="Optional spice level preference (e.g. Mild, Medium, High).",
    )

    @field_validator("query")
    @classmethod
    def validate_non_whitespace_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or pure whitespace.")
        return v.strip()


class DishResult(BaseModel):
    """Structured menu item recommendation returned in query results."""

    dish_name: str
    restaurant_name: str
    price_pkr: float
    location: str
    halal: bool
    spice_level: str
    allergens: List[str] = Field(default_factory=list)
    allergen_status: str = "UNKNOWN"
    match_reason: str
    source: str
    source_id: str
    reranker_score: Optional[float] = None
    verification_status: Optional[str] = None
    warning: Optional[str] = None


class CitationItem(BaseModel):
    """Structured citation metadata linking claims to verified source evidence."""

    source_id: str
    dish_name: str
    restaurant_name: str
    price_pkr: float
    location: str
    halal: bool
    spice_level: str
    allergens: List[str] = Field(default_factory=list)
    allergen_status: str
    source_menu: str
    match_reason: str
    reranker_score: Optional[float] = None
    verification_status: Optional[str] = None
    warning: Optional[str] = None


class QueryResponse(BaseModel):
    """Comprehensive response payload for POST /query endpoint."""

    status: str = Field(
        ...,
        description="Query outcome: SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, NO_MATCHING_ITEMS",
    )
    query: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    citations: List[CitationItem] = Field(default_factory=list)
    results: List[DishResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parsed_constraints: Dict[str, Any] = Field(default_factory=dict)
    filter_summary: Optional[str] = None
    latency: Dict[str, float] = Field(default_factory=dict)
    disclaimer: Optional[str] = None


class HealthResponse(BaseModel):
    """System health check payload for GET /health endpoint."""

    status: str = "healthy"
    version: str = "1.0.0"
    vector_db_ready: bool = True
    bm25_ready: bool = True
    reranker_ready: bool = True
    allergen_verifier_ready: bool = True
    total_indexed_items: int = 0


class MenuItemSummary(BaseModel):
    """Menu item summary record for GET /menu endpoint."""

    id: str
    dish_name: str
    restaurant_name: str
    price_pkr: float
    location: str
    halal: bool
    spice_level: str
    allergens: List[str] = Field(default_factory=list)
    allergen_status: str
    category: str
    protein_g: float
    availability: bool
    cuisine: str


class MenuListResponse(BaseModel):
    """Menu collection payload for GET /menu endpoint."""

    total: int
    count: int
    items: List[MenuItemSummary]


class MetricsResponse(BaseModel):
    """Telemetry metrics payload for GET /metrics endpoint."""

    collection_count: int
    embedding_dimension: int
    retrieval_mode: str
    default_top_k: int
