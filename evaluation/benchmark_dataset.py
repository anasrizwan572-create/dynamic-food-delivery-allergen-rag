"""Synthetic benchmark dataset for reproducible evaluation of the Food Delivery RAG system.

All evaluation cases are representative synthetic test benchmarks designed to rigorously
evaluate retrieval accuracy, hard constraint filtering, allergen safety, citation fidelity,
and conservative handling of unverified/conflicting records.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BenchmarkTestCase(BaseModel):
    """Structured test benchmark scenario for RAG evaluation."""

    query_id: str
    query: str
    category: str
    expected_status: str = "SUPPORTED"
    max_price_pkr: Optional[float] = None
    min_price_pkr: Optional[float] = None
    expected_location: Optional[str] = None
    expected_halal: Optional[bool] = None
    excluded_allergens: List[str] = Field(default_factory=list)
    spice_level: Optional[str] = None
    target_dish_ids: List[str] = Field(default_factory=list)
    forbidden_dish_ids: List[str] = Field(default_factory=list)
    must_contain_warning: Optional[str] = None
    description: str = ""


BENCHMARK_CASES: List[BenchmarkTestCase] = [
    BenchmarkTestCase(
        query_id="BM_001",
        query="Find Halal chicken under PKR 1500 without dairy in Gulberg",
        category="multi_constraint",
        expected_status="SUPPORTED",
        max_price_pkr=1500.0,
        expected_location="Gulberg",
        expected_halal=True,
        excluded_allergens=["dairy"],
        target_dish_ids=["dish_001", "dish_003", "dish_014", "dish_017"],
        forbidden_dish_ids=["dish_002", "dish_016", "dish_005", "dish_006"],
        description="Comprehensive query evaluating Halal, budget, location, and dairy exclusion.",
    ),
    BenchmarkTestCase(
        query_id="BM_002",
        query="Gluten-free beef burger under 2000 in Johar Town",
        category="allergen_gluten",
        expected_status="SUPPORTED",
        max_price_pkr=2000.0,
        expected_location="Johar Town",
        expected_halal=True,
        excluded_allergens=["gluten"],
        target_dish_ids=["dish_022"],
        forbidden_dish_ids=["dish_021"],
        description="Must return lettuce-wrap burger (dish_022) and exclude standard bun burger (dish_021).",
    ),
    BenchmarkTestCase(
        query_id="BM_003",
        query="Chicken dishes without peanuts in Gulberg",
        category="allergen_peanuts",
        expected_status="SUPPORTED",
        expected_location="Gulberg",
        expected_halal=True,
        excluded_allergens=["peanuts"],
        target_dish_ids=["dish_001", "dish_003", "dish_014"],
        forbidden_dish_ids=["dish_004"],
        description="Must exclude chicken satay skewers (dish_004) containing peanuts.",
    ),
    BenchmarkTestCase(
        query_id="BM_004",
        query="Mystery Special Daily Daal",
        category="safety_unknown",
        expected_status="INSUFFICIENT_EVIDENCE",
        target_dish_ids=["dish_009"],
        forbidden_dish_ids=[],
        must_contain_warning="ALLERGEN_INFORMATION_UNKNOWN",
        description="Evaluates conservative safety for dish_009 (UNKNOWN allergen status). Must not claim safe.",
    ),
    BenchmarkTestCase(
        query_id="BM_005",
        query="Chefs Secret Karahi",
        category="safety_conflict",
        expected_status="CONFLICTING_EVIDENCE",
        target_dish_ids=["dish_033"],
        forbidden_dish_ids=[],
        must_contain_warning="CONFLICTING_ALLERGEN_EVIDENCE",
        description="Evaluates conservative safety for dish_033 (CONFLICT allergen status). Must flag conflict.",
    ),
    BenchmarkTestCase(
        query_id="BM_006",
        query="Seafood without shellfish in Gulberg",
        category="allergen_shellfish",
        expected_status="SUPPORTED",
        expected_location="Gulberg",
        excluded_allergens=["shellfish"],
        target_dish_ids=["dish_025"],
        forbidden_dish_ids=["dish_032", "dish_010", "dish_011"],
        description="Must return Lahori Fried Fish (dish_025) and exclude prawn dishes containing shellfish.",
    ),
    BenchmarkTestCase(
        query_id="BM_007",
        query="Meal under PKR 500 in DHA",
        category="negative_budget",
        expected_status="NO_MATCHING_ITEMS",
        max_price_pkr=500.0,
        expected_location="DHA",
        target_dish_ids=[],
        forbidden_dish_ids=["dish_006", "dish_010", "dish_015", "dish_023", "dish_029"],
        description="Negative query. Impossible budget ceiling for DHA (min price 850) must yield NO_MATCHING_ITEMS.",
    ),
    BenchmarkTestCase(
        query_id="BM_008",
        query="Dinner meal under PKR 20 in Gulberg",
        category="negative_budget",
        expected_status="NO_MATCHING_ITEMS",
        max_price_pkr=20.0,
        expected_location="Gulberg",
        target_dish_ids=[],
        forbidden_dish_ids=["dish_001", "dish_003"],
        description="Negative query. Impossible budget ceiling must yield immediate NO_MATCHING_ITEMS.",
    ),
    BenchmarkTestCase(
        query_id="BM_009",
        query="High protein roasted chicken under 1000 in Gulberg",
        category="budget_chicken",
        expected_status="SUPPORTED",
        max_price_pkr=1000.0,
        expected_location="Gulberg",
        expected_halal=True,
        target_dish_ids=["dish_017", "dish_014"],
        forbidden_dish_ids=["dish_001", "dish_003", "dish_002"],
        description="Must filter to chicken items under PKR 1000 (dish_017 at 950, dish_014 at 850).",
    ),
    BenchmarkTestCase(
        query_id="BM_010",
        query="Vegetarian meal without dairy in F-7",
        category="dietary_dairy_veg",
        expected_status="SUPPORTED",
        expected_location="F-7",
        excluded_allergens=["dairy"],
        target_dish_ids=["dish_019"],
        forbidden_dish_ids=["dish_018"],
        description="Must return Aloo Palak Vegan (dish_019) and exclude Palak Paneer (dish_018) with cheese.",
    ),
    BenchmarkTestCase(
        query_id="BM_011",
        query="Non-Halal meal in F-7",
        category="negative_dietary",
        expected_status="NO_MATCHING_ITEMS",
        expected_location="F-7",
        expected_halal=False,
        target_dish_ids=[],
        forbidden_dish_ids=["dish_018", "dish_019", "dish_020"],
        description="Negative query. All items in F-7 are Halal; requesting non-Halal must yield NO_MATCHING_ITEMS.",
    ),
    BenchmarkTestCase(
        query_id="BM_012",
        query="Peshawari Chicken Karahi",
        category="grounded_citation",
        expected_status="SUPPORTED",
        target_dish_ids=["dish_003"],
        forbidden_dish_ids=[],
        description="Tests exact dish citation propagation and price verification for dish_003.",
    ),
    BenchmarkTestCase(
        query_id="BM_013",
        query="Meal under PKR 400 in F-7",
        category="negative_budget",
        expected_status="NO_MATCHING_ITEMS",
        max_price_pkr=400.0,
        expected_location="F-7",
        target_dish_ids=[],
        forbidden_dish_ids=["dish_018", "dish_019", "dish_020"],
        description="Negative query. Impossible budget ceiling for F-7 (min price 650) must yield NO_MATCHING_ITEMS.",
    ),
    BenchmarkTestCase(
        query_id="BM_014",
        query="Daal in Saddar without dairy",
        category="allergen_daal",
        expected_status="CONFLICTING_EVIDENCE",
        expected_location="Saddar",
        excluded_allergens=["dairy"],
        target_dish_ids=[],
        forbidden_dish_ids=["dish_030"],
        must_contain_warning="CONFLICTING_ALLERGEN_EVIDENCE",
        description="Daal Mash Special (dish_030) contains butter/desi ghee (dairy); must be excluded. Candidate pool includes dish_033 which flags CONFLICTING_EVIDENCE.",
    ),
    BenchmarkTestCase(
        query_id="BM_015",
        query="Sweet dessert under 700 without tree nuts in Gulberg",
        category="allergen_dessert",
        expected_status="PARTIALLY_SUPPORTED",
        max_price_pkr=700.0,
        expected_location="Gulberg",
        excluded_allergens=["tree nuts", "tree_nuts"],
        target_dish_ids=[],
        forbidden_dish_ids=["dish_008", "dish_035"],
        must_contain_warning="ALLERGEN_INFORMATION_UNKNOWN",
        description="Shahi Tukra (dish_008) and Badami Kheer (dish_035) contain almonds/nuts; must be excluded. Candidate pool includes dish_009 which flags PARTIALLY_SUPPORTED.",
    ),
]
