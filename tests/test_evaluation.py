"""Unit and integration tests for Milestone 12: Evaluation Benchmark Suite.

Verifies:
- Benchmark dataset structure and validation
- Mathematical evaluation metric calculations (MRR, NDCG, Recall@k, Allergen Recall, Price Precision, Citation Accuracy)
- Retrieval evaluation runner
- Generation evaluation runner
- Safety invariant audit and reporting
"""

import json
import os
import pytest
from typing import Any, Dict, List

from evaluation.benchmark_dataset import BENCHMARK_CASES, BenchmarkTestCase
from evaluation.metrics import (
    compute_allergen_safety_recall,
    compute_citation_accuracy,
    compute_dietary_recall,
    compute_forbidden_leakage,
    compute_mrr,
    compute_ndcg_at_k,
    compute_price_precision,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_safety_invariant_compliance,
)


class TestBenchmarkDataset:
    """Validate synthetic benchmark dataset integrity and structure."""

    def test_benchmark_dataset_size(self):
        """Ensure benchmark dataset contains at least 15 representative scenarios."""
        assert len(BENCHMARK_CASES) == 15

    def test_unique_query_ids(self):
        """Verify all query IDs are unique and well-formatted."""
        query_ids = [c.query_id for c in BENCHMARK_CASES]
        assert len(query_ids) == len(set(query_ids))
        for qid in query_ids:
            assert qid.startswith("BM_")

    def test_test_case_validation(self):
        """Verify Pydantic model validation on benchmark cases."""
        for case in BENCHMARK_CASES:
            assert isinstance(case, BenchmarkTestCase)
            assert len(case.query.strip()) > 0
            assert case.expected_status in [
                "SUPPORTED",
                "NO_MATCHING_ITEMS",
                "INSUFFICIENT_EVIDENCE",
                "CONFLICTING_EVIDENCE",
                "PARTIALLY_SUPPORTED",
            ]

    def test_safety_cases_present(self):
        """Verify UNKNOWN and CONFLICT safety invariant test cases exist."""
        unknown_cases = [c for c in BENCHMARK_CASES if c.category == "safety_unknown"]
        conflict_cases = [c for c in BENCHMARK_CASES if c.category == "safety_conflict"]
        assert len(unknown_cases) >= 1
        assert "dish_009" in unknown_cases[0].target_dish_ids
        assert unknown_cases[0].must_contain_warning == "ALLERGEN_INFORMATION_UNKNOWN"

        assert len(conflict_cases) >= 1
        assert "dish_033" in conflict_cases[0].target_dish_ids
        assert conflict_cases[0].must_contain_warning == "CONFLICTING_ALLERGEN_EVIDENCE"

    def test_negative_rejection_cases_present(self):
        """Verify negative queries expecting NO_MATCHING_ITEMS are present."""
        rejection_cases = [c for c in BENCHMARK_CASES if c.expected_status == "NO_MATCHING_ITEMS"]
        assert len(rejection_cases) >= 3


class TestEvaluationMetrics:
    """Test mathematical accuracy of all benchmark metrics."""

    def test_reciprocal_rank_calculation(self):
        # Target at rank 1 -> RR = 1.0
        assert compute_reciprocal_rank(["dish_001", "dish_002"], ["dish_001"]) == 1.0
        # Target at rank 2 -> RR = 0.5
        assert compute_reciprocal_rank(["dish_002", "dish_001"], ["dish_001"]) == 0.5
        # Target not retrieved -> RR = 0.0
        assert compute_reciprocal_rank(["dish_003", "dish_004"], ["dish_001"]) == 0.0
        # Empty inputs -> RR = 0.0
        assert compute_reciprocal_rank([], ["dish_001"]) == 0.0
        assert compute_reciprocal_rank(["dish_001"], []) == 0.0

    def test_mrr_calculation(self):
        retrieved = [
            ["dish_001", "dish_002"],  # RR = 1.0
            ["dish_003", "dish_001"],  # RR = 0.5
            ["dish_004", "dish_005"],  # RR = 0.0
        ]
        targets = [["dish_001"], ["dish_001"], ["dish_001"]]
        mrr = compute_mrr(retrieved, targets)
        assert round(mrr, 4) == round((1.0 + 0.5 + 0.0) / 3, 4)

    def test_recall_at_k(self):
        retrieved = ["dish_001", "dish_002", "dish_003"]
        targets = ["dish_001", "dish_004"]
        # In top 2: only dish_001 retrieved -> 1/2 = 0.5
        assert compute_recall_at_k(retrieved, targets, k=2) == 0.5
        # In top 1: only dish_001 retrieved -> 1/2 = 0.5
        assert compute_recall_at_k(retrieved, targets, k=1) == 0.5
        # Negative query: empty targets and empty retrieved -> 1.0
        assert compute_recall_at_k([], [], k=5) == 1.0
        # Negative query: empty targets but items retrieved -> 0.0
        assert compute_recall_at_k(["dish_001"], [], k=5) == 0.0

    def test_ndcg_at_k(self):
        # Perfect ranking: target at rank 1
        ndcg_perfect = compute_ndcg_at_k(["dish_001", "dish_002"], ["dish_001"], k=5)
        assert round(ndcg_perfect, 4) == 1.0
        # Lower ranking: target at rank 2
        ndcg_rank2 = compute_ndcg_at_k(["dish_002", "dish_001"], ["dish_001"], k=5)
        assert ndcg_rank2 < ndcg_perfect
        assert ndcg_rank2 > 0.0
        # No targets retrieved
        assert compute_ndcg_at_k(["dish_002", "dish_003"], ["dish_001"], k=5) == 0.0

    def test_forbidden_leakage(self):
        # 1 forbidden out of 2 candidates -> 0.5
        assert compute_forbidden_leakage(["dish_001", "dish_006"], ["dish_006"]) == 0.5
        # 0 forbidden out of 2 candidates -> 0.0
        assert compute_forbidden_leakage(["dish_001", "dish_002"], ["dish_006"]) == 0.0
        # Empty candidates
        assert compute_forbidden_leakage([], ["dish_006"]) == 0.0

    def test_allergen_safety_recall(self):
        # Safe item without forbidden allergens
        safe_item = [{"dish_name": "Chicken Tikka", "allergens": [], "allergen_status": "VERIFIED"}]
        assert compute_allergen_safety_recall(safe_item, ["dairy"]) == 1.0

        # Unsafe item declaring excluded allergen
        unsafe_item = [{"dish_name": "Butter Chicken", "allergens": ["dairy"], "allergen_status": "VERIFIED"}]
        assert compute_allergen_safety_recall(unsafe_item, ["dairy"]) == 0.0

        # Item in forbidden_ids
        forbidden_item = [{"source_id": "dish_006", "dish_name": "Pizza", "allergens": []}]
        assert compute_allergen_safety_recall(forbidden_item, [], forbidden_ids=["dish_006"]) == 0.0

        # UNKNOWN item without warning is penalized
        unknown_no_warn = [{"dish_name": "Mystery", "allergens": [], "allergen_status": "UNKNOWN", "warning": None}]
        assert compute_allergen_safety_recall(unknown_no_warn, ["dairy"]) == 0.0

        # UNKNOWN item WITH required warning is accepted
        unknown_with_warn = [{
            "dish_name": "Mystery",
            "allergens": [],
            "allergen_status": "UNKNOWN",
            "warning": "Warning for 'Mystery': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."
        }]
        assert compute_allergen_safety_recall(unknown_with_warn, ["dairy"]) == 1.0

    def test_price_precision(self):
        items = [{"price_pkr": 1000.0}, {"price_pkr": 1200.0}]
        # All items <= 1500
        assert compute_price_precision(items, 1500.0) == 1.0
        # 1 item <= 1100 out of 2 -> 0.5
        assert compute_price_precision(items, 1100.0) == 0.5
        # None specified -> 1.0
        assert compute_price_precision(items, None) == 1.0

    def test_dietary_recall(self):
        items = [{"halal": True}, {"halal": True}]
        assert compute_dietary_recall(items, expected_halal=True) == 1.0
        items_with_nonhalal = [{"halal": True}, {"halal": False}]
        assert compute_dietary_recall(items_with_nonhalal, expected_halal=True) == 0.5

    def test_citation_accuracy(self):
        # 2 valid citations, 0 hallucinated -> 1.0
        report_clean = {"valid_citations": ["c1", "c2"], "hallucinated_citations": []}
        assert compute_citation_accuracy(report_clean) == 1.0
        # 1 valid, 1 hallucinated -> 0.5
        report_hallucinated = {"valid_citations": ["c1"], "hallucinated_citations": ["c2"]}
        assert compute_citation_accuracy(report_hallucinated) == 0.5

    def test_safety_invariant_compliance(self):
        resp_valid = {
            "status": "INSUFFICIENT_EVIDENCE",
            "warnings": ["Warning for 'Mystery': ALLERGEN_INFORMATION_UNKNOWN."],
            "answer": "Insufficient evidence to guarantee allergen safety.",
        }
        assert compute_safety_invariant_compliance(
            resp_valid,
            expected_status="INSUFFICIENT_EVIDENCE",
            must_contain_warning="ALLERGEN_INFORMATION_UNKNOWN",
        ) is True

        resp_invalid_status = dict(resp_valid, status="SUPPORTED")
        assert compute_safety_invariant_compliance(
            resp_invalid_status,
            expected_status="INSUFFICIENT_EVIDENCE",
        ) is False


class TestBenchmarkExecutionReports:
    """Validate generated benchmark reports exist and contain valid schema."""

    def test_benchmark_reports_exist(self):
        report_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation", "reports")
        summary_path = os.path.join(report_dir, "benchmark_summary.json")
        retrieval_path = os.path.join(report_dir, "retrieval_benchmark.json")
        generation_path = os.path.join(report_dir, "generation_benchmark.json")

        assert os.path.exists(summary_path), "benchmark_summary.json must exist"
        assert os.path.exists(retrieval_path), "retrieval_benchmark.json must exist"
        assert os.path.exists(generation_path), "generation_benchmark.json must exist"

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            assert "num_benchmark_cases" in summary_data
            assert summary_data["num_benchmark_cases"] == 15
            assert "retrieval_highlights" in summary_data
            assert "generation_and_safety_metrics" in summary_data

            gen_metrics = summary_data["generation_and_safety_metrics"]
            assert gen_metrics["allergen_safety_recall"] == 1.0
            assert gen_metrics["price_precision"] == 1.0
            assert gen_metrics["dietary_recall"] == 1.0
            assert gen_metrics["citation_accuracy"] == 1.0
            assert gen_metrics["safety_invariant_compliance"] == 1.0
