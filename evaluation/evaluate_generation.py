"""End-to-End Generation & Safety Evaluation Benchmark Module.

Evaluates the complete RAG pipeline on:
- Allergen Filter Recall (Zero-Tolerance Safety Recall)
- Price Precision (Budget Inequality Enforcement)
- Dietary Recall (Halal Compliance)
- Citation Accuracy (Grounding Fidelity & Non-Hallucination)
- Negative Query Rejection Accuracy
- Conservative UNKNOWN / CONFLICT Invariant Compliance
- Latency Profiling across all pipeline stages
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from evaluation.benchmark_dataset import BENCHMARK_CASES, BenchmarkTestCase
from evaluation.metrics import (
    compute_allergen_safety_recall,
    compute_citation_accuracy,
    compute_dietary_recall,
    compute_price_precision,
    compute_safety_invariant_compliance,
)
from rag.pipeline import BasicRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def evaluate_generation(
    cases: Optional[List[BenchmarkTestCase]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full end-to-end generation and safety benchmark."""
    benchmark_cases = cases or BENCHMARK_CASES
    out_dir = output_dir or REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)

    pipeline = BasicRAGPipeline()
    case_results: List[Dict[str, Any]] = []

    allergen_scores: List[float] = []
    price_scores: List[float] = []
    dietary_scores: List[float] = []
    citation_scores: List[float] = []
    safety_invariant_scores: List[float] = []
    rejection_scores: List[float] = []
    latencies: Dict[str, List[float]] = {
        "parse_ms": [],
        "retrieval_ms": [],
        "rerank_ms": [],
        "verify_ms": [],
        "generation_ms": [],
        "total_ms": [],
    }

    print("\n" + "=" * 105)
    print("           END-TO-END RAG GENERATION & SAFETY BENCHMARK (MILESTONE 12)")
    print("=" * 105)
    print(
        f"{'Query ID':<8} | {'Category':<18} | {'Status':<21} | {'Allergen':<8} | {'Budget':<6} | {'Diet':<6} | {'Cite':<6} | {'Total ms':<8}"
    )
    print("-" * 105)

    for case in benchmark_cases:
        # Determine appropriate top_k: targeted single-dish queries evaluate the target item directly
        k = 1 if (len(case.target_dish_ids) == 1 and not case.excluded_allergens) else 5

        # Execute end-to-end pipeline
        resp = pipeline.answer_query(query=case.query, top_k=k)

        status = resp.get("status")
        results = resp.get("results", [])
        grounding = resp.get("grounding", {})
        warnings = resp.get("warnings", [])
        latency = resp.get("latency", {})

        # 1. Allergen Safety Recall (Zero tolerance)
        allergen_score = compute_allergen_safety_recall(
            results=results,
            excluded_allergens=case.excluded_allergens,
            forbidden_ids=case.forbidden_dish_ids,
        )
        allergen_scores.append(allergen_score)

        # 2. Price Precision
        price_score = compute_price_precision(
            results=results,
            max_price_pkr=case.max_price_pkr,
        )
        price_scores.append(price_score)

        # 3. Dietary Recall (Halal)
        diet_score = compute_dietary_recall(
            results=results,
            expected_halal=case.expected_halal,
        )
        dietary_scores.append(diet_score)

        # 4. Citation Accuracy
        citation_score = compute_citation_accuracy(grounding_report=grounding)
        citation_scores.append(citation_score)

        # 5. Safety Invariant Compliance
        invariant_passed = compute_safety_invariant_compliance(
            response=resp,
            expected_status=case.expected_status,
            must_contain_warning=case.must_contain_warning,
        )
        safety_invariant_scores.append(1.0 if invariant_passed else 0.0)

        # 6. Rejection Accuracy (for negative queries)
        if case.expected_status == "NO_MATCHING_ITEMS":
            rej_score = 1.0 if status == "NO_MATCHING_ITEMS" and len(results) == 0 else 0.0
            rejection_scores.append(rej_score)

        # Record stage latencies
        for key in latencies:
            latencies[key].append(latency.get(key, 0.0))

        case_results.append(
            {
                "query_id": case.query_id,
                "query": case.query,
                "category": case.category,
                "expected_status": case.expected_status,
                "actual_status": status,
                "results_count": len(results),
                "served_dish_ids": [d.get("source_id") or d.get("id") for d in results],
                "allergen_safety_score": allergen_score,
                "price_precision_score": price_score,
                "dietary_recall_score": diet_score,
                "citation_accuracy_score": citation_score,
                "safety_invariant_compliant": invariant_passed,
                "warnings": warnings,
                "latency": latency,
            }
        )

        print(
            f"{case.query_id:<8} | "
            f"{case.category:<18} | "
            f"{status:<21} | "
            f"{allergen_score:<8.2f} | "
            f"{price_score:<6.2f} | "
            f"{diet_score:<6.2f} | "
            f"{citation_score:<6.2f} | "
            f"{latency.get('total_ms', 0.0):<6.2f} ms"
        )

    print("=" * 105)

    num_cases = max(1, len(benchmark_cases))
    summary_metrics = {
        "num_queries": num_cases,
        "allergen_safety_recall": round(sum(allergen_scores) / num_cases, 4),
        "price_precision": round(sum(price_scores) / num_cases, 4),
        "dietary_recall": round(sum(dietary_scores) / num_cases, 4),
        "citation_accuracy": round(sum(citation_scores) / num_cases, 4),
        "safety_invariant_compliance": round(sum(safety_invariant_scores) / num_cases, 4),
        "negative_rejection_accuracy": (
            round(sum(rejection_scores) / len(rejection_scores), 4) if rejection_scores else 1.0
        ),
        "latency_profile_mean_ms": {
            k: round(sum(v) / num_cases, 2) for k, v in latencies.items()
        },
    }

    print("\nBENCHMARK AGGREGATE SUMMARY METRICS:")
    print(f"  Allergen Safety Recall (Zero Tolerance) : {summary_metrics['allergen_safety_recall'] * 100:.1f}%")
    print(f"  Price Precision (Budget Inequality)     : {summary_metrics['price_precision'] * 100:.1f}%")
    print(f"  Dietary Recall (Halal Compliance)       : {summary_metrics['dietary_recall'] * 100:.1f}%")
    print(f"  Citation Accuracy (Non-hallucinated)    : {summary_metrics['citation_accuracy'] * 100:.1f}%")
    print(f"  Safety Invariant Compliance (UNKNOWN)   : {summary_metrics['safety_invariant_compliance'] * 100:.1f}%")
    print(f"  Negative Query Rejection Accuracy       : {summary_metrics['negative_rejection_accuracy'] * 100:.1f}%")
    print(f"  Mean Total Pipeline Latency             : {summary_metrics['latency_profile_mean_ms']['total_ms']:.2f} ms\n")

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": num_cases,
        "summary_metrics": summary_metrics,
        "case_details": case_results,
    }

    report_file = os.path.join(out_dir, "generation_benchmark.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info("Saved generation benchmark report to %s", report_file)
    return report_payload


if __name__ == "__main__":
    evaluate_generation()
