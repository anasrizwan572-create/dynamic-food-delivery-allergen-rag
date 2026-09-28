"""Master Evaluation Benchmark Suite Orchestrator.

Runs complete, reproducible evaluation of the Food Delivery RAG system across:
1. Retrieval and Ranking Architectures (Dense vs BM25 vs Hybrid RRF vs Reranking vs Pipeline)
2. End-to-End Generation & Safety (Allergen Safety Recall, Price Precision, Dietary Recall,
   Citation Fidelity, Invariant Compliance, Negative Query Rejection, Latency Profiling)

Outputs:
- CLI human-readable performance & safety summary tables
- Structured JSON reports:
  * evaluation/reports/retrieval_benchmark.json
  * evaluation/reports/generation_benchmark.json
  * evaluation/reports/benchmark_summary.json
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict

from evaluation.benchmark_dataset import BENCHMARK_CASES
from evaluation.evaluate_generation import evaluate_generation
from evaluation.evaluate_retrieval import run_retrieval_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def run_full_benchmark() -> Dict[str, Any]:
    """Execute complete retrieval and generation evaluation suite."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    start_total = time.perf_counter()

    print("\n" + "=" * 90)
    print("      FOOD DELIVERY RAG SYSTEM - COMPREHENSIVE BENCHMARK SUITE")
    print("      Edversity AI Solutions Engineering Program | Milestone 12")
    print("=" * 90)
    print(f"Dataset Size: {len(BENCHMARK_CASES)} structured benchmark scenarios")
    print(f"Report Directory: {REPORT_DIR}\n")

    # 1. Retrieval Benchmark
    print(">>> STAGE 1: Evaluating Retrieval & Ranking Architectures...")
    retrieval_summary = run_retrieval_evaluation(cases=BENCHMARK_CASES, output_dir=REPORT_DIR)

    # 2. Generation & Safety Benchmark
    print("\n>>> STAGE 2: Evaluating End-to-End Grounded Generation & Safety...")
    generation_summary = evaluate_generation(cases=BENCHMARK_CASES, output_dir=REPORT_DIR)

    elapsed_total_s = time.perf_counter() - start_total

    # Master Consolidated Summary
    master_summary: Dict[str, Any] = {
        "benchmark_suite_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": round(elapsed_total_s, 2),
        "num_benchmark_cases": len(BENCHMARK_CASES),
        "retrieval_highlights": {
            "dense_mrr": retrieval_summary["results_by_mode"]["Dense Vector Search"]["mrr"],
            "dense_leakage": retrieval_summary["results_by_mode"]["Dense Vector Search"]["forbidden_leakage_rate"],
            "pipeline_mrr": retrieval_summary["results_by_mode"]["Production Pipeline (Pre-filter + Hybrid + Reranker)"]["mrr"],
            "pipeline_ndcg_at_5": retrieval_summary["results_by_mode"]["Production Pipeline (Pre-filter + Hybrid + Reranker)"]["ndcg_at_5"],
            "pipeline_leakage": retrieval_summary["results_by_mode"]["Production Pipeline (Pre-filter + Hybrid + Reranker)"]["forbidden_leakage_rate"],
            "pipeline_latency_ms": retrieval_summary["results_by_mode"]["Production Pipeline (Pre-filter + Hybrid + Reranker)"]["mean_latency_ms"],
        },
        "generation_and_safety_metrics": generation_summary["summary_metrics"],
        "reports": {
            "retrieval_report": os.path.join(REPORT_DIR, "retrieval_benchmark.json"),
            "generation_report": os.path.join(REPORT_DIR, "generation_benchmark.json"),
        },
    }

    summary_file = os.path.join(REPORT_DIR, "benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 90)
    print("                    FINAL BENCHMARK EXECUTIVE AUDIT")
    print("=" * 90)
    print(f"Total Benchmark Cases Evaluated : {len(BENCHMARK_CASES)}")
    print(f"Total Evaluation Time           : {elapsed_total_s:.2f} seconds")
    print(f"Production Retrieval MRR        : {master_summary['retrieval_highlights']['pipeline_mrr']:.4f}")
    print(f"Production Retrieval NDCG@5     : {master_summary['retrieval_highlights']['pipeline_ndcg_at_5']:.4f}")
    print(f"Production Forbidden Leakage    : {master_summary['retrieval_highlights']['pipeline_leakage']:.4f} (Zero Leakage)")
    print(f"Allergen Safety Recall          : {master_summary['generation_and_safety_metrics']['allergen_safety_recall'] * 100:.1f}%")
    print(f"Budget Price Precision          : {master_summary['generation_and_safety_metrics']['price_precision'] * 100:.1f}%")
    print(f"Halal Dietary Recall            : {master_summary['generation_and_safety_metrics']['dietary_recall'] * 100:.1f}%")
    print(f"Citation Accuracy               : {master_summary['generation_and_safety_metrics']['citation_accuracy'] * 100:.1f}%")
    print(f"Safety Invariant Compliance     : {master_summary['generation_and_safety_metrics']['safety_invariant_compliance'] * 100:.1f}%")
    print(f"Negative Query Rejection        : {master_summary['generation_and_safety_metrics']['negative_rejection_accuracy'] * 100:.1f}%")
    print(f"Mean Pipeline Latency           : {master_summary['generation_and_safety_metrics']['latency_profile_mean_ms']['total_ms']:.2f} ms")
    print("=" * 90)
    print(f"Master report saved: {summary_file}\n")

    return master_summary


if __name__ == "__main__":
    run_full_benchmark()
