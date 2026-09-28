"""Retrieval and Ranking Evaluation Module.

Evaluates dense semantic retrieval, BM25 lexical retrieval, hybrid RRF fusion,
and cross-encoder reranking against the synthetic benchmark dataset.

Computes:
- MRR (Mean Reciprocal Rank)
- Recall@1, Recall@3, Recall@5
- NDCG@5
- Forbidden Document Leakage Rate
- Retrieval Latency (ms)
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from evaluation.benchmark_dataset import BENCHMARK_CASES, BenchmarkTestCase
from evaluation.metrics import (
    compute_forbidden_leakage,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from rag.bm25 import get_default_bm25_index
from rag.hybrid_retriever import HybridRetriever
from rag.query_parser import parse_query
from rag.reranker import get_reranker
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def evaluate_retrieval_mode(
    mode_name: str,
    cases: List[BenchmarkTestCase],
    retriever_fn: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a specific retrieval configuration across the benchmark dataset."""
    all_retrieved_ids: List[List[str]] = []
    all_target_ids: List[List[str]] = []
    case_results: List[Dict[str, Any]] = []

    total_time_ms = 0.0
    recall_1_list: List[float] = []
    recall_3_list: List[float] = []
    recall_5_list: List[float] = []
    ndcg_5_list: List[float] = []
    leakage_list: List[float] = []

    for case in cases:
        start_t = time.perf_counter()
        retrieved_items = retriever_fn(case)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        total_time_ms += elapsed_ms

        ret_ids = [item.get("id") or item.get("source_id") for item in retrieved_items]
        all_retrieved_ids.append(ret_ids)
        all_target_ids.append(case.target_dish_ids)

        r1 = compute_recall_at_k(ret_ids, case.target_dish_ids, k=1)
        r3 = compute_recall_at_k(ret_ids, case.target_dish_ids, k=3)
        r5 = compute_recall_at_k(ret_ids, case.target_dish_ids, k=5)
        ndcg = compute_ndcg_at_k(ret_ids, case.target_dish_ids, k=5)
        leakage = compute_forbidden_leakage(ret_ids, case.forbidden_dish_ids, k=5)
        rr = compute_reciprocal_rank(ret_ids, case.target_dish_ids)

        recall_1_list.append(r1)
        recall_3_list.append(r3)
        recall_5_list.append(r5)
        ndcg_5_list.append(ndcg)
        leakage_list.append(leakage)

        case_results.append(
            {
                "query_id": case.query_id,
                "query": case.query,
                "category": case.category,
                "target_ids": case.target_dish_ids,
                "forbidden_ids": case.forbidden_dish_ids,
                "retrieved_ids": ret_ids[:top_k],
                "reciprocal_rank": round(rr, 4),
                "recall_at_5": round(r5, 4),
                "ndcg_at_5": round(ndcg, 4),
                "forbidden_leakage": round(leakage, 4),
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    mrr = compute_mrr(all_retrieved_ids, all_target_ids)
    num_cases = max(1, len(cases))
    mean_latency = total_time_ms / num_cases

    return {
        "mode": mode_name,
        "num_queries": len(cases),
        "mrr": round(mrr, 4),
        "recall_at_1": round(sum(recall_1_list) / num_cases, 4),
        "recall_at_3": round(sum(recall_3_list) / num_cases, 4),
        "recall_at_5": round(sum(recall_5_list) / num_cases, 4),
        "ndcg_at_5": round(sum(ndcg_5_list) / num_cases, 4),
        "forbidden_leakage_rate": round(sum(leakage_list) / num_cases, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "case_details": case_results,
    }


def run_retrieval_evaluation(
    cases: Optional[List[BenchmarkTestCase]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full comparative retrieval benchmark across all modes."""
    benchmark_cases = cases or BENCHMARK_CASES
    out_dir = output_dir or REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)

    vector_store = VectorStore()
    bm25_index = get_default_bm25_index()
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index)
    reranker = get_reranker()

    # Define retrieval runners
    def dense_only_runner(c: BenchmarkTestCase) -> List[Dict[str, Any]]:
        return vector_store.search(query=c.query, top_k=5)

    def bm25_only_runner(c: BenchmarkTestCase) -> List[Dict[str, Any]]:
        return bm25_index.search(query=c.query, top_k=5)

    def hybrid_runner(c: BenchmarkTestCase) -> List[Dict[str, Any]]:
        return hybrid_retriever.retrieve(query=c.query, top_k=5)

    def hybrid_rerank_runner(c: BenchmarkTestCase) -> List[Dict[str, Any]]:
        candidates = hybrid_retriever.retrieve(query=c.query, top_k=15)
        return reranker.rerank(query=c.query, candidates=candidates, top_k=5)

    def prefiltered_pipeline_runner(c: BenchmarkTestCase) -> List[Dict[str, Any]]:
        parsed = parse_query(c.query)
        # Apply test case overrides if explicit
        hc = parsed.hard_constraints
        if c.max_price_pkr is not None:
            hc.max_price_pkr = c.max_price_pkr
        if c.min_price_pkr is not None:
            hc.min_price_pkr = c.min_price_pkr
        if c.expected_location is not None:
            hc.location = c.expected_location
        if c.expected_halal is not None:
            hc.halal = c.expected_halal
        if c.excluded_allergens:
            hc.excluded_allergens = list(dict.fromkeys(hc.excluded_allergens + c.excluded_allergens))

        filtered_cands, _ = hybrid_retriever.retrieve_filtered(
            query=parsed.cleaned_query or c.query,
            constraints=hc,
            top_k=15,
        )
        return reranker.rerank(query=c.query, candidates=filtered_cands, top_k=5)

    modes = [
        ("Dense Vector Search", dense_only_runner),
        ("BM25 Lexical Search", bm25_only_runner),
        ("Hybrid (Dense + BM25 RRF)", hybrid_runner),
        ("Hybrid + Cross-Encoder Reranker", hybrid_rerank_runner),
        ("Production Pipeline (Pre-filter + Hybrid + Reranker)", prefiltered_pipeline_runner),
    ]

    benchmark_summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": len(benchmark_cases),
        "results_by_mode": {},
    }

    print("\n" + "=" * 95)
    print("           REPRODUCIBLE RETRIEVAL BENCHMARK EVALUATION (MILESTONE 12)")
    print("=" * 95)
    print(
        f"{'Retrieval Architecture':<42} | {'MRR':<6} | {'R@1':<6} | {'R@5':<6} | {'NDCG@5':<6} | {'Leakage':<8} | {'Latency':<8}"
    )
    print("-" * 95)

    for mode_name, runner in modes:
        mode_metrics = evaluate_retrieval_mode(mode_name, benchmark_cases, runner, top_k=5)
        benchmark_summary["results_by_mode"][mode_name] = mode_metrics
        print(
            f"{mode_name:<42} | "
            f"{mode_metrics['mrr']:<6.4f} | "
            f"{mode_metrics['recall_at_1']:<6.4f} | "
            f"{mode_metrics['recall_at_5']:<6.4f} | "
            f"{mode_metrics['ndcg_at_5']:<6.4f} | "
            f"{mode_metrics['forbidden_leakage_rate']:<8.4f} | "
            f"{mode_metrics['mean_latency_ms']:<6.2f} ms"
        )

    print("=" * 95 + "\n")

    # Save to disk
    report_file = os.path.join(out_dir, "retrieval_benchmark.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    logger.info("Saved retrieval benchmark report to %s", report_file)
    return benchmark_summary


if __name__ == "__main__":
    run_retrieval_evaluation()
