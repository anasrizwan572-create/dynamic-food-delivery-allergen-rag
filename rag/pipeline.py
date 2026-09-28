"""Constraint-Aware RAG Pipeline: integrates query parsing, metadata pre-filtering, and grounded generation."""

import logging
import time
from typing import Any, Dict, List, Optional

from rag.config import config
from rag.context_builder import build_context
from rag.generator import BaseGenerator, get_generator
from rag.allergen_verifier import AllergenSafetyStatus, AllergenSafetyVerifier, get_allergen_verifier
from rag.citation import CitationEngine, GroundingStatus, get_citation_engine
from rag.hybrid_retriever import HybridRetriever, get_default_hybrid_retriever
from rag.metadata_filter import FilterResult
from rag.prompts import RAG_SYSTEM_PROMPT
from rag.query_parser import ParsedQuery, parse_query
from rag.reranker import BaseReranker, get_reranker
from rag.retriever import BasicRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pipeline_retriever(mode: Optional[str] = None):
    """Factory returning configured retriever based on retrieval_mode."""
    retrieval_mode = (mode or config.retrieval_mode).lower()
    if retrieval_mode == "dense":
        logger.info("Pipeline using dense-only BasicRetriever.")
        return BasicRetriever()
    logger.info("Pipeline using HybridRetriever (Dense + BM25 + RRF).")
    return get_default_hybrid_retriever()


class BasicRAGPipeline:
    """Constraint-aware RAG pipeline for menu retrieval, cross-encoder reranking, allergen verification, and grounded synthesis."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        generator: Optional[BaseGenerator] = None,
        reranker: Optional[BaseReranker] = None,
        allergen_verifier: Optional[AllergenSafetyVerifier] = None,
        citation_engine: Optional[CitationEngine] = None,
    ):
        self.retriever = retriever or get_pipeline_retriever()
        self.generator = generator or get_generator()
        self.reranker = reranker or get_reranker()
        self.allergen_verifier = allergen_verifier or get_allergen_verifier()
        self.citation_engine = citation_engine or get_citation_engine()

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        constraints_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the constraint-aware RAG pipeline for a given user query.

        Args:
            query: User's natural language question.
            top_k: Number of documents to retrieve (defaults to config.top_k).
            constraints_override: Optional explicit constraint parameters to override/augment parsed query.

        Returns:
            Structured dictionary with query status, parsed constraints, answer, citations, and telemetry.
        """
        k = top_k or config.top_k
        total_start = time.perf_counter()

        # Handle empty/whitespace query
        if not query or not query.strip():
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "query": query,
                "answer": "Based on the retrieved menu information, there is insufficient evidence to process an empty query.",
                "sources": [],
                "citations": [],
                "results": [],
                "grounding": {
                    "is_grounded": True,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "cited_ids": [],
                    "valid_citations": [],
                    "hallucinated_citations": [],
                    "unsupported_claims": [],
                    "warnings": ["Empty query string provided."],
                    "evidence_count": 0,
                    "verification_notes": ["Empty query string provided."],
                },
                "retrieved_documents": [],
                "warnings": ["Empty query string provided."],
                "parsed_constraints": {},
                "filter_summary": "Empty query provided.",
                "latency": {"parse_ms": 0.0, "retrieval_ms": 0.0, "rerank_ms": 0.0, "verify_ms": 0.0, "generation_ms": 0.0, "total_ms": 0.0},
                "disclaimer": "All constraints were evaluated deterministically.",
            }

        logger.info("Executing RAG pipeline for query: '%s' (top_k=%d)", query, k)

        # 1. Query & Constraint Parsing
        parse_start = time.perf_counter()
        parsed = parse_query(query)
        parse_ms = (time.perf_counter() - parse_start) * 1000.0

        hard_constraints = parsed.hard_constraints
        soft_prefs = parsed.soft_preferences

        # Apply any explicit constraint overrides passed via API
        if constraints_override:
            if constraints_override.get("max_price_pkr") is not None:
                hard_constraints.max_price_pkr = float(constraints_override["max_price_pkr"])
            if constraints_override.get("min_price_pkr") is not None:
                hard_constraints.min_price_pkr = float(constraints_override["min_price_pkr"])
            if constraints_override.get("location"):
                from ingestion.normalize_data import normalize_location
                hard_constraints.location = normalize_location(constraints_override["location"])
            if constraints_override.get("halal") is not None:
                hard_constraints.halal = bool(constraints_override["halal"])
            if constraints_override.get("excluded_allergens"):
                from ingestion.normalize_data import normalize_allergens
                norm_algs, _ = normalize_allergens(constraints_override["excluded_allergens"])
                hard_constraints.excluded_allergens = list(
                    dict.fromkeys(hard_constraints.excluded_allergens + norm_algs)
                )

        search_text = parsed.cleaned_query or query

        # 2. Vector Retrieval with Deterministic Metadata Pre-Filtering
        retrieval_start = time.perf_counter()
        candidate_k = max(k * 2, config.reranker_candidate_k)
        candidates, filter_result = self.retriever.retrieve_filtered(
            query=search_text,
            constraints=hard_constraints,
            top_k=candidate_k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 3. Handle NO_MATCHING_ITEMS condition
        if filter_result.status == "NO_MATCHING_ITEMS" or not candidates:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            logger.info("Query '%s' yielded NO_MATCHING_ITEMS.", query)
            return {
                "status": "NO_MATCHING_ITEMS",
                "query": query,
                "answer": "No menu item satisfies all of the requested constraints (e.g., budget, location, Halal, or allergen exclusions).",
                "sources": [],
                "citations": [],
                "results": [],
                "grounding": {
                    "is_grounded": True,
                    "status": "NO_MATCHING_ITEMS",
                    "cited_ids": [],
                    "valid_citations": [],
                    "hallucinated_citations": [],
                    "unsupported_claims": [],
                    "warnings": filter_result.warnings,
                    "evidence_count": 0,
                    "verification_notes": [filter_result.summary_message],
                },
                "retrieved_documents": [],
                "warnings": filter_result.warnings,
                "parsed_constraints": parsed.to_dict(),
                "filter_summary": filter_result.summary_message,
                "latency": {
                    "parse_ms": round(parse_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "rerank_ms": 0.0,
                    "verify_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
                "disclaimer": "All constraints were evaluated deterministically before candidate selection.",
            }

        # 4. Cross-Encoder Reranking (Milestone 7)
        rerank_start = time.perf_counter()
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=candidate_k,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        # 5. Allergen Safety Verification Gate (Milestone 8)
        verify_start = time.perf_counter()
        excluded_allergens = hard_constraints.excluded_allergens or []
        verification_report = self.allergen_verifier.verify_candidates(
            candidates=reranked_candidates,
            excluded_allergens=excluded_allergens,
        )
        verify_ms = (time.perf_counter() - verify_start) * 1000.0

        final_candidates = verification_report.verified_candidates[:k]
        combined_warnings = list(dict.fromkeys(filter_result.warnings + verification_report.warnings))

        if not final_candidates:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            logger.info("All candidates were rejected by the Allergen Safety Verifier.")
            all_unknown = all(
                res.status == AllergenSafetyStatus.INSUFFICIENT_EVIDENCE
                for res in verification_report.item_results
            )
            has_conflict = any(
                res.status == AllergenSafetyStatus.CONFLICTING_EVIDENCE
                for res in verification_report.item_results
            )
            if all_unknown and verification_report.item_results:
                rej_status = "INSUFFICIENT_EVIDENCE"
                rej_answer = (
                    "Some menu items may match your request, but the available allergen information "
                    "is insufficient to verify them safely."
                )
            elif has_conflict:
                rej_status = "CONFLICTING_EVIDENCE"
                rej_answer = (
                    "Some menu items may match your request, but conflicting allergen evidence was "
                    "detected across menu sources."
                )
            else:
                rej_status = "NO_MATCHING_ITEMS"
                rej_answer = (
                    "No menu item satisfies all of the requested constraints (e.g., budget, location, "
                    "Halal, or allergen exclusions)."
                )

            return {
                "status": rej_status,
                "query": query,
                "answer": rej_answer,
                "sources": [],
                "citations": [],
                "results": [],
                "grounding": {
                    "is_grounded": True,
                    "status": rej_status,
                    "cited_ids": [],
                    "valid_citations": [],
                    "hallucinated_citations": [],
                    "unsupported_claims": [],
                    "warnings": combined_warnings,
                    "evidence_count": 0,
                    "verification_notes": [rej_answer],
                },
                "retrieved_documents": [],
                "warnings": combined_warnings,
                "parsed_constraints": parsed.to_dict(),
                "filter_summary": "All candidate dishes failed allergen safety verification.",
                "latency": {
                    "parse_ms": round(parse_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "rerank_ms": round(rerank_ms, 2),
                    "verify_ms": round(verify_ms, 2),
                    "generation_ms": 0.0,
                    "total_ms": round(total_ms, 2),
                },
                "disclaimer": "All constraints were evaluated deterministically before candidate selection.",
            }

        # 6. Context Building from Strictly Verified Candidates
        context_str, source_ids = build_context(final_candidates)

        # 7. Grounded Generation
        gen_start = time.perf_counter()
        answer = self.generator.generate(
            query=query,
            context=context_str,
            retrieved_docs=final_candidates,
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000.0

        # 8. Citation Engine Grounding & Verification (Milestone 9)
        grounding_report = self.citation_engine.verify_grounding(
            answer=answer,
            retrieved_docs=final_candidates,
            query=query,
            parsed_constraints=parsed.to_dict(),
        )

        all_warnings = list(dict.fromkeys(combined_warnings + grounding_report.warnings))

        if grounding_report.status in [GroundingStatus.INSUFFICIENT_EVIDENCE, GroundingStatus.CONFLICTING_EVIDENCE]:
            pipeline_status = grounding_report.status.value
        elif grounding_report.status == GroundingStatus.PARTIALLY_SUPPORTED:
            pipeline_status = "PARTIALLY_SUPPORTED"
        else:
            pipeline_status = "SUPPORTED"

        # Structured results array per Section 15 of specification
        results_list = [
            {
                "dish_name": c.dish_name,
                "restaurant_name": c.restaurant_name,
                "price_pkr": c.price_pkr,
                "location": c.location,
                "halal": c.halal,
                "spice_level": c.spice_level,
                "allergens": c.allergens,
                "allergen_status": c.allergen_status,
                "match_reason": c.match_reason,
                "source": c.source_menu,
                "source_id": c.source_id,
                "reranker_score": c.reranker_score,
                "verification_status": c.verification_status,
                "warning": c.warning,
            }
            for c in grounding_report.valid_citations
        ]

        valid_source_ids = [c.source_id for c in grounding_report.valid_citations]

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "status": pipeline_status,
            "query": query,
            "answer": answer,
            "sources": valid_source_ids if valid_source_ids else source_ids,
            "citations": [c.model_dump() for c in grounding_report.valid_citations],
            "results": results_list,
            "grounding": grounding_report.model_dump(),
            "retrieved_documents": final_candidates,
            "warnings": all_warnings,
            "parsed_constraints": parsed.to_dict(),
            "filter_summary": filter_result.summary_message,
            "latency": {
                "parse_ms": round(parse_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "verify_ms": round(verify_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "disclaimer": "All constraints were evaluated deterministically before candidate selection.",
        }


# Global convenience instance
_default_pipeline: Optional[BasicRAGPipeline] = None


def answer_query(
    query: str,
    top_k: int = 5,
    constraints_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to run the default constraint-aware RAG pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = BasicRAGPipeline()
    return _default_pipeline.answer_query(
        query=query, top_k=top_k, constraints_override=constraints_override
    )
