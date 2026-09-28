"""End-to-end integration tests for BasicRAGPipeline."""

import pytest
from rag.pipeline import BasicRAGPipeline, answer_query


def test_answer_query_basic_structure():
    res = answer_query("What chicken meals are available?", top_k=3)
    assert "query" in res
    assert "answer" in res
    assert "sources" in res
    assert "retrieved_documents" in res
    assert "latency" in res
    assert isinstance(res["sources"], list)
    assert len(res["retrieved_documents"]) <= 3
    assert res["latency"]["total_ms"] >= 0


def test_no_hallucinated_source_ids():
    res = answer_query("Find biryani in Saddar", top_k=3)
    retrieved_ids = {doc["id"] for doc in res["retrieved_documents"]}
    
    # Every source ID reported in sources must be among the retrieved documents
    for src_id in res["sources"]:
        assert src_id in retrieved_ids, f"Source ID {src_id} was hallucinated!"


def test_grounded_response_cites_sources():
    res = answer_query("Peshawari Chicken Karahi", top_k=2)
    # The generated text should contain at least one source bracket citation
    assert "[" in res["answer"] and "]" in res["answer"]
    assert len(res["sources"]) > 0


def test_empty_query_pipeline():
    res = answer_query("", top_k=3)
    assert res["retrieved_documents"] == []
    assert res["sources"] == []
    assert "insufficient evidence" in res["answer"].lower()
