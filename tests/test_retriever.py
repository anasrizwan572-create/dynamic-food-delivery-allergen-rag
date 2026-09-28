"""Unit tests for the BasicRetriever module."""

import pytest
from rag.retriever import BasicRetriever, retrieve


def test_basic_retriever_query_execution():
    retriever = BasicRetriever()
    results = retriever.retrieve("chicken karahi", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        first = results[0]
        assert "id" in first
        assert "dish_name" in first
        assert "restaurant_name" in first
        assert "score" in first
        assert "price_pkr" in first
        assert "location" in first
        assert "metadata" in first


def test_retriever_top_k_parameter():
    retriever = BasicRetriever()
    k1 = retriever.retrieve("chicken", top_k=1)
    k4 = retriever.retrieve("chicken", top_k=4)
    assert len(k1) == 1
    assert len(k4) == 4


def test_retriever_empty_query():
    retriever = BasicRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_global_retrieve_function():
    results = retrieve("biryani", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
