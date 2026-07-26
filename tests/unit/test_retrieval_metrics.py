"""Testy metryk retrievalu — funkcje czyste, bez bazy i bez modeli."""

from __future__ import annotations

import pytest
from evals.retrieval_metrics import first_hit_position, recall_at_k, reciprocal_rank

pytestmark = pytest.mark.unit


def test_recall_at_k_counts_within_window_only():
    docs = ["a", "b", "c", "d", "e", "target"]
    assert recall_at_k(docs, ["target"], 5) == 0.0
    assert recall_at_k(docs, ["target"], 10) == 1.0


def test_recall_at_k_with_two_expected_docs():
    docs = ["x", "a", "y", "b"]
    assert recall_at_k(docs, ["a", "b"], 4) == 1.0
    assert recall_at_k(docs, ["a", "b"], 2) == 0.5


def test_recall_at_k_counts_chunks_not_unique_docs():
    """
    Liczymy po chunkach, bo to chunki wchodzą do kontekstu. Pięć chunków
    jednego dokumentu na początku listy realnie wypycha z promptu wszystko inne,
    i metryka ma to odzwierciedlać.
    """
    docs = ["spam", "spam", "spam", "spam", "spam", "target"]
    assert recall_at_k(docs, ["target"], 5) == 0.0


def test_recall_at_k_no_expected_docs_is_one():
    assert recall_at_k(["a"], [], 5) == 1.0


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["a", "target"], ["target"]) == 0.5
    assert reciprocal_rank(["target"], ["target"]) == 1.0
    assert reciprocal_rank(["a", "b"], ["target"]) == 0.0


def test_reciprocal_rank_takes_earliest_of_several_expected():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


def test_first_hit_position():
    assert first_hit_position(["x", "y", "a"], ["a"]) == 3
    assert first_hit_position(["x"], ["a"]) is None
