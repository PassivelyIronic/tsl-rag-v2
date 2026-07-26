"""
Metryki retrievalu — funkcje czyste, bez bazy i bez modeli.

Wydzielone, żeby dały się przetestować jednostkowo. Definicje poniżej są
świadomym wyborem i mają znaczenie przy czytaniu wyników.
"""

from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], expected_docs: list[str], k: int) -> float:
    """
    Ile z oczekiwanych dokumentów pojawia się wśród pierwszych `k` **chunków**.

    Liczymy po chunkach, nie po unikalnych dokumentach, bo to chunki wchodzą
    do kontekstu modelu. Dokument reprezentowany przez jeden chunk na pozycji
    20 realnie nie trafi do promptu przy `rerank_top_n=5`, więc liczenie po
    unikalnych dokumentach zawyżałoby obraz.
    """
    if not expected_docs:
        return 1.0
    window = set(retrieved_doc_ids[:k])
    return len(window & set(expected_docs)) / len(set(expected_docs))


def reciprocal_rank(retrieved_doc_ids: list[str], expected_docs: list[str]) -> float:
    """
    1 / pozycja pierwszego chunka należącego do oczekiwanego dokumentu.
    Zwraca 0.0, gdy żaden nie wystąpił. Średnia po pytaniach daje MRR.
    """
    if not expected_docs:
        return 1.0
    wanted = set(expected_docs)
    for position, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in wanted:
            return 1.0 / position
    return 0.0


def first_hit_position(retrieved_doc_ids: list[str], expected_docs: list[str]) -> int | None:
    """Pozycja pierwszego trafnego chunka (1-indeksowana) albo None."""
    wanted = set(expected_docs)
    for position, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in wanted:
            return position
    return None
