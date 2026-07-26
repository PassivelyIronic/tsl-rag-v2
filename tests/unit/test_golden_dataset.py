"""
Walidacja golden datasetu.

Sens tych testów: dataset jest rozszerzany partiami generowanymi nad korpusem
PDF poza repo. Bez walidacji zepsuty wpis — nieznany identyfikator dokumentu,
literówka w kategorii, zdanie zamiast fragmentów faktów — ujawnia się dopiero
w trakcie przebiegu, po kilkunastu minutach i kilkunastu wywołaniach API,
i wygląda jak porażka modelu, nie jak błąd danych.
"""

from __future__ import annotations

import pytest
from evals.golden_dataset.questions import (
    CATEGORIES,
    GOLDEN_DATASET,
    MAX_FACT_LENGTH,
    GoldenQuestion,
    validate,
)

pytestmark = pytest.mark.unit


def test_dataset_loads_and_validates():
    """GOLDEN_DATASET wczytuje się bez problemów walidacyjnych."""
    assert GOLDEN_DATASET, "dataset jest pusty"
    assert validate(GOLDEN_DATASET) == []


def test_every_question_has_unique_id():
    ids = [q.id for q in GOLDEN_DATASET]
    assert len(ids) == len(set(ids))
    assert all(ids), "każde pytanie musi mieć 'id'"


def test_categories_are_from_closed_set():
    assert {q.category for q in GOLDEN_DATASET} <= CATEGORIES


def test_out_of_scope_questions_have_no_expected_docs():
    for q in GOLDEN_DATASET:
        if q.category == "out_of_scope":
            assert q.expected_docs == []


def test_expected_docs_exist_in_registry():
    from tsl_rag.ingestion.cli import DOCUMENT_REGISTRY

    known = {stem.lower() for stem in DOCUMENT_REGISTRY}
    for q in GOLDEN_DATASET:
        for doc in q.expected_docs:
            assert doc in known, f"{q.id}: nieznany dokument {doc}"


def test_validator_rejects_unknown_document():
    bad = [
        GoldenQuestion(
            id="x",
            question="Pytanie?",
            expected_answer="fakt",
            expected_docs=["nie_ma_takiego_aktu"],
            category="numeric_fact",
        )
    ]
    problems = validate(bad)
    assert any("nieznany dokument" in p for p in problems)


def test_validator_rejects_long_expected_fact():
    """
    Fragment oczekiwanej odpowiedzi dłuższy niż limit nigdy nie dopasuje się
    dosłownie, więc dawałby 0 punktów niezależnie od jakości odpowiedzi.
    """
    bad = [
        GoldenQuestion(
            id="x",
            question="Pytanie?",
            expected_answer="x" * (MAX_FACT_LENGTH + 1),
            expected_docs=["ec_561_2006"],
            category="numeric_fact",
        )
    ]
    assert any("dłuższy niż" in p for p in validate(bad))


def test_validator_requires_two_docs_for_cross_document():
    bad = [
        GoldenQuestion(
            id="x",
            question="Pytanie?",
            expected_answer="fakt",
            expected_docs=["ec_561_2006"],
            category="cross_document",
        )
    ]
    assert any("co najmniej 2" in p for p in validate(bad))


def test_key_facts_splits_on_commas():
    q = GoldenQuestion(
        id="x",
        question="Pytanie?",
        expected_answer="9 godzin, 10 godzin, dwa razy w tygodniu",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    )
    assert q.key_facts == ["9 godzin", "10 godzin", "dwa razy w tygodniu"]


@pytest.mark.xfail(
    reason="Dataset ma 15 pytań; docelowo min. 5 na kategorię (PLAN.md Faza 1). "
    "Ten test jest bramką dla rozszerzenia datasetu — po dołożeniu pytań "
    "zdejmij xfail.",
    strict=False,
)
def test_each_category_has_at_least_five_questions():
    counts: dict[str, int] = {}
    for q in GOLDEN_DATASET:
        counts[q.category] = counts.get(q.category, 0) + 1
    thin = {c: counts.get(c, 0) for c in CATEGORIES if counts.get(c, 0) < 5}
    assert not thin, f"kategorie z mniej niż 5 pytaniami: {thin}"
