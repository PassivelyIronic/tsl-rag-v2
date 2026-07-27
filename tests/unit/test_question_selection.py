import pytest
from evals.golden_dataset.questions import GOLDEN_DATASET
from evals.run_evals import select_questions

pytestmark = pytest.mark.unit


def test_no_limit_returns_full_dataset():
    assert len(select_questions(list(GOLDEN_DATASET), None)) == len(GOLDEN_DATASET)


def test_limit_above_size_returns_full_dataset():
    assert len(select_questions(list(GOLDEN_DATASET), 10_000)) == len(GOLDEN_DATASET)


def test_limit_is_respected():
    assert len(select_questions(list(GOLDEN_DATASET), 21)) == 21


def test_zero_or_negative_limit_rejected():
    with pytest.raises(ValueError):
        select_questions(list(GOLDEN_DATASET), 0)


def test_subset_covers_every_category():
    """
    Sedno flagi --limit: dataset jest pogrupowany tematycznie, więc "pierwsze N"
    gubi całe kategorie (pierwsze 21 pytań to wyłącznie numeric_fact, procedure
    i scope). Podzbiór bez out_of_scope nie mierzy refusal_precision, a bez
    penalty nie mierzy kategorii najsłabszej w retrievalu.
    """
    all_categories = {q.category for q in GOLDEN_DATASET}
    subset = select_questions(list(GOLDEN_DATASET), 21)
    assert {q.category for q in subset} == all_categories


def test_selection_is_deterministic():
    """
    Przebiegi przed/po muszą dotyczyć DOKŁADNIE tych samych pytań — inaczej
    różnica metryki miesza efekt zmiany z różnicą trudności pytań.
    """
    first = [q.id for q in select_questions(list(GOLDEN_DATASET), 21)]
    second = [q.id for q in select_questions(list(GOLDEN_DATASET), 21)]
    assert first == second


def test_selection_preserves_dataset_order():
    positions = {q.id: i for i, q in enumerate(GOLDEN_DATASET)}
    subset = select_questions(list(GOLDEN_DATASET), 21)
    indices = [positions[q.id] for q in subset]
    assert indices == sorted(indices)


def test_subset_is_balanced_across_categories():
    """
    Dobór round-robin: liczebności kategorii w podzbiorze różnią się co najwyżej
    o 1, dopóki żadna kategoria się nie wyczerpie.
    """
    subset = select_questions(list(GOLDEN_DATASET), 12)
    counts = [sum(1 for q in subset if q.category == c) for c in {q.category for q in subset}]
    assert max(counts) - min(counts) <= 1
