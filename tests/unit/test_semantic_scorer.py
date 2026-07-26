"""
Testy scorera semantycznego — na sztucznych wektorach, bez ładowania modelu.

Najważniejszy przypadek to ten na końcu: odpowiedź semantycznie bliska,
ale z inną liczbą, MUSI zostać uznana za błędną. "Dzienny czas jazdy to
11 godzin" jest bardzo blisko "9 godzin" — te same słowa, ten sam przepis,
różnica jednej cyfry. Metryka, która tego nie łapie, jest w tym projekcie
gorsza niż bezużyteczna, bo dotyczy kar i limitów czasu pracy.
"""

from __future__ import annotations

import asyncio

import pytest
from evals.semantic_scorer import score_answer, split_into_sentences

pytestmark = pytest.mark.unit


def _fake_embed(mapping: dict[str, list[float]]):
    """Zwraca funkcję embed, która przypisuje z góry ustalone wektory."""

    async def embed(texts: list[str]) -> list[list[float]]:
        return [mapping.get(t, [0.0, 0.0, 1.0]) for t in texts]

    return embed


def test_split_into_sentences():
    assert split_into_sentences("Pierwsze zdanie. Drugie zdanie!") == [
        "Pierwsze zdanie.",
        "Drugie zdanie!",
    ]
    assert split_into_sentences("") == []


def test_descriptive_fact_matched_by_similarity_not_wording():
    """Parafraza ma się liczyć — to jest cały powód istnienia tego scorera."""
    fact = "państwa trzecie"
    sentence = "Umowa obejmuje kraje spoza Unii Europejskiej."
    embed = _fake_embed({fact: [1.0, 0.0, 0.0], sentence: [0.95, 0.05, 0.0]})

    result = asyncio.run(score_answer([fact], sentence, embed))
    assert result["semantic_score"] == 1.0
    assert result["per_fact"][0]["kind"] == "opisowy"


def test_descriptive_fact_rejected_when_unrelated():
    fact = "państwa trzecie"
    sentence = "Tachograf podlega przeglądowi w warsztacie."
    embed = _fake_embed({fact: [1.0, 0.0, 0.0], sentence: [0.0, 1.0, 0.0]})

    result = asyncio.run(score_answer([fact], sentence, embed))
    assert result["semantic_score"] == 0.0


def test_numeric_fact_requires_literal_number_despite_high_similarity():
    """
    Sedno sprawy: semantycznie identyczne zdanie z INNĄ liczbą musi wypaść.
    Wektory ustawione tak, żeby podobieństwo wynosiło praktycznie 1.0 —
    mimo to fakt liczbowy ma zostać odrzucony.
    """
    fact = "9 godzin"
    wrong = "Dzienny czas prowadzenia pojazdu wynosi 11 godzin."
    embed = _fake_embed({fact: [1.0, 0.0, 0.0], wrong: [1.0, 0.0, 0.0]})

    result = asyncio.run(score_answer([fact], wrong, embed))
    assert result["semantic_score"] == 0.0
    assert result["per_fact"][0]["kind"] == "liczbowy"
    assert result["per_fact"][0]["similarity"] > 0.99  # podobieństwo wysokie...
    assert result["per_fact"][0]["present"] is False  # ...ale fakt odrzucony


def test_numeric_fact_accepted_when_number_present():
    fact = "9 godzin"
    right = "Dzienny czas prowadzenia pojazdu wynosi 9 godzin."
    embed = _fake_embed({fact: [1.0, 0.0, 0.0], right: [1.0, 0.0, 0.0]})

    result = asyncio.run(score_answer([fact], right, embed))
    assert result["semantic_score"] == 1.0


def test_empty_answer_scores_zero():
    result = asyncio.run(score_answer(["9 godzin"], "", _fake_embed({})))
    assert result["semantic_score"] == 0.0


def test_no_expected_facts_scores_one():
    """out_of_scope nie ma oczekiwanych faktów — ocenia się je odmową."""
    result = asyncio.run(score_answer([], "cokolwiek", _fake_embed({})))
    assert result["semantic_score"] == 1.0
