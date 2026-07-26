from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoldenQuestion:
    question: str
    expected_answer: str
    expected_docs: list[str]
    category: str


GOLDEN_DATASET: list[GoldenQuestion] = [
    # --- Czas prowadzenia pojazdu ---
    GoldenQuestion(
        question="Jaki jest maksymalny dzienny czas prowadzenia pojazdu?",
        expected_answer="9 godzin, przedłużony do 10 godzin nie częściej niż dwa razy w tygodniu",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Jaki jest maksymalny tygodniowy czas prowadzenia pojazdu?",
        expected_answer="56 godzin",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Ile godzin może wynieść łączny czas prowadzenia pojazdu w ciągu dwóch kolejnych tygodni?",
        expected_answer="90 godzin",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    # --- Przerwy i odpoczynek ---
    GoldenQuestion(
        question="Jak długa przerwa przysługuje kierowcy po 4,5 godzinach jazdy?",
        expected_answer="45 minut",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Jaki jest minimalny dzienny czas odpoczynku kierowcy?",
        expected_answer="11 godzin",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Czy dzienny okres odpoczynku może być podzielony i na jakie części?",
        expected_answer="3 godziny, 9 godzin",
        expected_docs=["ec_561_2006"],
        category="procedure",
    ),
    # --- Odpoczynek tygodniowy ---
    GoldenQuestion(
        question="Jaki jest regularny tygodniowy okres odpoczynku?",
        expected_answer="45 godzin",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Ile kolejnych tygodni kierowca może korzystać ze skróconego odpoczynku tygodniowego?",
        expected_answer="dwa",
        expected_docs=["ec_561_2006"],
        category="numeric_fact",
    ),
    # --- Dyrektywa 2002/15 ---
    GoldenQuestion(
        question="Jaki jest maksymalny średni tygodniowy czas pracy zgodnie z dyrektywą 2002/15?",
        expected_answer="48 godzin",
        expected_docs=["directive_2002_15"],
        category="numeric_fact",
    ),
    GoldenQuestion(
        question="Do ilu godzin może być przedłużony tygodniowy czas pracy na mocy dyrektywy 2002/15?",
        expected_answer="60 godzin",
        expected_docs=["directive_2002_15"],
        category="numeric_fact",
    ),
    # --- Konflikt dokumentów ---
    GoldenQuestion(
        question="Jak rozporządzenie EC 561/2006 i dyrektywa 2002/15 regulują tygodniowy czas prowadzenia pojazdu i pracy?",
        expected_answer="56 godzin prowadzenia, 48 godzin średni czas pracy",
        expected_docs=["ec_561_2006", "directive_2002_15"],
        category="cross_document",
    ),
    # --- AETR ---
    GoldenQuestion(
        question="Czy umowa AETR ma zastosowanie do kierowców poza Unią Europejską?",
        expected_answer="państwa trzecie",
        expected_docs=["aetr"],
        category="scope",
    ),
    # --- Kary ---
    GoldenQuestion(
        question="Jakie kary grożą za przekroczenie dziennego czasu prowadzenia pojazdu?",
        expected_answer="grzywna",
        expected_docs=["tariff_driver_2022"],
        category="penalty",
    ),
    # --- Poza zakresem ---
    GoldenQuestion(
        question="Jakie jest ograniczenie prędkości na polskich autostradach?",
        expected_answer="I cannot answer",
        expected_docs=[],
        category="out_of_scope",
    ),
    GoldenQuestion(
        question="Jakie jest minimalne wynagrodzenie kierowcy ciężarówki w Niemczech?",
        expected_answer="I cannot answer",
        expected_docs=[],
        category="out_of_scope",
    ),
]
