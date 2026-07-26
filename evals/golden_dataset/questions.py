"""
Golden dataset — wczytywanie i walidacja.

Pytania mieszkają w `questions.json`, nie w kodzie. Powody:

1. Dataset jest **danymi**, i to danymi generowanymi częściowo poza repo
   (NotebookLM nad korpusem PDF). Wklejanie ich w kod oznaczałoby przeglądanie
   diffa Pythona po każdym rozszerzeniu i ryzyko, że literówka w składni
   wywali cały eval.
2. Format da się zwalidować przy wczytaniu i w teście, więc zepsuty dataset
   pada z konkretnym komunikatem, a nie w połowie przebiegu na 40 pytaniach.
3. Ten sam plik czyta przyszły `run_retrieval_evals.py`, który nie wywołuje
   LLM-a — a to on ma być bramką w CI.

Opis formatu dla osoby (lub modelu) generującego pytania: `docs/GOLDEN_DATASET.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_DATASET_PATH = Path(__file__).with_name("questions.json")

# Kategorie muszą być zamkniętym zbiorem — inaczej literówka w kategorii tworzy
# nową, jednoelementową grupę w agregacie i cicho psuje interpretację wyników.
CATEGORIES: frozenset[str] = frozenset(
    {
        "numeric_fact",
        "procedure",
        "cross_document",
        "scope",
        "penalty",
        "out_of_scope",
    }
)

# "standard"      — pytanie napisane poprawną polszczyzną z diakrytykami
# "bez_ogonkow"   — to samo pytanie bez polskich znaków; mierzy składanie
#                   diakrytyków w tokenizerze BM25, czego dataset v1 nie robił
# "potoczne"      — język potoczny, bez terminologii z aktu prawnego
VARIANTS: frozenset[str] = frozenset({"standard", "bez_ogonkow", "potoczne"})

# Ocena keyword-match sprawdza, czy każdy fragment `expected_answer` (rozdzielony
# przecinkami) występuje w odpowiedzi jako podciąg. Długie fragmenty nigdy nie
# trafią dosłownie, więc taki wpis zawsze da 0 i wygląda jak porażka modelu.
MAX_FACT_LENGTH = 60


@dataclass
class GoldenQuestion:
    question: str
    expected_answer: str
    expected_docs: list[str]
    category: str
    id: str = ""
    variant: str = "standard"
    expected_articles: list[str] = field(default_factory=list)
    source_note: str = ""

    @property
    def key_facts(self) -> list[str]:
        """Fragmenty, których szuka ocena keyword-match."""
        return [f.strip() for f in self.expected_answer.lower().split(",") if f.strip()]


def _known_document_ids() -> set[str]:
    # Import lokalny: `evals` nie ma zależeć od pakietu ingestu przy imporcie
    # modułu, a tylko przy walidacji.
    from tsl_rag.ingestion.cli import DOCUMENT_REGISTRY

    return {stem.lower() for stem in DOCUMENT_REGISTRY}


def validate(questions: list[GoldenQuestion], *, strict_docs: bool = True) -> list[str]:
    """
    Zwraca listę problemów. Pusta lista = dataset poprawny.

    `strict_docs=False` pomija sprawdzenie identyfikatorów dokumentów — przydatne,
    gdy walidujemy dataset bez dostępu do rejestru dokumentów.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    known_docs = _known_document_ids() if strict_docs else set()

    for i, q in enumerate(questions):
        where = f"[{i}] {q.id or q.question[:40]!r}"

        if not q.id:
            problems.append(f"{where}: brak pola 'id'")
        elif q.id in seen_ids:
            problems.append(f"{where}: zduplikowane 'id'")
        else:
            seen_ids.add(q.id)

        if not q.question.strip():
            problems.append(f"{where}: puste 'question'")

        if q.category not in CATEGORIES:
            problems.append(
                f"{where}: nieznana kategoria {q.category!r}, dozwolone: {sorted(CATEGORIES)}"
            )

        if q.variant not in VARIANTS:
            problems.append(
                f"{where}: nieznany wariant {q.variant!r}, dozwolone: {sorted(VARIANTS)}"
            )

        if q.category == "out_of_scope":
            if q.expected_docs:
                problems.append(f"{where}: out_of_scope musi mieć puste 'expected_docs'")
        else:
            if not q.expected_docs:
                problems.append(f"{where}: brak 'expected_docs' (wymagane poza out_of_scope)")
            if not q.expected_answer.strip():
                problems.append(f"{where}: brak 'expected_answer' (wymagane poza out_of_scope)")
            for fact in q.key_facts:
                if len(fact) > MAX_FACT_LENGTH:
                    problems.append(
                        f"{where}: fragment oczekiwanej odpowiedzi dłuższy niż "
                        f"{MAX_FACT_LENGTH} znaków ({len(fact)}) — ocena keyword-match "
                        f"nigdy go nie dopasuje: {fact[:50]!r}..."
                    )

        if q.category == "cross_document" and len(q.expected_docs) < 2:
            problems.append(f"{where}: cross_document wymaga co najmniej 2 'expected_docs'")

        if strict_docs:
            for doc in q.expected_docs:
                if doc not in known_docs:
                    problems.append(
                        f"{where}: nieznany dokument {doc!r}. Dozwolone: {sorted(known_docs)}"
                    )

    return problems


def load_dataset(path: Path | None = None, *, strict_docs: bool = True) -> list[GoldenQuestion]:
    """Wczytuje i waliduje dataset. Rzuca ValueError przy pierwszym problemie."""
    dataset_path = path or _DATASET_PATH
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    questions = [
        GoldenQuestion(
            id=item.get("id", ""),
            question=item["question"],
            expected_answer=item.get("expected_answer", ""),
            expected_docs=item.get("expected_docs", []),
            category=item["category"],
            variant=item.get("variant", "standard"),
            expected_articles=item.get("expected_articles", []),
            source_note=item.get("source_note", ""),
        )
        for item in raw["questions"]
    ]

    problems = validate(questions, strict_docs=strict_docs)
    if problems:
        listing = "\n  - ".join(problems)
        raise ValueError(
            f"Golden dataset {dataset_path.name} ma {len(problems)} problemów:\n  - {listing}"
        )

    return questions


GOLDEN_DATASET: list[GoldenQuestion] = load_dataset()
