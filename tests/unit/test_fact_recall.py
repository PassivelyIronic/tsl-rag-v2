import pytest
from evals.run_retrieval_evals import fact_recall_at_k

from tsl_rag.core.models import Chunk, DocumentMetadata, DocumentType, LegalHierarchyLevel
from tsl_rag.retrieval.retriever import RetrievalResult

pytestmark = pytest.mark.unit


def _result(text: str, doc_id: str = "ec_561_2006") -> RetrievalResult:
    m = DocumentMetadata(
        document_id=doc_id,
        document_type=DocumentType.EU_REGULATION,
        title=f"Title {doc_id}",
        jurisdiction="EU",
        hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
        contains_table=False,
        contains_penalty=False,
        is_definition=False,
    )
    return RetrievalResult(chunk=Chunk(chunk_id=f"{doc_id}::x", text=text, metadata=m))


def test_all_facts_present_gives_one():
    results = [_result("Dzienny czas jazdy to 9 godzin, wyjątkowo 10 godzin.")]
    assert fact_recall_at_k(results, ["9 godzin", "10 godzin"], 5) == 1.0


def test_missing_fact_lowers_score():
    results = [_result("Dzienny czas jazdy to 9 godzin.")]
    assert fact_recall_at_k(results, ["9 godzin", "10 godzin"], 5) == 0.5


def test_only_first_k_chunks_count():
    """
    Sedno metryki: liczy się to, co faktycznie wejdzie do kontekstu.
    Fakt w chunku poza odcięciem jest dla generacji nieistniejący.
    """
    results = [_result("nic tu nie ma")] * 5 + [_result("kara wynosi 500")]
    assert fact_recall_at_k(results, ["500"], 5) == 0.0
    assert fact_recall_at_k(results, ["500"], 20) == 1.0


def test_numeric_fact_requires_digit_boundary():
    """
    Ta sama zasada co w ocenie odpowiedzi: oczekiwane "200" nie może
    potwierdzać się przez "2000". Kwoty w taryfikatorach to 50-12000,
    więc to nie jest przypadek brzegowy.
    """
    results = [_result("Kara wynosi 2000 zł.")]
    assert fact_recall_at_k(results, ["200"], 5) == 0.0
    assert fact_recall_at_k(results, ["2000"], 5) == 1.0


def test_diacritics_are_folded():
    results = [_result("Tygodniowy czas prowadzenia pojazdu.")]
    assert fact_recall_at_k(results, ["tygodniowy czas"], 5) == 1.0


def test_no_facts_is_neutral():
    assert fact_recall_at_k([_result("cokolwiek")], [], 5) == 1.0


def test_empty_results_with_facts_is_zero():
    assert fact_recall_at_k([], ["9 godzin"], 5) == 0.0
