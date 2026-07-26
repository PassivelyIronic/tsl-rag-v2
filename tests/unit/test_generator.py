import pytest

from tsl_rag.core.models import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    LegalHierarchyLevel,
)
from tsl_rag.generation.generator import _build_context, _extract_citations
from tsl_rag.retrieval.retriever import RetrievalResult

pytestmark = pytest.mark.unit

# Limit kontekstu jest teraz parametrem _build_context (pochodzi z Settings),
# a nie stałą modułową — testy podają go wprost, żeby nie zależeć od .env.
_CTX_LIMIT = 12_000


def _fake_result(cid: str, doc_id: str, text: str, article: str = "6") -> RetrievalResult:
    m = DocumentMetadata(
        document_id=doc_id,
        document_type=DocumentType.EU_REGULATION,
        title=f"Title {doc_id}",
        jurisdiction="EU",
        hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
        article=article,
        contains_table=False,
        contains_penalty=False,
        is_definition=False,
    )
    r = RetrievalResult(chunk=Chunk(chunk_id=cid, text=text, metadata=m))
    r.rerank_score = 0.9
    return r


def test_build_context_includes_header():
    results = [_fake_result("d::0001", "ec_561_2006", "Daily driving limit is 9h.")]
    context, used = _build_context(results, _CTX_LIMIT)
    assert "ec_561_2006" in context
    assert "Art." in context
    assert len(used) == 1


def test_build_context_respects_limit():
    # Jeden chunk z ogromnym tekstem przekraczającym limit
    big_text = "X" * 15_000
    results = [_fake_result("d::0001", "ec_561_2006", big_text)]
    context, used = _build_context(results, _CTX_LIMIT)
    assert len(used) == 0  # za duży → nie wchodzi do kontekstu


def test_extract_citations_parses_format():
    results = [_fake_result("d::0001", "ec_561_2006", "text", article="6(1)")]
    answer = "Driving limit is 9h. [ec_561_2006 | Art. 6(1)]"
    cits = _extract_citations(answer, results)
    assert len(cits) == 1
    assert cits[0].document_id == "ec_561_2006"


def test_extract_citations_deduplicates():
    results = [_fake_result("d::0001", "ec_561_2006", "text")]
    answer = "[ec_561_2006 | Art. 6] and again [ec_561_2006 | Art. 6]"
    cits = _extract_citations(answer, results)
    assert len(cits) == 1


def test_extract_citations_reports_article_from_answer_not_metadata():
    """
    Regresja: numer artykułu brany był z metadanych pierwszego chunka
    dokumentu, więc odpowiedź cytująca Art. 6 raportowana była jako Art. 11,
    jeśli chunk z artykułem 11 stał wyżej w retrievalu. Cytowanie wskazujące
    inny przepis niż tekst odpowiedzi jest gorsze niż brak cytowania.
    """
    results = [
        _fake_result("d::0001", "ec_561_2006", "wyjątki", article="11"),
        _fake_result("d::0002", "ec_561_2006", "czas jazdy", article="6"),
    ]
    cits = _extract_citations("Limit to 9 godzin. [ec_561_2006 | Art. 6]", results)
    assert len(cits) == 1
    assert cits[0].article == "6"
    # Powiązany chunk to ten z pasującym artykułem, nie pierwszy z listy
    assert cits[0].chunk_id == "d::0002"


def test_extract_citations_parses_paragraph_and_polish_spelling():
    results = [_fake_result("d::0001", "tariff_driver_2022", "kary", article=None)]
    cits = _extract_citations("Kara wynosi 500 PLN. [tariff_driver_2022 | ust. 3]", results)
    assert len(cits) == 1
    assert cits[0].paragraph == "3"

    results2 = [_fake_result("d::0002", "aetr", "zakres", article="2")]
    cits2 = _extract_citations("Dotyczy państw trzecich. [aetr | Artykuł 2]", results2)
    assert cits2[0].article == "2"


def test_extract_citations_skips_document_absent_from_context():
    """
    Model potrafi zacytować dokument, którego nie dostał. Takiego cytowania
    nie ma czym potwierdzić, więc nie trafia na listę źródeł.
    """
    results = [_fake_result("d::0001", "ec_561_2006", "text")]
    cits = _extract_citations("Nieprawda. [rozporzadzenie_zmyslone | Art. 1]", results)
    assert cits == []
