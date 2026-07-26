from tsl_rag.core.models import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    LegalHierarchyLevel,
)
from tsl_rag.generation.generator import _build_context, _extract_citations
from tsl_rag.retrieval.retriever import RetrievalResult


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
    context, used = _build_context(results)
    assert "ec_561_2006" in context
    assert "Art." in context
    assert len(used) == 1


def test_build_context_respects_limit():
    # Jeden chunk z ogromnym tekstem przekraczającym limit
    big_text = "X" * 15_000
    results = [_fake_result("d::0001", "ec_561_2006", big_text)]
    context, used = _build_context(results)
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
