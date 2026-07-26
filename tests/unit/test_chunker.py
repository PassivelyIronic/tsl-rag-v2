from tsl_rag.core.models import DocumentType, LegalHierarchyLevel
from tsl_rag.ingestion.chunkers.legal_chunker import LegalChunker  # ← poprawiony import
from tsl_rag.ingestion.parsers.legal_pdf_parser import ParsedElement


def _make_elem(text: str, article: str = "Article 4") -> ParsedElement:
    return ParsedElement(
        text=text,
        hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
        chapter="CHAPTER II",
        article=article,
        page_number=1,
    )


def test_short_article_produces_one_chunk():
    chunker = LegalChunker("test_doc", DocumentType.EU_REGULATION, "Test")
    elems = [_make_elem("Short paragraph.")]
    chunks = chunker.chunk(elems)
    assert len(chunks) == 1
    assert chunks[0].metadata.article == "Article 4"


def test_long_article_splits_and_overlaps():
    long_text = "Word word word. " * 150

    chunker = LegalChunker(
        "test_doc", DocumentType.EU_REGULATION, "Test", max_tokens=200, overlap_tokens=40
    )
    elems = [_make_elem(long_text)]
    chunks = chunker.chunk(elems)
    assert len(chunks) >= 3
    end_of_first = chunks[0].text[-100:]
    start_of_second = chunks[1].text[:200]
    assert any(w in start_of_second for w in end_of_first.split()[-5:])


def test_table_is_never_split():
    table_elem = ParsedElement(
        text="| Naruszenie | Kara |\n|---|---|\n| Row 1 | 500 PLN |",
        hierarchy_level=LegalHierarchyLevel.TABLE,
        contains_table=True,
        chapter="CHAPTER III",
        article=None,
        page_number=5,
    )
    chunker = LegalChunker("test_doc", DocumentType.EU_REGULATION, "Test", max_tokens=10)
    chunks = chunker.chunk([table_elem])
    assert len(chunks) == 1
    assert chunks[0].metadata.contains_table is True
