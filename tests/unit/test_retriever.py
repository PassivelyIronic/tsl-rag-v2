import pytest

from tsl_rag.core.models import Chunk, DocumentMetadata, DocumentType, LegalHierarchyLevel
from tsl_rag.retrieval.retriever import _reciprocal_rank_fusion, _tokenize

pytestmark = pytest.mark.unit


def _fake_result(cid: str, dense: float = 0.0, bm25: float = 0.0):
    from tsl_rag.retrieval.retriever import RetrievalResult

    m = DocumentMetadata(
        document_id="test",
        document_type=DocumentType.EU_REGULATION,
        title="T",
        jurisdiction="EU",
        hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
        contains_table=False,
        contains_penalty=False,
        is_definition=False,
    )
    r = RetrievalResult(chunk=Chunk(chunk_id=cid, text=f"text {cid}", metadata=m))
    r.dense_score = dense
    r.bm25_score = bm25
    return r


def test_rrf_deduplicates_and_boosts_overlap():
    # Ten sam chunk wysoko w obu listach → powinien wygrać
    dense = [_fake_result("A", dense=0.9), _fake_result("B", dense=0.8)]
    bm25 = [_fake_result("A", bm25=10.0), _fake_result("C", bm25=8.0)]
    fused = _reciprocal_rank_fusion(dense, bm25)
    assert fused[0].chunk.chunk_id == "A"  # A wysoko w obu → wygrywa
    assert len(fused) == 3  # A, B, C — bez duplikatów


def test_tokenize_lowercases_and_splits():
    tokens = _tokenize("Article 4(1): Driver's rest.")
    assert "article" in tokens
    assert "4" in tokens
    assert "1" in tokens
    assert "driver" in tokens
    # Apostrofy i dwukropki usunięte
    assert "driver's" not in tokens


def test_tokenize_keeps_polish_words_whole():
    """
    Regresja: wzorzec [a-z0-9]+ rozrywał polskie słowa na diakrytykach —
    "prędkość" stawało się ["pr", "dko"]. Korpus i pytania są po polsku,
    więc to psuło leksykalną połowę retrievalu.
    """
    tokens = _tokenize("Maksymalna prędkość i czas odpoczynku kierowcy")
    assert "predkosc" in tokens
    assert "odpoczynku" in tokens
    assert "kierowcy" in tokens
    # Żaden token nie jest urwanym fragmentem słowa
    assert "pr" not in tokens
    assert "dko" not in tokens


def test_tokenize_folds_diacritics_both_ways():
    """
    Zapytanie bez ogonków musi trafić w tekst z ogonkami — użytkownik
    nietechniczny często pisze "predkosc", nie "prędkość".
    """
    assert _tokenize("prędkość") == _tokenize("predkosc")
    assert _tokenize("czas jazdy kierowcą") == _tokenize("czas jazdy kierowca")
    # "ł" nie ma rozkładu NFKD, więc wymaga osobnej obsługi
    assert _tokenize("łączny") == _tokenize("laczny")


def test_rrf_weights_shift_ranking():
    """
    Wagi RRF muszą faktycznie wpływać na ranking — do niedawna te dwa pola
    konfiguracji istniały, ale RRF ich nie czytał.
    """
    dense = [_fake_result("D", dense=0.9)]
    bm25 = [_fake_result("B", bm25=10.0)]

    dense_first = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.9, bm25_weight=0.1)
    bm25_first = _reciprocal_rank_fusion(dense, bm25, dense_weight=0.1, bm25_weight=0.9)

    assert dense_first[0].chunk.chunk_id == "D"
    assert bm25_first[0].chunk.chunk_id == "B"
