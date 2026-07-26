from tsl_rag.core.models import Chunk, DocumentMetadata, DocumentType, LegalHierarchyLevel
from tsl_rag.retrieval.retriever import _reciprocal_rank_fusion, _tokenize


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
