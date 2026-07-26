import pytest

from tsl_rag.core.models import Chunk, DocumentMetadata, DocumentType, LegalHierarchyLevel
from tsl_rag.ingestion.embedders.embedder import _chunk_to_record, _make_batches

pytestmark = pytest.mark.unit


def _fake_chunk(cid: str) -> Chunk:
    m = DocumentMetadata(
        document_id="test",
        document_type=DocumentType.EU_REGULATION,
        title="Test",
        jurisdiction="EU",
        hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
        contains_table=False,
        contains_penalty=False,
        is_definition=False,
    )
    c = Chunk(chunk_id=cid, text="Sample text.", metadata=m)
    c.embedding = [0.1] * 768
    return c


def test_make_batches_correct_size():
    chunks = [_fake_chunk(str(i)) for i in range(35)]
    batches = _make_batches(chunks, 16)
    assert len(batches) == 3
    assert len(batches[0]) == 16
    assert len(batches[2]) == 3


def test_chunk_to_record_embedding_format():
    chunk = _fake_chunk("doc::0001")
    record = _chunk_to_record(chunk, "test-model")
    emb_str = record[15]  # $16 — embedding
    assert emb_str.startswith("[")
    assert emb_str.endswith("]")
    assert emb_str.count(",") == 767


def test_chunk_to_record_stores_embedding_model_in_metadata():
    """
    Model musi jechać razem z wektorem. Bez tego zmiana EMBEDDING_PROVIDER
    bez ponownego ingestu porównuje zapytanie z jednego modelu do dokumentów
    z innego — wyniki są bezsensowne, a nic nie zgłasza błędu.
    """
    import json

    record = _chunk_to_record(_fake_chunk("doc::0001"), "intfloat/multilingual-e5-base")
    metadata = json.loads(record[16])  # $17 — metadata JSONB
    assert metadata["embedding_model"] == "intfloat/multilingual-e5-base"
    assert metadata["embedding_dimensions"] == 768


def test_chunk_to_record_rejects_missing_embedding():
    chunk = _fake_chunk("doc::0001")
    chunk.embedding = None
    with pytest.raises(ValueError, match="nie ma embeddingu"):
        _chunk_to_record(chunk, "test-model")
