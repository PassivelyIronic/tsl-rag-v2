"""
Tag seed dumpu.

Sens tych testów: tag jest jedynym powiązaniem między kontraktem platformy
(`database.seed`) a zawartością bazy, którą kontrakt obiecuje. Fałszywy negatyw —
ten sam tag na dwie różne zawartości — nie objawia się błędem, tylko losowymi
wynikami retrievalu w środowisku, które „przecież ma poprawny seed".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from corpus_tag import TagInputs  # noqa: E402

pytestmark = pytest.mark.unit


def _inputs(**overrides) -> TagInputs:
    base = {
        "corpus": {"A.pdf": "aa", "B.pdf": "bb"},
        "pipeline": {"parser.py": "cc", "chunker.py": "dd"},
        "chunking": {"max_tokens": 450, "min_tokens": 60, "overlap_tokens": 60},
        "embedding": {
            "provider": "local",
            "model": "intfloat/multilingual-e5-base",
            "dimensions": 768,
            "passage_prefix": "passage: ",
        },
    }
    base.update(overrides)
    return TagInputs(**base)


def test_tag_has_expected_shape():
    tag = _inputs().tag()
    assert tag.startswith("corpus-")
    assert len(tag) == len("corpus-") + 12
    assert all(c in "0123456789abcdef" for c in tag.removeprefix("corpus-"))


def test_tag_is_deterministic():
    assert _inputs().tag() == _inputs().tag()


def test_tag_ignores_dict_insertion_order():
    """
    Kolejność wstawiania kluczy nie może zmieniać tagu — inaczej ten sam korpus
    dawałby inny tag zależnie od kolejności odczytu katalogu.
    """
    a = _inputs(corpus={"A.pdf": "aa", "B.pdf": "bb"})
    b = _inputs(corpus={"B.pdf": "bb", "A.pdf": "aa"})
    assert a.tag() == b.tag()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus", {"A.pdf": "aa", "B.pdf": "ZMIENIONE"}),
        ("pipeline", {"parser.py": "ZMIENIONE", "chunker.py": "dd"}),
        ("chunking", {"max_tokens": 400, "min_tokens": 60, "overlap_tokens": 60}),
    ],
)
def test_tag_changes_when_content_changes(field, value):
    assert _inputs(**{field: value}).tag() != _inputs().tag()


def test_tag_changes_when_parser_changes_but_chunker_does_not():
    """
    Drift 444 → 438 chunków pochodził ze zmiany obsługi miękkiego łącznika U+00AD,
    która mieszka w PARSERZE, nie w chunkerze. Hash obejmujący samą konfigurację
    chunkera przepuściłby dokładnie ten przypadek.
    """
    changed_parser = _inputs(pipeline={"parser.py": "PO_ZMIANIE", "chunker.py": "dd"})
    assert changed_parser.tag() != _inputs().tag()


def test_tag_changes_when_passage_prefix_changes():
    """
    Prefiks `passage:` wchodzi do treści embedowanej dla każdego chunka, więc jego
    zmiana zmienia każdy wektor w dumpie, nie ruszając ani jednego bajtu korpusu.
    """
    embedding = dict(_inputs().embedding)
    embedding["passage_prefix"] = ""
    assert _inputs(embedding=embedding).tag() != _inputs().tag()


def test_tag_changes_when_embedding_model_changes():
    embedding = dict(_inputs().embedding)
    embedding["model"] = "BAAI/bge-m3"
    assert _inputs(embedding=embedding).tag() != _inputs().tag()
