"""
Tag seed dumpu, liczony z ZAWARTOŚCI wejść, nie z SHA commita.

Po co
-----
Dump `document_chunks` jest publikowany jako artefakt OCI obok obrazu API i pinowany
w kontrakcie platformy (`database.seed`). Gdyby tagować go SHA obrazu, każdy commit
w kodzie API produkowałby nowy artefakt o identycznej zawartości — albo, po
optymalizacji tego marnotrawstwa, ten sam tag zacząłby wskazywać na dwie różne
zawartości. Tag musi zależeć wyłącznie od tego, co realnie zmienia wiersze w dumpie
(DECISIONS D-016).

Co wchodzi do hasha
-------------------
1. **Korpus** — nazwa i sha256 każdego pliku w `data/raw/`.
2. **Ścieżka produkująca tekst** — bajty `legal_pdf_parser.py` i `legal_chunker.py`.
   D-016 mówi o „konfiguracji chunkera", ale dowód, na którym ta decyzja stoi
   (drift 444 → 438 chunków), pochodzi ze zmiany obsługi miękkiego łącznika U+00AD,
   a ta mieszka w PARSERZE. Hash obejmujący tylko chunker przepuściłby dokładnie ten
   przypadek, którym uzasadniono jego istnienie.
3. **Parametry chunkowania** — `chunker_max_tokens`, `chunker_min_tokens`,
   `chunker_overlap_tokens`.
4. **Embeddingi** — aktywny provider, nazwa modelu, liczba wymiarów i prefiks
   `passage`. Prefiks jest tu nieoczywisty, ale wchodzi do treści embedowanej dla
   każdego chunka, więc jego zmiana zmienia każdy wektor w dumpie, nie zmieniając
   ani jednego znaku w `data/raw/`.

Czego świadomie NIE ma
----------------------
- SHA commita, wersji obrazu, numeru builda — patrz wyżej.
- `POSTGRES_DSN` i reszty konfiguracji runtime — nie wpływają na zawartość.
- Pominięcia duplikatu `TARIFF_EMPLOYER_2022.pdf`. Plik jest bajtowo identyczny
  z `TARIFF_COMPANY_2022.pdf` i ingest go pomija, ale hashujemy KAŻDY plik w katalogu.
  Jego usunięcie zmieni więc tag mimo niezmienionej zawartości dumpu. To jest fałszywy
  alarm — nowy tag na tę samą zawartość — i jest akceptowalny, bo kosztuje jeden zbędny
  przebieg ingestu. Fałszywy negatyw, czyli ten sam tag na dwie różne zawartości,
  akceptowalny nie jest, a filtrowanie po `_KNOWN_DUPLICATES` wymagałoby importu
  `ingestion.cli`, który ciągnie parsery PDF z grupy `ingest`.

Użycie
------
    uv run python scripts/corpus_tag.py               # sam tag: corpus-a41f9c1b2d3e
    uv run python scripts/corpus_tag.py --explain     # tag + rozbicie na wejścia
    uv run python scripts/corpus_tag.py --json        # do konsumpcji przez CI
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tsl_rag.core.console import ensure_utf8_output  # noqa: E402
from tsl_rag.core.settings import Settings  # noqa: E402

ensure_utf8_output()

# Pliki, których treść decyduje o tym, jaki tekst trafia do chunków.
_PIPELINE_SOURCES = (
    Path("src/tsl_rag/ingestion/parsers/legal_pdf_parser.py"),
    Path("src/tsl_rag/ingestion/chunkers/legal_chunker.py"),
)

_CORPUS_DIR = Path("data/raw")
_TAG_PREFIX = "corpus-"

# 12 znaków heksadecymalnych. D-016 podaje przykład sześcioznakowy, ale jako
# ilustrację formatu, nie jako wymaganie; 48 bitów kosztuje sześć znaków w nazwie
# artefaktu i zdejmuje pytanie o kolizje z listy rzeczy do przemyślenia.
_TAG_LENGTH = 12

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class TagInputs:
    """Wszystko, co wchodzi do hasha — w postaci, którą da się wypisać i porównać."""

    corpus: dict[str, str]
    pipeline: dict[str, str]
    chunking: dict[str, int]
    embedding: dict[str, str | int]

    def canonical(self) -> str:
        """
        Kanoniczna reprezentacja hashowanych wejść.

        `sort_keys` i jawny separator, bo tag musi być identyczny niezależnie od
        wersji Pythona i kolejności wstawiania kluczy do słownika.
        """
        return json.dumps(
            {
                "corpus": self.corpus,
                "pipeline": self.pipeline,
                "chunking": self.chunking,
                "embedding": self.embedding,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def tag(self) -> str:
        digest = hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()
        return f"{_TAG_PREFIX}{digest[:_TAG_LENGTH]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_settings() -> Settings:
    """
    Ustawienia rozwiązywane tak samo jak przy ingeście, żeby tag opisywał
    konfigurację, która faktycznie wyprodukuje dump.

    `postgres_dsn` jest polem wymaganym, a do policzenia tagu nie jest potrzebny
    i nie wchodzi do hasha — dlatego przy jego braku podstawiamy wartość zastępczą,
    zamiast wymagać działającej bazy do policzenia sumy kontrolnej plików.
    """
    if not os.getenv("POSTGRES_DSN"):
        return Settings(postgres_dsn="postgresql+asyncpg://unused/unused")
    return Settings()


def collect_inputs(repo_root: Path | None = None) -> TagInputs:
    root = repo_root or _REPO_ROOT
    settings = _load_settings()

    corpus_dir = root / _CORPUS_DIR
    if not corpus_dir.is_dir():
        raise typer.BadParameter(f"nie ma katalogu korpusu: {corpus_dir}")

    corpus_files = sorted(p for p in corpus_dir.iterdir() if p.is_file())
    if not corpus_files:
        raise typer.BadParameter(f"katalog korpusu jest pusty: {corpus_dir}")

    corpus = {p.name: _sha256_file(p) for p in corpus_files}

    pipeline: dict[str, str] = {}
    for relative in _PIPELINE_SOURCES:
        source = root / relative
        if not source.is_file():
            # Twardy błąd, nie pominięcie: cicha zmiana tagu przy przeniesieniu
            # pliku byłaby dokładnie tym rozjazdem, któremu ten skrypt zapobiega.
            raise typer.BadParameter(f"brak pliku ścieżki produkującej tekst: {relative}")
        pipeline[relative.as_posix()] = _sha256_file(source)

    return TagInputs(
        corpus=corpus,
        pipeline=pipeline,
        chunking={
            "max_tokens": settings.chunker_max_tokens,
            "min_tokens": settings.chunker_min_tokens,
            "overlap_tokens": settings.chunker_overlap_tokens,
        },
        embedding={
            "provider": settings.embedding_provider,
            "model": settings.active_embedding_model,
            "dimensions": settings.embedding_dimensions,
            "passage_prefix": settings.local_embed_passage_prefix,
        },
    )


@app.command()
def main(
    explain: bool = typer.Option(False, "--explain", help="Wypisz rozbicie hasha na wejścia."),
    as_json: bool = typer.Option(False, "--json", help="Wypisz tag i wejścia jako JSON."),
) -> None:
    """Liczy tag seed dumpu z zawartości korpusu, ścieżki tekstowej i konfiguracji embeddingów."""
    inputs = collect_inputs()
    tag = inputs.tag()

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "tag": tag,
                    "corpus": inputs.corpus,
                    "pipeline": inputs.pipeline,
                    "chunking": inputs.chunking,
                    "embedding": inputs.embedding,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not explain:
        typer.echo(tag)
        return

    typer.echo(f"tag: {tag}\n")
    typer.echo(f"korpus ({len(inputs.corpus)} plików):")
    for name, digest in inputs.corpus.items():
        typer.echo(f"  {digest[:12]}  {name}")
    typer.echo("\nścieżka produkująca tekst:")
    for name, digest in inputs.pipeline.items():
        typer.echo(f"  {digest[:12]}  {name}")
    typer.echo("\nchunkowanie:")
    for key, value in inputs.chunking.items():
        typer.echo(f"  {key} = {value}")
    typer.echo("\nembeddingi:")
    for key, value in inputs.embedding.items():
        typer.echo(f"  {key} = {value!r}")


if __name__ == "__main__":
    app()
