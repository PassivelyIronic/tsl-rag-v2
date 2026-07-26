"""
Parser prawnych dokumentów PDF.

Strategia:
- pdfplumber → tabele (taryfikatory z grzywnami)
- pymupdf   → szybka ekstrakcja tekstu + wykrywanie struktury
- Regex     → Chapter / Article / Paragraph hierarchy
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
from loguru import logger

from tsl_rag.core.models import DocumentMetadata, DocumentType, LegalHierarchyLevel

CHAPTER_RE = re.compile(r"^(ROZDZIAŁ|CHAPTER)\s+([IVXLCDM]+|\d+)", re.I)
ARTICLE_RE = re.compile(r"^Artykuł\s+(\d+[a-z]?)\s*$", re.I)
PARA_RE = re.compile(r"^\s*(\d+)\.\s+\S")

# Miękki łącznik (U+00AD) wraz z białymi znakami po nim. PDF-y EUR-Lexu
# zawierają go w każdym miejscu podziału wiersza, więc po ekstrakcji tekst
# zawiera "przynaj­ mniej", "wyko­ rzystać", "tygodnio­ wego".
_SOFT_HYPHEN_BREAK = re.compile(r"\xad\s*")

# Wielokrotne spacje i tabulatory wewnątrz wiersza. Ekstrakcja zostawia
# podwójne spacje po liczbach ("9  godzin"), co utrudnia dopasowania dosłowne.
# Podział na wiersze zachowujemy — na nim opiera się wykrywanie hierarchii.
_INLINE_SPACES = re.compile(r"[ \t]{2,}")


def normalize_pdf_text(text: str) -> str:
    """
    Sprząta artefakty ekstrakcji z PDF-a, zanim tekst pójdzie dalej.

    Powód, dla którego to nie jest kosmetyka: miękki łącznik rozrywa słowo
    na dwa fragmenty, a tokenizer BM25 robi z nich dwa bezużyteczne tokeny
    ("tygodnio" + "wego"). Poprawne zapytanie nie ma wtedy jak trafić w to
    miejsce, a model dostaje w kontekście tekst z połamanymi słowami.
    W korpusie przed naprawą: 1258 wystąpień w 307 z 444 chunków.

    Zmiana wpływa na treść chunków, więc wymaga ponownego ingestu i pomiaru
    retrievalu przed i po (CLAUDE.md §9).
    """
    without_hyphens = _SOFT_HYPHEN_BREAK.sub("", text)
    lines = [_INLINE_SPACES.sub(" ", line).strip() for line in without_hyphens.splitlines()]
    return "\n".join(lines).strip()


@dataclass
class ParsedElement:
    text: str
    hierarchy_level: LegalHierarchyLevel
    chapter: str | None = None
    article: str | None = None
    paragraph: str | None = None
    contains_table: bool = False
    raw_table: list[list[str]] = field(default_factory=list)
    page_number: int | None = None


class LegalPDFParser:
    """
    Parsuje PDF do sekwencji ParsedElement z pełnym kontekstem hierarchicznym.
    Tabele wykrywane przez pdfplumber, tekst przez pymupdf.
    """

    def __init__(self, doc_type: DocumentType) -> None:
        self.doc_type = doc_type

    def parse(self, pdf_path: Path) -> list[ParsedElement]:
        logger.info(f"Parsing {pdf_path.name}")
        table_pages = self._extract_tables(pdf_path)
        elements = self._extract_text_elements(pdf_path, table_pages)
        logger.info(f"  → {len(elements)} elements extracted")
        return elements

    # ------------------------------------------------------------------

    def _extract_tables(self, pdf_path: Path) -> dict[int, list[list[str]]]:
        tables: dict[int, list[list[str]]] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_tables()
                if raw:
                    merged = [[cell or "" for cell in row] for tbl in raw for row in tbl]
                    tables[page.page_number] = merged
        return tables

    def _extract_text_elements(
        self,
        pdf_path: Path,
        table_pages: dict[int, list[list[str]]],
    ) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        current_chapter: str | None = None
        current_article: str | None = None
        current_para: str | None = None

        doc = fitz.open(str(pdf_path))

        for page in doc:
            page_num = page.number + 1

            # Jeśli strona ma tabelę → emit jako osobny element
            if page_num in table_pages:
                elements.append(
                    ParsedElement(
                        text=self._table_to_markdown(table_pages[page_num]),
                        hierarchy_level=LegalHierarchyLevel.TABLE,
                        chapter=current_chapter,
                        article=current_article,
                        contains_table=True,
                        raw_table=table_pages[page_num],
                        page_number=page_num,
                    )
                )
                continue

            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
            for block in blocks:
                text = normalize_pdf_text(block[4])
                if not text or len(text) < 3:
                    continue

                first_line = text.split("\n")[0].strip()

                if CHAPTER_RE.match(first_line):
                    current_chapter = first_line
                    current_article = None
                    current_para = None
                    elements.append(
                        ParsedElement(
                            text=text,
                            hierarchy_level=LegalHierarchyLevel.CHAPTER,
                            chapter=current_chapter,
                            page_number=page_num,
                        )
                    )

                elif ARTICLE_RE.match(first_line):
                    current_article = first_line
                    current_para = None
                    elements.append(
                        ParsedElement(
                            text=text,
                            hierarchy_level=LegalHierarchyLevel.ARTICLE,
                            chapter=current_chapter,
                            article=current_article,
                            page_number=page_num,
                        )
                    )

                else:
                    if PARA_RE.match(text) and current_article:
                        num = PARA_RE.match(text).group(1)  # type: ignore[union-attr]
                        art_num = current_article.split()[-1]
                        current_para = f"{art_num}({num})"

                    elements.append(
                        ParsedElement(
                            text=text,
                            hierarchy_level=LegalHierarchyLevel.PARAGRAPH,
                            chapter=current_chapter,
                            article=current_article,
                            paragraph=current_para,
                            page_number=page_num,
                        )
                    )

        doc.close()
        return elements

    @staticmethod
    def _table_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""

        # Komórki też przechodzą normalizację — taryfikatory kar są prawie
        # wyłącznie tabelami, a miękkie łączniki i podziały wiersza siedzą
        # w środku opisów naruszeń, czyli dokładnie tam, gdzie pada dopasowanie.
        def cell(value: str | None) -> str:
            return normalize_pdf_text(value or "").replace("\n", " ")

        clean = [[cell(c) for c in row] for row in rows]
        header = "| " + " | ".join(clean[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(clean[0])) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in clean[1:])
        return "\n".join([header, sep, body])


def build_metadata(
    elem: ParsedElement,
    document_id: str,
    document_type: DocumentType,
    document_title: str,
    jurisdiction: str = "EU",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        document_type=document_type,
        title=document_title,
        jurisdiction=jurisdiction,
        chapter=elem.chapter,
        article=elem.article,
        paragraph=elem.paragraph,
        hierarchy_level=elem.hierarchy_level,
        contains_table=elem.contains_table,
        contains_penalty=elem.contains_table,
        is_definition=bool(
            re.search(r"na potrzeby niniejszego|for the purposes of", elem.text, re.I)
        ),
    )
