from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from loguru import logger

from tsl_rag.core.models import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    LegalHierarchyLevel,
)
from tsl_rag.ingestion.parsers.legal_pdf_parser import ParsedElement

MAX_TOKENS: int = 450
MIN_TOKENS: int = 60
OVERLAP_TOKENS: int = 60

_CHARS_PER_TOKEN: float = 4.2


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _last_n_token_chars(text: str, n_tokens: int) -> str:
    chars = int(n_tokens * _CHARS_PER_TOKEN)
    return text[-chars:] if len(text) > chars else text


@dataclass
class _ArticleBuffer:
    chapter: str | None
    article: str | None
    elements: list[ParsedElement] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements)

    def tokens(self) -> int:
        return _approx_tokens(self.text())

    def has_table(self) -> bool:
        return any(e.contains_table for e in self.elements)

    def page_range(self) -> tuple[int | None, int | None]:
        pages = [e.page_number for e in self.elements if e.page_number]
        return (min(pages), max(pages)) if pages else (None, None)


class LegalChunker:
    def __init__(
        self,
        document_id: str,
        document_type: DocumentType,
        document_title: str,
        jurisdiction: str = "EU",
        max_tokens: int = MAX_TOKENS,
        min_tokens: int = MIN_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> None:
        self.document_id = document_id
        self.document_type = document_type
        self.document_title = document_title
        self.jurisdiction = jurisdiction
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, elements: Sequence[ParsedElement]) -> list[Chunk]:
        """
        Main entry point.  Returns an ordered list of Chunk objects.
        """
        buffers = self._group_into_article_buffers(elements)
        buffers = self._merge_short_buffers(buffers)

        chunks: list[Chunk] = []
        for buf in buffers:
            chunks.extend(self._split_buffer(buf))

        # Assign stable, deterministic IDs
        for idx, chunk in enumerate(chunks):
            chunk.chunk_id = f"{self.document_id}::{idx:04d}"

        logger.info(
            f"[{self.document_id}] {len(elements)} elements → "
            f"{len(buffers)} article buffers → {len(chunks)} chunks"
        )
        return chunks

    def _group_into_article_buffers(
        self, elements: Sequence[ParsedElement]
    ) -> list[_ArticleBuffer]:
        buffers: list[_ArticleBuffer] = []
        current_buffer: _ArticleBuffer | None = None

        for elem in elements:
            # Tables are always atomic — their own buffer regardless of article
            if elem.hierarchy_level == LegalHierarchyLevel.TABLE:
                if current_buffer:
                    buffers.append(current_buffer)
                    current_buffer = None
                table_buf = _ArticleBuffer(chapter=elem.chapter, article=elem.article)
                table_buf.elements.append(elem)
                buffers.append(table_buf)
                continue

            article_key = (elem.chapter, elem.article)
            if current_buffer is None:
                current_buffer = _ArticleBuffer(chapter=elem.chapter, article=elem.article)
            elif (current_buffer.chapter, current_buffer.article) != article_key:
                buffers.append(current_buffer)
                current_buffer = _ArticleBuffer(chapter=elem.chapter, article=elem.article)

            current_buffer.elements.append(elem)

        if current_buffer:
            buffers.append(current_buffer)

        return buffers

    def _merge_short_buffers(self, buffers: list[_ArticleBuffer]) -> list[_ArticleBuffer]:
        if not buffers:
            return buffers

        merged: list[_ArticleBuffer] = []
        i = 0
        while i < len(buffers):
            current = buffers[i]

            # Never merge table buffers
            if current.has_table():
                merged.append(current)
                i += 1
                continue

            if (
                current.tokens() < self.min_tokens
                and i + 1 < len(buffers)
                and not buffers[i + 1].has_table()
                and buffers[i + 1].chapter == current.chapter
            ):
                next_buf = buffers[i + 1]
                combined = _ArticleBuffer(
                    chapter=current.chapter,
                    # Keep the article of the first (or None for preamble text)
                    article=current.article or next_buf.article,
                    elements=current.elements + next_buf.elements,
                )
                buffers[i + 1] = combined  # replace next in-place, skip current
                i += 1
                continue

            merged.append(current)
            i += 1

        return merged

    def _split_buffer(self, buf: _ArticleBuffer) -> list[Chunk]:
        """
        If the buffer fits within max_tokens → one chunk.
        Otherwise → sliding-window split with overlap.
        Tables always produce a single chunk.
        """
        full_text = buf.text()

        if buf.has_table():
            return [self._make_chunk(full_text, buf, is_table=True)]

        if _approx_tokens(full_text) <= self.max_tokens:
            return [self._make_chunk(full_text, buf)]

        segments = re.split(r"\n{2,}", full_text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        overlap_tail = ""  # carried forward between windows

        for seg in segments:
            seg_tokens = _approx_tokens(seg)

            # Segment itself exceeds max → force-split on sentence boundary
            if seg_tokens > self.max_tokens:
                # Flush current window first
                if current_parts:
                    window_text = overlap_tail + "\n\n".join(current_parts)
                    chunks.append(self._make_chunk(window_text.strip(), buf))
                    overlap_tail = _last_n_token_chars(window_text, self.overlap_tokens)
                    current_parts = []
                    current_tokens = 0

                sub_chunks = self._sentence_split(seg, buf, overlap_tail)
                if sub_chunks:
                    chunks.extend(sub_chunks)  # <--- POPRAWKA (Dodajemy wszystkie)
                    overlap_tail = _last_n_token_chars(sub_chunks[-1].text, self.overlap_tokens)
                    current_parts = []
                    current_tokens = 0
                continue

            if current_tokens + seg_tokens > self.max_tokens:
                # Emit current window
                window_text = overlap_tail + "\n\n".join(current_parts)
                chunks.append(self._make_chunk(window_text.strip(), buf))
                overlap_tail = _last_n_token_chars(window_text, self.overlap_tokens)
                current_parts = [seg]
                current_tokens = seg_tokens
            else:
                current_parts.append(seg)
                current_tokens += seg_tokens

        # Flush remainder
        if current_parts:
            window_text = overlap_tail + "\n\n".join(current_parts)
            chunks.append(self._make_chunk(window_text.strip(), buf))

        return chunks

    def _sentence_split(self, text: str, buf: _ArticleBuffer, overlap_tail: str) -> list[Chunk]:
        """Last-resort: split oversized segment on sentence boundaries."""
        sentences = re.split(r"(?<=[.;])\s+", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        tail = overlap_tail

        for sent in sentences:
            t = _approx_tokens(sent)
            if current_tokens + t > self.max_tokens and current:
                window = tail + " ".join(current)
                chunks.append(self._make_chunk(window.strip(), buf))
                tail = _last_n_token_chars(window, self.overlap_tokens)
                current = [sent]
                current_tokens = t
            else:
                current.append(sent)
                current_tokens += t

        if current:
            window = tail + " ".join(current)
            chunks.append(self._make_chunk(window.strip(), buf))

        return chunks

    def _make_chunk(
        self,
        text: str,
        buf: _ArticleBuffer,
        is_table: bool = False,
    ) -> Chunk:
        page_start, page_end = buf.page_range()
        metadata = DocumentMetadata(
            document_id=self.document_id,
            document_type=self.document_type,
            title=self.document_title,
            jurisdiction=self.jurisdiction,
            chapter=buf.chapter,
            article=buf.article,
            hierarchy_level=(
                LegalHierarchyLevel.TABLE if is_table else LegalHierarchyLevel.PARAGRAPH
            ),
            contains_table=is_table,
            contains_penalty=is_table,  # tables in TSL docs == fine tariffs
            is_definition=bool(
                re.search(
                    r"na potrzeby niniejszego|for the purposes of|means ",
                    text,
                    re.I,
                )
            ),
            page_start=page_start,
            page_end=page_end,
        )

        return Chunk(chunk_id="", text=text, metadata=metadata)
