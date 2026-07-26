"""
Cross-encoder reranker wrapper.

Używa sentence-transformers CrossEncoder (CPU, ~90MB).
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sentence_transformers import CrossEncoder


@dataclass
class RankedResult:
    index: int  # pozycja w oryginalnej liście kandydatów
    score: float
    text: str


class CrossEncoderReranker:
    """
    Lazy-loaded cross-encoder. Model ładowany przy pierwszym wywołaniu
    (nie przy imporcie) — żeby CLI ingest nie czekał na ładowanie modelu.

    Usage
    -----
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    results  = reranker.rerank(query, candidates, top_n=5)
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[str],  # texty chunków
        top_n: int = 5,
    ) -> list[RankedResult]:
        """
        Zwraca top_n wyników posortowanych malejąco po score cross-encodera.
        """
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, text) for text in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            [
                RankedResult(index=i, score=float(s), text=candidates[i])
                for i, s in enumerate(scores)
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:top_n]
