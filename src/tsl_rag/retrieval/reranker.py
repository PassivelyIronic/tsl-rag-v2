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

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    def load(self) -> CrossEncoder:
        """
        Ładuje model, jeśli jeszcze nie jest wczytany. Publiczne, bo API woła
        to przy starcie (prewarm), a nie dopiero przy pierwszym zapytaniu.
        """
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self.model_name} (max_length={self.max_length})")
            model = CrossEncoder(self.model_name, max_length=self.max_length)

            # Limit pozycji jest cechą architektury, nie konfiguracji. Modele
            # oparte na XLM-RoBERTa (m.in. bge-reranker-base) mają 514 i przy
            # dłuższym wejściu wywalają się w środku forward() komunikatem
            # "index 514 is out of bounds for dimension 1 with size 514",
            # który nic nie mówi o przyczynie. Przycinamy z ostrzeżeniem.
            limit = getattr(model.tokenizer, "model_max_length", None)
            if isinstance(limit, int) and 0 < limit < self.max_length:
                logger.warning(
                    f"{self.model_name} obsługuje najwyżej {limit} tokenów, "
                    f"a RERANKER_MAX_LENGTH={self.max_length}. Przycinam do {limit}. "
                    f"Dłuższe okno wymaga modelu, który je obsługuje (np. bge-reranker-v2-m3)."
                )
                model.max_length = limit
                self.max_length = limit

            self._model = model
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

        model = self.load()
        pairs = [(query, text) for text in candidates]
        # Wyciszenie poniżej: sygnatura predict() w sentence-transformers opisuje
        # pojedynczą parę lub listę pojedynczych wejść; lista par (query, doc)
        # to udokumentowany i jedyny sensowny sposób użycia cross-encodera,
        # ale nie mieści się w tej adnotacji.
        scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[arg-type]

        ranked = sorted(
            [
                RankedResult(index=i, score=float(s), text=candidates[i])
                for i, s in enumerate(scores)
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:top_n]
