# TSL-RAG

[![CI](https://github.com/PassivelyIronic/tsl-rag-v2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PassivelyIronic/tsl-rag-v2/actions/workflows/ci.yml)

Hybrydowy RAG nad prawem transportowym UE i PL — czas pracy kierowców, kabotaż,
taryfikatory kar. Odpowiada po polsku i **cytuje akt oraz artykuł** przy każdym fakcie.

Zbudowany pod jedną osobę nietechniczną, na darmowej infrastrukturze, bez lokalnego GPU.

```
pytanie
  → embedding zapytania (multilingual-e5-base, CPU, in-process)
  → dense (pgvector) ─┐
  → BM25 (in-memory) ─┴→ ważony RRF (k=5) → top-5
  → prompt z kontekstem (limit w znakach)
  → generacja (łańcuch providerów z fallbackiem)
  → odpowiedź + cytowania [doc_id | Art. X]
```

## Stan

| | |
|---|---|
| Korpus | 438 chunków, 13 aktów prawnych |
| `recall@5` / MRR | **0.958** / **0.874** |
| Fakty w kontekście (`fakty@5`) | **0.882** |
| Latencja retrievalu | 94 ms (rozgrzany proces) |
| Latencja odpowiedzi | ~5 s (mediana 4.0 s) |
| Koszt | 0 USD |
| Testy | 151 (unit, integration, e2e) |

Wszystkie liczby pochodzą z uruchomień zapisanych w `evals/results/`.

## Problem

Odpowiedź na pytanie „ile wynosi kara za przekroczenie dziennego czasu jazdy" wymaga
sięgnięcia do właściwego z **czterech** taryfikatorów i powiązania go z rozporządzeniem,
które ten limit ustala. Płaskie wyszukiwanie wektorowe myli akty o zbliżonej tematyce,
a pewna siebie odpowiedź z błędnym cytowaniem jest tu gorsza niż odmowa — nikt jej
nie zweryfikuje.

## Uruchomienie

```bash
uv sync                     # środowisko
docker compose up -d        # Postgres + pgvector (host: 5433)
cp env.example .env         # uzupełnij OPENROUTER_API_KEY

uv sync --extra ingest      # parsery PDF — tylko do ingestu
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/

uv run python -m tsl_rag.api.main    # API na :8000
uv run streamlit run ui.py           # UI (wymaga API)
```

Ewaluacja retrievalu jest darmowa i nie wymaga klucza providera:

```bash
uv run python -m evals.run_retrieval_evals   # ~40 s, exit 1 poniżej progu
```

## Decyzje projektowe

Każda z poniższych wynika z pomiaru, nie z preferencji. Surowe wyniki wszystkich
przebiegów leżą w [`evals/results/`](evals/results/) i są wersjonowane.

**Hybrydowy retrieval, nie sam wektor.** Numery artykułów i kwoty kar to dopasowania
dosłowne, w których embeddingi są słabe. BM25 osiąga tu `recall@5` = 0.948 samodzielnie,
dense — 0.854.

**Reranking jest wyłączony.** Cross-encoder kosztował całość latencji retrievalu: bez niego
0.1 s przy `recall@5` = 0.938, najlepszy zmierzony wariant 0.969 za 43 s. Model i okno
wskazują najlepszy wariant, więc powrót to jedna zmienna.

**Stała RRF wynosi 5, nie literaturowe 60.** Przy dwóch listach po 20 pozycji `k=60`
spłaszcza ranking (ranga 1 dostaje 1/61, ranga 20 — 1/80), więc pozycja przestaje się liczyć
i wygrywa sama zgodność list. Chunk oceniony przez jedną listę na 3. miejscu, nieobecny
w drugiej, lądował po fuzji na 10. Zmiana dała `recall@5` 0.938 → 0.958 i `fakty@5`
0.840 → 0.882, bez regresji w żadnej kategorii.

**Recall liczony jest dwojako.** `recall@k` mówi tylko, że właściwy AKT trafił do kontekstu.
`fact_recall@k` sprawdza, czy w treści pobranych chunków stoi oczekiwany fakt — czyli czy
trafił właściwy PRZEPIS. Ta druga metryka wychwyciła zmianę, która podnosiła recall
dokumentowy o 0.031, jednocześnie wyrzucając odpowiedź z kontekstu w 11 pp przypadków.

**Rozumowanie modelu wyłącza token w promptcie, nie parametr API.** Na
`nemotron-nano-9b-v2` parametr `reasoning` nie działa mimo deklaracji providera;
`/no_think` zbija latencję z 25.7 s do 5.0 s i poprawia wszystkie metryki treści.

**Embeddingi liczą się lokalnie na CPU**, nie przez API. Zapytanie embeduje się przy każdym
pytaniu, więc jest to najbardziej wrażliwy na awarię punkt pipeline'u — brak zależności
sieciowej i rate limitu jest tu wart 84 ms.

**Generacja ma łańcuch fallbacku.** Przejście na kolejne ogniwo następuje przy
404/400/429/5xx, timeoucie i przy **pustej odpowiedzi**. Odmowa awarią nie jest — szukanie
kolejnego modelu po odmowie to szukanie modelu skłonnego halucynować.

**Cytowanie jest funkcją krytyczną.** Odpowiedź bez cytowania albo z cytowaniem
niewłaściwego aktu traktowana jest jako porażka, nawet gdy treść brzmi sensownie. Cache
nie zapisuje odpowiedzi bez cytowań ani odmów.

**Progi jakości są wersjonowane** w [`evals/thresholds.yaml`](evals/thresholds.yaml)
i bramkują `run_retrieval_evals` kodem wyjścia. Progu nie obniża się po to, żeby przebieg
przeszedł. Bramkują metryki retrievalu, bo metryki generacji mają przy tym rozmiarze zbioru
rozrzut do 0.133 między przebiegami identycznego kodu.

## Architektura

| Warstwa | Wybór | Dlaczego |
|---|---|---|
| Retrieval | pgvector + rank-bm25 + RRF | Leksyka i semantyka łapią różne pytania |
| Embeddingi | `multilingual-e5-base`, CPU | Bez GPU, bez API, bez limitu |
| Generacja | OpenRouter (model darmowy) | Jedyna realnie darmowa opcja bez karty |
| API | FastAPI, osobny serwis | Probe'y, metryki, przyszły HPA |
| UI | Streamlit | Jeden użytkownik, jeden ekran |
| Observability | OpenTelemetry + Prometheus | Spany per etap, `/metrics`, `trace_id` w logach |

`embedding_provider` i `chat_provider` to **dwa niezależne przełączniki** — mieszanie ich
jest najczęstszym źródłem błędów w tym repo. Prefiksy E5 (`query:` / `passage:`) są jawną
konfiguracją, nie są zgadywane z nazwy modelu.

## Jakość

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/tsl_rag
uv run pytest -m unit            # bez zależności zewnętrznych
uv run pytest -m integration     # wymaga Postgresa z korpusem
uv run pytest -m e2e             # pełna aplikacja, generacja zaatrapowana
```

To samo uruchamia CI przy każdym pushu.

## Licencja

MIT. Korpus w `data/raw/` to publicznie dostępne akty prawne.
