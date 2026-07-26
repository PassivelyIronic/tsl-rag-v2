# TSL-RAG v2 — asystent prawa transportowego UE i PL

Hybrydowy system RAG nad przepisami o czasie pracy kierowców: rozporządzenia UE, dyrektywy,
umowa AETR, polska ustawa i taryfikatory kar. Odpowiada po polsku i **cytuje konkretny akt
oraz artykuł** — odpowiedź bez cytowania traktowana jest jako porażka, nie jako brak ozdobnika.

> **Status: w budowie.** To reimplementacja (v2) wcześniejszego projektu, prowadzona pod jeden
> konkretny cel: żeby system działał bez lokalnego GPU, za darmo i bez nadzoru autora.
> Ten cel **nie jest jeszcze osiągnięty** — embeddingi nadal wymagają lokalnej Ollamy.
> Plan dojścia i stan każdej fazy: [`PLAN.md`](PLAN.md).
> Wszystkie liczby w tym dokumencie pochodzą z faktycznych uruchomień; nic nie jest szacowane.

## Problem

Odpowiedź na pytanie „ile wynosi kara za przekroczenie dziennego czasu jazdy" wymaga
sięgnięcia do właściwego z **czterech** taryfikatorów i powiązania go z rozporządzeniem,
które ten limit ustala. Płaskie wyszukiwanie wektorowe myli akty o zbliżonej tematyce —
a w tej dziedzinie pewna siebie odpowiedź z błędnym cytowaniem jest gorsza niż odmowa,
bo nikt jej nie zweryfikuje.

Stąd trzy decyzje, które definiują ten projekt:

1. **Hybrydowy retrieval** — wyszukiwanie leksykalne (BM25) obok semantycznego, bo numery
   artykułów i kwoty kar to dopasowania dosłowne, w których embeddingi są słabe.
2. **Wymuszone cytowania** — prompt i parser odpowiedzi wymagają `[doc_id | Art. X]`.
3. **Mierzalna jakość** — harness ewaluacji rozdziela porażkę retrievalu od porażki generacji.
   Bez tego rozróżnienia optymalizacja jest zgadywaniem.

## Architektura

```text
pytanie
  ├→ embedding zapytania ──→ dense search (pgvector, cosine)  ─┐
  └→ tokenizacja PL ───────→ BM25 (rank-bm25, in-memory)      ─┴→ ważony RRF
                                                                    │
                                                      cross-encoder rerank (CPU)
                                                                    │
                                                        budowa kontekstu + prompt
                                                                    │
                                                     generacja (chat_provider)
                                                                    │
                                                  odpowiedź + [doc_id | Art. X]

ingest:  PDF → LegalPDFParser → LegalChunker (bufory per artykuł) → embedding → pgvector
```

**Embedding i generacja to dwa niezależne przełączniki** (`EMBEDDING_PROVIDER`, `CHAT_PROVIDER`).
Rozdzielone celowo: embedding liczony jest przy każdym zapytaniu i musi być najstabilniejszym
elementem układu, generacja może iść przez darmowy model w chmurze.

## Stack

| Warstwa | Technologia | Dlaczego ta |
|---|---|---|
| Embedding | `nomic-embed-text` (768d) przez Ollamę | Obecny stan; **to jest blocker** — patrz PLAN.md Faza 2 |
| Baza wektorowa | PostgreSQL 16 + pgvector, HNSW | Filtrowanie po metadanych w SQL obok wyszukiwania wektorowego |
| Wyszukiwanie leksykalne | `rank-bm25` in-memory | 444 chunki mieszczą się w pamięci; brak osobnego serwisu |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder ~90 MB, działa na CPU |
| Generacja | OpenRouter (`nvidia/nemotron-nano-9b-v2:free`) | Jedyny darmowy model zweryfikowany empirycznie w tym repo |
| API | FastAPI | Osobny serwis — pod probe'y i skalowanie |
| UI | Streamlit | Jeden ekran, jeden użytkownik. React byłby kolejnym serwisem do utrzymania |
| Eval | własny harness + Gemini jako sędzia | Sędzia musi być modelem innym niż oceniany |

Uzasadnienie każdego odstępstwa od stacku kanonicznego: [`CLAUDE.md`](CLAUDE.md) §6.
Analiza providerów i zaobserwowanych trybów awarii: [`docs/PROVIDERS.md`](docs/PROVIDERS.md).

## Korpus

Stan z bazy, `SELECT * FROM corpus_stats` (2026-07-26): **444 chunki, 13 dokumentów**,
z czego 39 chunków tabelarycznych.

| Dokument | Typ | Chunki | w tym tabele |
|---|---|---:|---:|
| `eu_165_2014` — tachografy | rozporządzenie UE | 85 | 0 |
| `eu_1071_2009` — zawód przewoźnika | rozporządzenie UE | 62 | 0 |
| `directive_2020_1057` — delegowanie kierowców | dyrektywa | 51 | 0 |
| `ec_561_2006` — czas prowadzenia pojazdu | rozporządzenie UE | 46 | 0 |
| `aetr` — umowa AETR | umowa międzynarodowa | 41 | 0 |
| `eu_1072_2009` — kabotaż | rozporządzenie UE | 37 | 3 |
| `eu_2020_1054` — Pakiet Mobilności | rozporządzenie UE | 36 | 0 |
| `eu_2016_403` — klasyfikacja naruszeń | rozporządzenie UE | 24 | 16 |
| `directive_2002_15` — czas pracy | dyrektywa | 23 | 0 |
| `pl_driver_hours_act` — ustawa PL | prawo krajowe | 16 | 0 |
| `tariff_company_2022` — taryfikator przedsiębiorcy | taryfikator | 15 | 12 |
| `tariff_driver_2022` — taryfikator kierowcy | taryfikator | 5 | 5 |
| `tariff_manager_2022` — taryfikator zarządzającego | taryfikator | 3 | 3 |

`data/raw/` zawiera 14 plików. Czternasty, `TARIFF_EMPLOYER_2022.pdf`, jest bajtowo identyczny
z `TARIFF_COMPANY_2022.pdf` (to samo md5) i dlatego jest pomijany przy ingeście — to ten sam
dokument pobrany dwa razy, nie brakujący akt prawny.

## Ewaluacja

Golden dataset: **15 pytań w 6 kategoriach**. Harness mierzy nie tylko trafność odpowiedzi,
ale też **na jakim etapie system się wywrócił**:

| Metryka | Co mówi |
|---|---|
| `answer_score` | Czy odpowiedź zawiera oczekiwane fakty |
| `citation_hit_rate` | Recall cytowań — ile oczekiwanych dokumentów zacytowano |
| `citation_precision` | Precyzja cytowań — ile z zacytowanych było oczekiwanych |
| `retrieval_recall` | Czy retriever w ogóle podał właściwy dokument do kontekstu |
| `failure_stage` | `retrieval` (nie znalazł) vs `generation` (znalazł, model zignorował) |
| `refusal_precision` | Czy pytania poza zakresem kończą się odmową, nie halucynacją |

Rozdzielenie `retrieval_recall` od `citation_hit_rate` jest tu najważniejsze: samo zero
w cytowaniach nie mówi, czy winny jest retriever, czy model.

### Wyniki (3 przebiegi, 2026-07-26)

`nvidia/nemotron-nano-9b-v2:free` przez OpenRouter, embeddingi `nomic-embed-text`,
`top_k=20`, `rerank_top_n=5`, wagi RRF 0.5/0.5, ocena keyword-match.
Pliki: `evals/results/run_010`–`run_012`.

| Metryka | `run_010` | `run_011` | `run_012` | |
|---|---|---|---|---|
| `answer_score` | 0.633 | 0.600 | 0.633 | |
| `citation_hit_rate` | 0.733 | 0.667 | 0.800 | |
| `citation_precision` | 0.800 | 0.800 | 0.733 | |
| `retrieval_recall` | 0.867 | 0.867 | 0.867 | stabilny |
| `refusal_precision` | 1.000 | 1.000 | 1.000 | stabilny |
| latencja średnia | 19.1 s | 17.5 s | 15.4 s | |

**`run_011` i `run_012` to ten sam kod** — między nimi nie ma żadnej zmiany. Rozrzut
`citation_hit_rate` wynosi więc 0.133 przy niezmienionym systemie, co jest **większe niż efekt
typowej zmiany w retrievalu**. Wniosek: przy 15 pytaniach metryki zależne od generacji nie
nadają się na bramkę CI. Metryki retrievalu są za to identyczne w każdym przebiegu — bo nie
przechodzą przez LLM — i to na nich stanie bramka promocji.

Publikowanie jednego z tych przebiegów jako „wyniku systemu" byłoby wprowadzaniem w błąd,
dlatego podane są wszystkie trzy.

Uruchomienie:

```bash
# ocena keyword-match, bez żadnego klucza API
uv run python -m evals.run_evals --output evals/results/run_XXX.json

# ocena semantyczna (wymaga GEMINI_API_KEY)
uv run python -m evals.run_evals --use-judge --output evals/results/run_XXX_judge.json

# porównanie modeli generacji na tym samym retrievalu
uv run python -m evals.compare_models --models "<slug>" --resume
```

Wyniki są wersjonowane w `evals/results/` razem ze snapshotem konfiguracji, która je
wyprodukowała (model, wagi RRF, `top_k`, reranker). Liczba bez warunków pomiaru jest
bezużyteczna po tygodniu.

**Ograniczenia, świadomie nieukrywane:**

- 15 pytań to za mało na bramkowanie CI — **zmierzone, nie przypuszczane** (rozrzut 0.133
  między identycznymi przebiegami). Kategorie `cross_document`, `penalty` i `scope` mają
  po **jednym** pytaniu, więc ich wynik to 0.0 albo 1.0. Rozszerzenie datasetu to Faza 1 planu.
- Dwie z trzech porażek to porażki **retrievalu**, powtarzalne w każdym przebiegu:
  `penalty` (taryfikator kierowcy nie wchodzi do kontekstu — ma 5 chunków przeciw 15
  taryfikatora przedsiębiorcy) i `cross_document` (nie wchodzą oba oczekiwane akty).
  `scope` to porażka generacji: właściwy dokument jest w kontekście, model cytuje inny.
- Embeddingi wymagają lokalnej Ollamy, czyli cel „działa bez GPU autora" nie jest spełniony.
- Brak łańcucha fallbacku providerów — jedno `429` kończy zapytanie komunikatem o błędzie.
- Brak observability. Wiadomo, że generacja dominuje latencję, ale nie ma rozbicia per etap.

## Uruchomienie

Wymagania: Docker, Ollama, Python 3.11, [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. modele lokalne (embeddingi + model czatu do pracy offline)
ollama pull nomic-embed-text
ollama pull mistral:7b-instruct-q4_K_M

# 2. zależności
uv sync

# 3. konfiguracja — domyślne wartości działają bez żadnego klucza API
cp env.example .env

# 4. baza (Postgres + pgvector na porcie 5433, nie 5432)
docker compose up -d

# 5. ingest korpusu — oczekiwane: 13/14 plików, 1 pominięty duplikat
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/

# weryfikacja
docker exec tsl_rag_postgres_reimplemented \
  psql -U postgres -d tsl_rag -c "SELECT * FROM corpus_stats;"
```

Uruchomienie aplikacji — dwa terminale:

```bash
uv run python -m tsl_rag.api.main   # API na http://localhost:8000/docs
uv run streamlit run ui.py          # UI  na http://localhost:8501
```

Żeby generować odpowiedzi przez darmowy model w chmurze zamiast lokalnie, ustaw w `.env`
`CHAT_PROVIDER=openrouter` i `OPENROUTER_API_KEY` (klucz nie wymaga karty).
`EMBEDDING_PROVIDER` zostaje wtedy na `ollama` — to osobny przełącznik.

### Stan systemu

| Endpoint | Znaczenie |
|---|---|
| `GET /health` | Liveness — proces odpowiada. Nie odpytuje zależności |
| `GET /ready` | Readiness — Postgres, retriever i provider embeddingów. `503`, gdy któryś nie działa |
| `POST /query` | Główny endpoint RAG |
| `GET /query/documents` | Lista obsługiwanych dokumentów |

## Struktura

```
src/tsl_rag/
├── core/
│   ├── settings.py       # Pydantic Settings; EMBEDDING_PROVIDER i CHAT_PROVIDER osobno
│   ├── models.py         # Chunk, DocumentMetadata, RetrievalRequest, QueryResponse
│   ├── llm_client.py     # get_llm_client() = embeddingi, get_chat_client() = generacja
│   └── console.py        # UTF-8 na wyjściu (cp1250 na Windows wywalał ingest)
├── ingestion/
│   ├── parsers/          # pdfplumber (tabele) + pymupdf (tekst), detekcja hierarchii
│   ├── chunkers/         # bufory per artykuł, tabele nigdy nie dzielone
│   ├── embedders/        # batch embedding → upsert do pgvector
│   └── cli.py            # Typer: ingest / ingest-all
├── retrieval/
│   ├── retriever.py      # dense + BM25 + ważony RRF, tokenizer składający diakrytyki
│   └── reranker.py       # cross-encoder, ładowany raz przy starcie API
├── generation/generator.py   # prompt systemowy, wymuszenie i parsowanie cytowań
└── api/
    ├── app.py            # fabryka + lifespan (jeden retriever na proces)
    └── routers/          # query.py, health.py
evals/
├── golden_dataset/       # 15 pytań × 6 kategorii
├── judge.py              # Gemini jako LLM-as-a-judge
├── run_evals.py          # harness + rozdzielenie etapu porażki
├── compare_models.py     # porównanie modeli generacji na stałym retrievalu
└── results/              # wyniki przebiegów, wersjonowane
ui.py                     # Streamlit
docker/init.sql           # schemat, indeks HNSW, widok corpus_stats
```

## Decyzje projektowe

**Własny pipeline zamiast frameworka orkiestracji.** LangChain i LlamaIndex ukrywają
mechanikę retrievalu za abstrakcją. Tutaj hybrydowe wyszukiwanie, fuzja RRF i reranking są
napisane wprost — każdy element jest osobno testowalny i osobno mierzalny, co jest warunkiem
sensownej ewaluacji.

**Chunkowanie po granicach artykułów.** Przepis rozjechany między dwa chunki traci sens
prawny. `LegalChunker` traktuje granicę artykułu jako twardą i nigdy nie dzieli tabel —
taryfikator kar podzielony w połowie to gotowa halucynacja o wysokości grzywny.

**Tokenizer składający polskie diakrytyki.** BM25 na wzorcu `[a-z0-9]+` rozrywał polskie
słowa („prędkość" → `pr`, `dko`). Teraz tokeny są sprowadzane do ASCII po obu stronach, więc
zapytanie napisane bez ogonków też trafia — bo tak realnie pisze użytkownik.

**Gemini jako sędzia, nie model lokalny.** Ocenianie modelu przez model tej samej klasy
wprowadza obciążenie samooceny. Sędzia zapisuje też uzasadnienie oceny, więc porażki dają
się audytować.

**PostgreSQL, choć przy 444 chunkach `numpy` byłby szybszy.** Świadoma decyzja: repo ma być
bazą pod osobny projekt Kubernetes, w którym baza w `StatefulSet` jest wymaganym elementem.
Rozstrzygnięcie tego konfliktu opisane jest w `PLAN.md`.

## Testy i jakość

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/tsl_rag
uv run pytest -m unit
```

Ten sam zestaw uruchamia CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) na push
i pull request. Testy integracyjne i bramka na metrykach evalu jeszcze nie istnieją —
wymagają zaindeksowanej bazy i klucza providera, i są zaplanowane, nie udawane.

## Dokumentacja

| Plik | Zawartość |
|---|---|
| [`PLAN.md`](PLAN.md) | Stan faktyczny, fazy, bramki, rozstrzygnięcia konfliktów celów |
| [`CLAUDE.md`](CLAUDE.md) | Zasady pracy w repo, pułapki środowiskowe, definition of done |
| [`docs/PROVIDERS.md`](docs/PROVIDERS.md) | Analiza providerów inference i ich trybów awarii |

## Licencja

MIT
