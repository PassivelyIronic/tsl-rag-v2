# CLAUDE.md — TSL_RAG_reimplemented

Instrukcje dla asystenta pracującego w tym repo. Czytaj razem z `PLAN.md`
(co robimy i w jakiej kolejności) oraz `docs/PROVIDERS.md` (dlaczego takie, a nie inne API).

> **Handoff:** wykonanie prowadzi Claude Code w CLI. Kolejność startu:
> `PLAN.md` §Stan obecny → `PLAN.md` §Faza 0 → ten plik §5 (twarde zasady) i §6 (stack).
> Nie zaczynaj kodowania przed przeczytaniem §5 — zawiera zasady, których łamanie
> zniszczy wartość eval harnessu.

---

## 1. Czym jest ten projekt

Hybrydowy RAG nad prawem transportowym UE i PL (rozliczanie czasu pracy kierowców).
Fork projektu `TSL_RAG`, którego jedynym celem jest **odcięcie systemu od lokalnego GPU**
i doprowadzenie go do stanu, w którym nietechniczna osoba na słabym sprzęcie korzysta z niego
w przeglądarce, za darmo, o dowolnej porze.

Drugi, równoległy cel: repo ma być **bazą aplikacyjną pod portfolio Kubernetes** (deployment
na k3s + eval-gated CI/CD). To wpływa na decyzje architektoniczne — patrz §7.

**Użytkownik docelowy jest nietechniczny.** Konsekwencje projektowe:
- Błąd musi być komunikatem po polsku, nie stacktrace'em.
- Niedostępność providera nie może kończyć się pustym ekranem — ma być fallback albo jasny komunikat.
- Halucynacja z pewną siebie tonacją jest **gorsza** niż odmowa. Model, który cytuje zły akt
  prawny, wyrządza więcej szkody niż model mówiący "nie wiem" — bo nikt tego nie zweryfikuje.

---

## 2. Architektura — przepływ zapytania

```
pytanie
  → embedding zapytania        (embedding_provider)
  → dense retrieval (pgvector) ─┐
  → BM25 (rank-bm25, in-memory)─┴→ RRF fusion → cross-encoder rerank (CPU) → top-N
  → budowa promptu z kontekstem
  → generacja                  (chat_provider)
  → odpowiedź + cytowania [doc_id | Art. X]
```

Ingest: `PDF → legal_pdf_parser → legal_chunker (bufory per artykuł) → embedding → pgvector`

**Stan korpusu:** 444 chunki z 13 PDF-ów. `TARIFF_EMPLOYER_2022.pdf` jest pomijany —
brak wpisu w `DOCUMENT_REGISTRY`.

---

## 3. Krytyczna zasada: `embedding_provider` ≠ `chat_provider`

To są **dwa niezależne przełączniki** i mieszanie ich jest najczęstszym źródłem błędów w tym repo.

```python
embedding_provider: Literal["ollama", "openai"]              # get_llm_client()
chat_provider:      Literal["ollama", "openai", "openrouter"] # get_chat_client()
```

- `get_llm_client()` → **wyłącznie embeddingi** (ingest, retrieval, health-check)
- `get_chat_client()` → **wyłącznie generacja** (`RAGGenerator`)

Przy dodawaniu nowego providera: rozstrzygnij najpierw, którego z dwóch dotyczy.
Nie wprowadzaj z powrotem wspólnej flagi `llm_provider` — została celowo rozbita.

**Pułapka zaobserwowana w praktyce:** ustawienie `OPENROUTER_CHAT_MODEL` przy
`CHAT_PROVIDER=ollama` nie robi nic. Model jest odczytywany dopiero, gdy provider
jest przełączony. Skrypt `compare_models.py` nadpisuje `CHAT_PROVIDER` w locie —
`.env` go wtedy nie ogranicza.

---

## 4. Komendy

Środowisko: `uv` + `.venv` w katalogu projektu. Interpreter zarządzany przez uv (3.11).
**Zawsze `uv run ...`**, nigdy gołe `python` — na Windows aktywna conda `(base)` przechwyci wywołanie.

```powershell
uv sync                                                    # setup środowiska
docker compose up -d                                       # Postgres + pgvector
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/ # ingest
uv run python -m tsl_rag.api.main                          # FastAPI
uv run streamlit run src/tsl_rag/ui/app.py                 # UI
uv run python -m evals.run_evals                           # eval baseline
uv run python -m evals.compare_models --models "<slug>" --resume
```

`make` nie działa domyślnie na Windows — cele z `Makefile` odpalaj jako komendy wprost.

---

## 5. Twarde zasady (nie łam bez wyraźnej zgody użytkownika)

1. **Nie obniżaj progów ewaluacji, żeby test przeszedł.** Jeśli metryka spada poniżej progu,
   to jest wynik do zaraportowania, nie próg do zmiany. Zmiana progu wymaga osobnego,
   jawnego commita z uzasadnieniem — nigdy nie jest częścią commita, który tę metrykę psuje.

2. **Zero niezweryfikowanych twierdzeń w README i dokumentacji.** Każda liczba, benchmark
   i nazwa modelu ma pochodzić z faktycznego uruchomienia albo być oznaczona jako niezweryfikowana.
   Ten projekt ma historię błędów tego typu (zły domyślny model, zepsuta ścieżka `make ui`,
   błędne wywołanie `main.py` w README) — nie powtarzaj ich.

3. **Slug modelu weryfikuj przed wpisaniem do kodu.** Nazwa wyświetlana w katalogu ≠ slug API
   (`gemma-4-31b` vs `gemma-4-31b-it`). Slugi rotują — modele znikają z darmowych pul.
   Nie zgaduj z pamięci.

4. **`.env` nigdy nie trafia do repo.** Zmiany konfiguracji idą do `env.example` z pustą wartością.

5. **Nie usuwaj retrieval z pipeline'u, żeby "uprościć".** BM25 + dense + RRF + rerank to rdzeń
   projektu. Optymalizacje dotyczą warstwy providerów, nie retrievalu.

6. **Cytowania są funkcją krytyczną, nie ozdobą.** Odpowiedź bez `[doc_id | Art. X]` albo
   z cytowaniem niewłaściwego aktu prawnego traktuj jako porażkę, nawet jeśli treść brzmi sensownie.

---

## 6. Stack technologiczny

### 6.1 Stack kanoniczny (preferencje właściciela repo)

| Warstwa | Technologie |
|---|---|
| Backend | Python, FastAPI, Celery |
| Baza danych | PostgreSQL, Supabase |
| Frontend | React, Vite, shadcn/ui |
| Warstwa AI | OpenAI, Anthropic, AWS, Azure, Google Cloud |
| Infra / deploy | Docker, Railway, Hetzner (VPS), AWS, Azure, Google Cloud |

### 6.2 Użyte w tym projekcie — ze stacku

`Python 3.11` · `FastAPI` · `PostgreSQL` · `Docker`

### 6.3 Poza stackiem — dodane świadomie

| Technologia | W miejsce czego | Uzasadnienie |
|---|---|---|
| **Streamlit** | React / Vite / shadcn | Jeden nietechniczny użytkownik. React to osobny serwis do budowania, hostowania i utrzymania — bez zysku funkcjonalnego przy jednym ekranie z polem tekstowym |
| **uv** | pip / poetry | Już obecne w repo (`uv.lock`), deterministyczny lockfile, izolacja od condy na Windows |
| **pgvector** | — | Rozszerzenie PostgreSQL, nie odejście od stacku |
| **rank-bm25, sentence-transformers, cross-encoder** | — | Rdzeń retrievalu. Brak odpowiednika w stacku |
| **OpenRouter, Cloudflare Workers AI** | OpenAI / Anthropic / Azure / GCP | Jedyne opcje realnie $0 bez karty. OpenAI i Anthropic są płatne od pierwszego tokena, co jest sprzeczne z celem podstawowym |
| **k3s + Oracle Cloud Free Tier** | Railway / Hetzner / AWS / Azure / GCP | Railway i Hetzner kosztują. Oracle Free Tier daje 3 węzły always-on za $0, a k3s jest wprost wymagany przez projekt portfolio |
| **OpenTelemetry, Prometheus, Grafana, Jaeger** | — | Stack nie zawiera warstwy observability, a jest wymagana przez cel #2 |
| **ArgoCD, Argo Rollouts** | — | GitOps i canary. Brak odpowiednika w stacku |

### 6.4 Ze stacku, ale świadomie wykluczone tutaj

| Technologia | Dlaczego nie w tym projekcie | Gdzie ma sens |
|---|---|---|
| **Celery** | Ingest jest ręczny (14 PDF-ów, uruchamiany przez właściciela). Brak długotrwałych zadań w tle. Celery + Redis to dwa dodatkowe serwisy pełniące rolę atrapy | Projekt z realnie asynchronicznym workloadem — np. StreamScore |
| **Supabase** | Free tier usypia projekt po okresie bezczynności; nakłada się to na cold start hostingu, dając podwójne opóźnienie przy rzadkim użyciu. Przy 444 chunkach zarządzana baza nic nie wnosi ponad kontener Postgresa | Projekt, w którym auth i realtime Supabase są faktycznie używane |
| **React / Vite / shadcn** | Jak wyżej — jeden użytkownik, jeden ekran | Projekt z wieloma widokami lub użytkownikami |
| **OpenAI / Anthropic API** | Płatne od pierwszego tokena — sprzeczne z celem $0 **jako runtime**. Ale patrz `PLAN.md` Faza 1: jako *płatny model referencyjny w evalu* są wartościowe i tanie | Runtime, gdy budżet przestaje być zerowy |
| **AWS / Azure / GCP** | Brak trwale darmowego compute pod to zapotrzebowanie | Projekty, gdzie budżet lub kredyty istnieją |
| **Railway / Hetzner** | Płatne | jw. |

### 6.5 Reguła decyzyjna — kiedy wolno wyjść poza stack

Wyjście poza stack jest dozwolone, gdy spełniony jest **co najmniej jeden** warunek:

1. Technologia ze stacku nie ma wariantu $0, a cel podstawowy wymaga zera kosztów.
2. Technologia ze stacku dodałaby serwis, który nie pełni realnej funkcji (atrapa).
3. Stack nie zawiera odpowiednika wymaganej funkcji (retrieval, observability, GitOps).

Jeśli żaden warunek nie jest spełniony — **użyj tego, co w stacku**. Nie dodawaj technologii
spoza stacku dlatego, że jest nowa lub ciekawa. Każde wyjście poza stack dopisz do tabeli §6.3
z jednozdaniowym uzasadnieniem, żeby decyzja została udokumentowana, a nie odkryta później w kodzie.

---

## 7. Ograniczenia z portfolio Kubernetes

Repo ma zostać wdrożone na k3s (Oracle Cloud Free Tier, 3 węzły) w ramach osobnego projektu
portfolio. To zmienia część decyzji względem "najprostszego rozwiązania dla mamy":

- **PostgreSQL + pgvector zostaje** (mimo że przy 444 chunkach brute-force numpy byłby szybszy).
  Powód: StatefulSet z bazą jest wymaganym elementem projektu K8s. Prostota przegrywa tu
  z wartością portfolio — świadomie.
- **FastAPI zostaje jako osobny serwis** (nie in-process w Streamlicie) — potrzebny do HPA,
  probe'ów i canary deployments.
- **Eval harness musi być uruchamialny jako job wsadowy** (`exit code` ≠ 0 przy niespełnionym progu),
  bo staje się bramką promocji w ArgoCD.
- **Observability od początku w kodzie**, nie doklejana później: OpenTelemetry spans na etapach
  retrieve / rerank / generate, metryki Prometheus, logi strukturalne.

Nie upraszczaj tych elementów "bo dla jednego użytkownika to przerost" — one są tu z powodu
drugiego celu projektu.

---

## 8. Znane pułapki środowiskowe

| Objaw | Przyczyna | Reakcja |
|---|---|---|
| `RuntimeError: Event loop is closed` przy ingest | Windows ProactorEventLoop + `httpx.AsyncClient`, `asyncio.run()` per plik | Kosmetyczne. `failed: 0` = ingest OK. Naprawa: jeden `asyncio.run()` na całą pętlę |
| `404 — model unavailable for free` | Model wypadł z darmowej puli OpenRoutera | Sprawdź aktualny slug, nie retry'uj |
| `400 — not a valid model ID` | Zły slug (brak `-it` itp.) | Skopiuj slug z zakładki API modelu |
| `429 — Provider returned error` | Przeciążenie upstream providera, **nie** limit konta | Fallback na innego providera; retry rzadko pomaga |
| `Extra inputs are not permitted` w `Settings` | `.env` ma zmienne, których `settings.py` nie deklaruje | Zsynchronizuj `settings.py` z `.env` |
| Pusty retrieval po skopiowaniu repo | Docker Compose utworzył nowy wolumen (nazwa projektu = nazwa katalogu) | `ingest-all` od nowa albo `COMPOSE_PROJECT_NAME` |
| Port 5432 zajęty | Kontener oryginalnego TSL_RAG nadal działa | Ten fork używa **5433** na hoście |

---

## 9. Definition of done dla zmiany

Zanim uznasz zadanie za skończone:

- [ ] Kod przechodzi `ruff` i `mypy` (konfiguracja w `pyproject.toml`)
- [ ] Zmiana dotykająca retrievalu lub generacji ma przebieg `run_evals` przed/po
- [ ] Nowa zmienna konfiguracji jest w `settings.py` **i** `env.example`
- [ ] Nowy provider ma jawnie rozstrzygnięte, czy dotyczy embeddingu czy chatu
- [ ] Technologia spoza stacku (§6.5) ma uzasadnienie dopisane do tabeli §6.3
- [ ] Żadna liczba w dokumentacji nie pochodzi z pamięci — tylko z uruchomienia
- [ ] Błąd widoczny dla użytkownika końcowego jest po polsku i mówi, co zrobić
