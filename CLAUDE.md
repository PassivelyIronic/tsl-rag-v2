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

**To jest reimplementacja (v2), nie fork.** Z poprzedniej wersji (`../tsl-rag`) przejęte są
świadomie wybrane decyzje projektowe — hybrydowy retrieval, chunkowanie po granicach artykułów,
schemat pgvector, korpus PDF-ów — ale założenia, stack providerów i kryteria jakości są nowe.
Nie traktuj kodu odziedziczonego jako ustalonego: jeśli coś w nim jest sprzeczne z `PLAN.md`,
to plan wygrywa.

Cel: **odcięcie systemu od lokalnego GPU** i doprowadzenie go do stanu, w którym nietechniczna
osoba na słabym sprzęcie korzysta z niego w przeglądarce, za darmo, o dowolnej porze.

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
  → BM25 (rank-bm25, in-memory)─┴→ ważony RRF → cross-encoder rerank (CPU) → top-N
  → budowa promptu z kontekstem (limit w ZNAKACH: max_context_chars)
  → generacja                  (chat_provider)
  → odpowiedź + cytowania [doc_id | Art. X]
```

Dwie rzeczy nieoczywiste w tym przepływie:

- **Wagi RRF** (`bm25_weight` / `dense_weight`) są odczytywane z Settings i muszą sumować się
  do 1.0. Przy 0.5/0.5 ranking jest identyczny z nieważonym RRF — to jest baseline.
- **Tokenizer BM25 składa polskie diakrytyki** do ASCII po obu stronach (korpus i zapytanie),
  bo użytkownik często pisze bez ogonków. Nie „upraszczaj" go z powrotem do `[a-z0-9]+` —
  ten wzorzec rozrywał polskie słowa i jest pokryty testem regresyjnym.

Ingest: `PDF → legal_pdf_parser → legal_chunker (bufory per artykuł) → embedding → pgvector`

**Stan korpusu (zweryfikowany 2026-07-26):** 444 chunki z 13 dokumentów, zero powtórzonych
tekstów. `data/raw/` zawiera 14 PDF-ów, ale `TARIFF_EMPLOYER_2022.pdf` jest **bajtowo identyczny**
z `TARIFF_COMPANY_2022.pdf` (to samo md5) — jest więc pomijany celowo i opisany
w `_KNOWN_DUPLICATES` w `ingestion/cli.py`. Nie dodawaj mu wpisu w `DOCUMENT_REGISTRY`:
wstrzykuje 15 chunków identycznych z `tariff_company_2022`, które konkurują w retrievalu
i zajmują dwa miejsca w kontekście zamiast jednego.

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
docker compose up -d                                       # Postgres + pgvector (host: 5433)
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/ # ingest (oczekiwane: 13/14, 1 duplikat)
uv run python -m tsl_rag.api.main                          # FastAPI na :8000
uv run streamlit run ui.py                                 # UI (wymaga działającego API)
uv run python -m evals.run_evals --output evals/results/run_XXX.json
uv run python -m evals.compare_models --models "<slug>" --resume
```

Bramka jakości przed commitem — dokładnie to, co uruchamia CI:

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src/tsl_rag ; uv run pytest -m unit
```

`make` nie działa domyślnie na Windows — cele z `Makefile` odpalaj jako komendy wprost
(`make help` wypisuje listę celów, jeśli masz make w WSL-u).

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

Kubernetes jest **osobnym projektem, odległym w czasie i jeszcze nierozpoczętym**. TSL-RAG
będzie w nim tylko **jednym z tenantów** — nie jest to projekt „TSL-RAG na k8s", a klaster,
w którym ten system jest jedną z uruchomionych aplikacji.

Praktyczny wniosek: **nie dodawaj tutaj manifestów, Helm chartów ani konfiguracji ArgoCD.**
Nie projektuj też niczego „pod multi-tenancy" — namespace'y, quoty i polityki sieciowe są
sprawą klastra, nie aplikacji. Jedyne, co ma wynikać z tego kierunku, to żeby aplikacja była
**dobrze zachowującym się tenantem**: konfiguracja ze zmiennych środowiskowych, stan wyłącznie
w bazie, uczciwe probe'y, logi na stdout. To i tak jest dobra praktyka niezależnie od k8s.

Poniższe ograniczenia to jedyne, które już teraz wpływają na kod:

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
| ~~`RuntimeError: Event loop is closed` przy ingest~~ | ~~`asyncio.run()` per plik~~ | **Naprawione** — jeden `asyncio.run()` na całą pętlę (`_ingest_all_async`) |
| `UnicodeEncodeError: 'charmap' codec` przy ingest/evalu | Konsola na polskim Windowsie to cp1250, nie zna `→` ani `✓`. Proces przerywa się w połowie i wygląda to na błąd pipeline'u | Punkty wejścia CLI wołają `ensure_utf8_output()` z `core/console.py`. Dodając nowy skrypt CLI, zawołaj to samo |
| Dwa dokumenty o identycznej treści w wynikach retrievalu | PDF pobrany dwa razy pod inną nazwą (patrz `_KNOWN_DUPLICATES`) | Sprawdź `md5sum data/raw/*.pdf` przed dodaniem wpisu do rejestru. Duplikat nie jest nowym dokumentem |
| `404 — model unavailable for free` | Model wypadł z darmowej puli OpenRoutera | Sprawdź aktualny slug, nie retry'uj |
| `400 — not a valid model ID` | Zły slug (brak `-it` itp.) | Skopiuj slug z zakładki API modelu |
| `429 — Provider returned error` | Przeciążenie upstream providera, **nie** limit konta | Fallback na innego providera; retry rzadko pomaga |
| `Extra inputs are not permitted` w `Settings` | `.env` ma zmienne, których `settings.py` nie deklaruje | Zsynchronizuj `settings.py` z `.env` |
| Pusty retrieval po skopiowaniu repo | Docker Compose utworzył nowy wolumen (nazwa projektu = nazwa katalogu) | `ingest-all` od nowa albo `COMPOSE_PROJECT_NAME` |
| Port 5432 zajęty | Kontener oryginalnego TSL_RAG nadal działa | Ten fork używa **5433** na hoście |

---

## 9. Definition of done dla zmiany

Zanim uznasz zadanie za skończone:

- [ ] Kod przechodzi `ruff check`, `ruff format --check`, `mypy src/tsl_rag` i `pytest -m unit`
      (to samo, co `.github/workflows/ci.yml`)
- [ ] Zmiana dotykająca retrievalu lub generacji ma przebieg `run_evals` przed/po,
      a plik wyniku wylądował w `evals/results/` (jest wersjonowany, nie ignorowany)
- [ ] Naprawiony błąd ma test regresyjny, jeśli da się go pokryć jednostkowo
- [ ] Nowa zmienna konfiguracji jest w `settings.py` **i** `env.example`
- [ ] Nowy provider ma jawnie rozstrzygnięte, czy dotyczy embeddingu czy chatu
- [ ] Technologia spoza stacku (§6.5) ma uzasadnienie dopisane do tabeli §6.3
- [ ] Żadna liczba w dokumentacji nie pochodzi z pamięci — tylko z uruchomienia
- [ ] Błąd widoczny dla użytkownika końcowego jest po polsku i mówi, co zrobić
