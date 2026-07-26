# PLAN.md — TSL_RAG_reimplemented

## Cel

Doprowadzić fork do stanu, w którym:

1. **Cel podstawowy:** nietechniczna osoba na słabym sprzęcie korzysta z systemu w przeglądarce,
   za darmo, o dowolnej porze — bez lokalnego GPU i bez Twojej interwencji.
2. **Cel portfolio:** repo jest bazą aplikacyjną pod projekt Kubernetes (k3s + eval-gated CI/CD)
   i demonstruje observability, odporność na awarie providerów oraz mierzalną jakość RAG-a.

Te cele czasem są w konflikcie (§ Konflikt celów). Tam gdzie są — decyzja jest jawna, nie domyślna.

---

## Stan obecny (zweryfikowany, lipiec 2026)

**Działa:**
- Rozdzielone `embedding_provider` / `chat_provider`, integracja OpenRouter w `get_chat_client()`
- Ingest: 444 chunki z 13 PDF-ów, `stored: 444, failed: 0`
- Retrieval: BM25 (444 chunki) + dense + RRF + cross-encoder rerank na CPU
- `evals/compare_models.py` — porównanie modeli na golden dataset (15 pytań), retry, `--resume`

**Zmierzone — jedyny działający model generacji:**

`nvidia/nemotron-nano-9b-v2:free` (przez OpenRouter):

| Metryka | Wynik |
|---|---|
| avg_answer_score | 0.70 |
| avg_citation_hit_rate | 0.733 |
| refusal_precision | 1.00 |
| false_refusal_rate | 0.00 |
| avg_latency | ~22.6 s |

Rozbicie po kategoriach ujawnia trzy konkretne problemy:

| Kategoria | answer | citation | Diagnoza |
|---|---|---|---|
| numeric_fact (9) | 0.722 | 0.889 | Najmocniejsza strona |
| procedure (1) | 1.0 | 1.0 | OK |
| out_of_scope (2) | 1.0 | 1.0 | Odmowy działają poprawnie |
| **cross_document (1)** | **0.0** | **0.0** | Pusta odpowiedź mimo 37 s latencji — realna porażka syntezy 2 aktów |
| **penalty (1)** | **0.0** | **0.0** | Zacytował `tariff_company_2022` zamiast `tariff_driver_2022` — mylenie taryf firma/kierowca |
| **scope (1)** | **1.0** | **0.0** | Zacytował `eu_1072_2009` zamiast `aetr` — mylenie aktów o zbliżonej tematyce |

**Nie działa / zablokowane:**
- Embeddingi nadal wymagają lokalnej Ollamy → **główny blocker celu podstawowego**
- Martwe slugi w wynikach: `deepseek/deepseek-chat-v3.1:free`, `qwen/qwen3-235b-a22b:free` (404)
- `TARIFF_EMPLOYER_2022.pdf` pomijany przy ingest — brak wpisu w `DOCUMENT_REGISTRY`
- Brak jakiejkolwiek observability
- Brak fallbacku providerów — pojedynczy 429 kończy zapytanie błędem

---

## Konflikt celów — rozstrzygnięcia

| Kwestia | Optimum dla mamy | Optimum dla portfolio | Decyzja |
|---|---|---|---|
| Vector store | numpy brute-force (444 chunki, <10 ms) | Postgres StatefulSet | **Postgres** — portfolio wygrywa, koszt złożoności akceptowalny |
| API | in-process w Streamlicie | osobny FastAPI (HPA, probes, canary) | **Osobny FastAPI** |
| Frontend | Streamlit | React/Vite/shadcn | **Streamlit** — React nic nie wnosi dla 1 użytkownika, a jest kolejnym serwisem do utrzymania |
| Kolejka zadań | brak (ingest ręczny) | Celery + Redis | **Brak** — ingest odpalasz Ty, ręcznie, przy wymianie PDF-ów. Celery byłby atrapą |
| Hosting | HF Spaces (najprostszy) | k3s Oracle Free Tier | **k3s** jeśli Faza 6 dojdzie do skutku; HF Spaces jako ścieżka awaryjna |

### Mapowanie na stack technologiczny

Pełne rozpisanie w `CLAUDE.md` §6 wraz z regułą decyzyjną, kiedy wolno wyjść poza stack.
Skrót:

- **Ze stacku, użyte:** Python, FastAPI, PostgreSQL, Docker
- **Ze stacku, wykluczone tutaj:** Celery (ingest ręczny — byłby atrapą), Supabase (free tier
  usypia, podwójny cold start), React/Vite/shadcn (jeden użytkownik, jeden ekran),
  Railway/Hetzner/AWS/Azure/GCP (płatne), OpenAI/Anthropic **jako runtime** (płatne od
  pierwszego tokena) — ale patrz Faza 1, gdzie są przydatne jako model referencyjny
- **Poza stackiem, dodane świadomie:** Streamlit, uv, pgvector, rank-bm25,
  sentence-transformers, OpenRouter, Cloudflare Workers AI, k3s + Oracle Free Tier,
  OpenTelemetry/Prometheus/Grafana/Jaeger, ArgoCD + Argo Rollouts

Wykluczenia nie wynikają z tego, że te technologie są złe — wynikają z tego, że **tutaj**
byłyby dekoracją bez funkcji, a każdy dodatkowy serwis to kolejna rzecz, która może paść,
gdy Ciebie nie ma przy komputerze. Reguła z `CLAUDE.md` §6.5 pozwala wyjść poza stack tylko
wtedy, gdy stack nie ma wariantu $0, dodałby atrapę albo nie zawiera danej funkcji w ogóle.

---

## Faza 0 — Higiena (0.5 dnia)

Odblokowanie dalszej pracy, zero nowej funkcjonalności.

- [ ] Dodać wpis `tariff_employer_2022` do `DOCUMENT_REGISTRY`, ponowić ingest tego pliku
- [ ] Usunąć martwe wpisy (`deepseek…`, `qwen3-235b…`) z `evals/results/model_comparison.json` — to szum, nie dane
- [ ] `DEFAULT_CANDIDATES` w `compare_models.py` → tylko zweryfikowane slugi
- [ ] `ingest_all`: jeden `asyncio.run()` na całą pętlę zamiast per plik (usuwa hałas event loopa)
- [ ] `README.md`: usunąć/oznaczyć twierdzenia niezweryfikowane po refaktorze providerów

**Gate:** `ingest-all` przechodzi na 14/14 plików bez tracebacków.

---

## Faza 1 — Metryki retrievalu (1 dzień)

**Ta faza idzie przed jakąkolwiek zmianą modelu. Powód:** obecny harness mierzy wyłącznie
wynik end-to-end. Gdy `citation_hit = 0` — jak przy `penalty` i `scope` — **nie da się
odróżnić, czy retriever nie podał właściwego dokumentu, czy podał, a model go zignorował.**
Bez tego rozróżnienia każda optymalizacja jest zgadywaniem, a A/B testy embeddingów z Fazy 2
byłyby zanieczyszczone jakością LLM-a.

- [ ] `evals/run_retrieval_evals.py` — ewaluacja **samego** retrievalu, bez wywołania LLM:
  - `recall@k` dla `k ∈ {5, 10, 20}` względem `expected_docs`
  - `MRR` — na której pozycji pojawia się właściwy dokument
  - metryki osobno **przed** i **po** rerankingu (czy reranker pomaga, czy szkodzi)
- [ ] Wzbogacić rekord wyniku w `compare_models.py` o `retrieved_docs`, żeby dało się
      rozdzielić „retriever nie podał” od „model zignorował kontekst”
- [ ] Rozszerzyć golden dataset: **15 pytań to za mało na bramkowanie**. Kategorie `cross_document`,
      `penalty`, `scope` mają po **1** pytaniu — ich wynik to 0.0 albo 1.0, bez wartości pośrednich,
      czyli statystycznie bezużyteczne. Cel: min. 5 pytań na kategorię (~40 pytań łącznie)
- [ ] Wyjaśnić przypadek `penalty` i `scope`: retrieval czy generacja?
- [ ] **Model referencyjny (płatny, jednorazowo).** Uruchomić golden dataset raz na `gpt-4o`
      albo modelu Anthropic — czyli na tym, co jest w Twoim stacku kanonicznym. Cel: ustalić
      **sufit jakości** przy obecnym retrievalu. Bez tego punktu odniesienia nie wiadomo, czy
      `answer_score = 0.70` znaczy „darmowy model jest słaby”, czy „retrieval podaje zły kontekst
      i lepszy model też by nie pomógł”. Koszt: rząd kilku centów za przebieg 15-40 pytań.
      To nie jest odejście od celu $0 — runtime zostaje darmowy, płatny jest wyłącznie pomiar

**Gate:** dla każdego pytania z golden dataset wiadomo, czy porażka leży w retrievalu, czy w generacji.
Znany sufit jakości z modelu referencyjnego.

---

## Faza 2 — Odcięcie od lokalnego GPU (1-2 dni)

Blocker celu podstawowego. Embedding jest liczony przy **każdym zapytaniu**, nie tylko przy
ingest — to najbardziej wrażliwy na awarię punkt pipeline'u.

- [ ] Rozszerzyć `embedding_provider` o `"local"` (`sentence-transformers`, in-process, CPU)
- [ ] A/B kandydatów **metrykami z Fazy 1**, nie „na oko”:

| Model | Wymiary | Re-ingest? |
|---|---|---|
| `nomic-embed-text` (obecny, Ollama) | 768 | baseline |
| `intfloat/multilingual-e5-base` | 768 | nie — zgodne wymiary |
| `BAAI/bge-m3` | 1024 | **tak** + migracja kolumny `vector(n)` |

- [ ] Migracja `docker/init.sql` + skrypt migracyjny, jeśli wygra `bge-m3`
- [ ] Zmierzyć czas embeddingu jednego zapytania na CPU (cel: <1 s)
- [ ] Opcjonalnie: Cloudflare BGE jako `embedding_provider="cloudflare"` — zapas przy ograniczonej pamięci

**Gate:** pełne zapytanie end-to-end działa przy **zatrzymanej Ollamie**, `recall@5`
nie gorszy niż baseline o więcej niż 5 pp.

---

## Faza 3 — Model generacji + odporność providerów (1-2 dni)

Uzasadnienie fallbacku jest empiryczne, nie teoretyczne: w jednej sesji testowej wystąpiły
**trzy różne klasy awarii** OpenRoutera (404 wycofany model, 400 zły slug, 429 przeciążenie
upstream — 3/3 próby z backoffem). System, z którego ktoś korzysta bez Twojego nadzoru,
nie może zależeć od jednego darmowego endpointu.

- [ ] Integracja Cloudflare Workers AI jako `chat_provider="cloudflare"` (10k neuronów/dobę,
      reset 00:00 UTC, brak rotacji modeli — hosting na własnym GPU Cloudflare)
- [ ] Benchmark przez `compare_models.py`: kandydaci Cloudflare vs `nemotron-nano-9b-v2:free`
- [ ] **Łańcuch fallbacku** w `get_chat_client()` / `RAGGenerator`:
  - uporządkowana lista `(provider, model)`
  - przejście dalej przy `404` / `400` / `429` — bez retry na `404` i `400` (są deterministyczne)
  - circuit breaker: po N porażkach provider wypada z rotacji na T minut
  - log strukturalny każdego przełączenia (wejście dla metryk z Fazy 4)
- [ ] Komunikat po polsku, gdy **wszystkie** providery zawiodą — nie stacktrace
- [ ] Adresować `cross_document = 0.0`: prompt engineering albo większy model w łańcuchu

**Gate:** przy sztucznie zepsutym pierwszym providerze zapytanie kończy się poprawną odpowiedzią
z drugiego. `answer_score` ≥ 0.70 utrzymany, `cross_document` > 0.

---

## Faza 4 — Observability (1-2 dni)

Wprost pod projekt K8s #2 (OpenTelemetry + Jaeger + Prometheus + Grafana).

- [ ] OpenTelemetry, spans per etap: `embed_query` → `dense_search` → `bm25_search` →
      `rrf_fusion` → `rerank` → `generate`
- [ ] Metryki Prometheus na `/metrics`:
  - histogram latencji per etap (widać, że generacja to ~22 s z ~23 s całości)
  - licznik błędów per provider per kod (`404` / `429` / timeout) — bezpośrednio z awarii Fazy 3
  - licznik przełączeń fallbacku
  - licznik odmów (`out_of_scope`) — wzrost sygnalizuje regresję retrievalu
- [ ] Logi strukturalne JSON (loguru), `trace_id` spinający wpisy jednego zapytania
- [ ] `/health` (liveness) i `/ready` (readiness: Postgres + dostępność providera) — pod probes k8s
- [ ] Dashboard Grafany: latencja, error rate per provider, wykorzystanie darmowych limitów

**Gate:** jedno zapytanie widoczne jako kompletny trace z rozbiciem czasu na etapy.

---

## Faza 5 — Deployment produkcyjny (1 dzień)

- [ ] `docker/Dockerfile` produkcyjny, multi-stage (bez `dev-dependencies`)
- [ ] UI: komunikaty błędów po polsku, informacja o zimnym starcie, brak surowych tracebacków
- [ ] Podstawowa autoryzacja (nawet hasło w zmiennej środowiskowej — publiczny URL bez niej to zaproszenie do wypalenia darmowych limitów)
- [ ] Cache odpowiedzi na powtarzające się pytania (oszczędza limity, skraca latencję)
- [ ] Deployment: k3s (jeśli Faza 6) albo HF Spaces jako ścieżka awaryjna
- [ ] Krótka instrukcja dla użytkownika końcowego — po polsku, bez żargonu

**Gate:** osoba nietechniczna otwiera URL i uzyskuje poprawną odpowiedź z cytowaniem,
bez Twojej obecności.

---

## Faza 6 — Kubernetes / LLMOps (osobny projekt portfolio)

Wchodzi w zakres projektów #4 i #5 z Twojej kolejności portfolio. Repo ma być na to gotowe,
ale realizacja jest osobna.

- [ ] Manifesty k8s: `Deployment` (API), `StatefulSet` (Postgres+pgvector), `Service`, `Ingress`
- [ ] cert-manager + TLS, HPA na API
- [ ] ArgoCD GitOps
- [ ] **Eval-gated promotion:** `Job` uruchamia `run_evals` na golden dataset; `exit != 0`
      poniżej progu blokuje promocję. To jest bezpośrednie wykorzystanie harnessu z Faz 1-3
- [ ] Progi w wersjonowanym `evals/thresholds.yaml`; CI odrzuca commit obniżający próg
      bez jawnego override (zgodnie z zasadą #1 w `CLAUDE.md`)
- [ ] Argo Rollouts — canary po metrykach z Fazy 4
- [ ] Chaos engineering: ubicie poda Postgresa, symulacja awarii providera (Faza 3 to obsługuje)

---

## Ryzyka

| Ryzyko | Prawdopodobieństwo | Mitygacja |
|---|---|---|
| Kolejne darmowe modele znikają / są przeciążone | **Wysokie** — już wystąpiło 3× | Łańcuch fallbacku (Faza 3), min. 2 niezależne platformy |
| Oracle Cloud Free Tier odbiera bezczynne instancje | Średnie `[WERYFIKUJ aktualną politykę]` | HF Spaces jako ścieżka awaryjna; nie wiązać celu podstawowego z Fazą 6 |
| 15 pytań to za wąska podstawa do bramkowania | **Pewne** | Rozszerzenie datasetu w Fazie 1 przed użyciem progów w CI |
| Warunki NVIDIA wykluczają obsługę użytkowników końcowych | Znane | NVIDIA tylko do ewaluacji; runtime na Cloudflare/OpenRouter — patrz `docs/PROVIDERS.md` |
| `bge-m3` (~2.2 GB) nie mieści się w limitach pamięci targetu | Średnie | `multilingual-e5-base` jako lżejsza alternatywa, decyzja na danych z Fazy 2 |
| Latencja ~23 s frustruje użytkownika | Wysokie | Streaming odpowiedzi, cache, komunikat o postępie w UI |

---

## Kolejność i zależności

```
Faza 0 (higiena)
   └→ Faza 1 (metryki retrievalu)      ← narzędzie pomiarowe PRZED zmianami
        └→ Faza 2 (embedding, cel podstawowy odblokowany)
             └→ Faza 3 (generacja + fallback)
                  └→ Faza 4 (observability)
                       ├→ Faza 5 (deployment dla mamy)   ← cel podstawowy OSIĄGNIĘTY
                       └→ Faza 6 (Kubernetes, portfolio)
```

Po Fazie 5 cel podstawowy jest spełniony niezależnie od tego, czy Faza 6 dojdzie do skutku.
To jest celowe — sprawność narzędzia dla mamy nie może być zakładnikiem projektu portfolio.
