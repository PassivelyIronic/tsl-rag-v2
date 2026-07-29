# Przekazanie do projektu k3s

Dokument dla **innego repozytorium** — tego, w którym powstaje klaster. TSL-RAG jest
tam jednym z tenantów, nie tematem projektu. Tutaj jest wszystko, czego potrzebujesz,
żeby tę aplikację wdrożyć, i nic poza tym.

**W tym repo nie ma i nie będzie manifestów, Helm chartów ani konfiguracji ArgoCD.**
Namespace'y, quoty i polityki sieciowe są sprawą klastra.

---

## 1. Co wdrażasz

Dwa procesy plus baza:

| Komponent | Obraz / źródło | Port | Stan |
|---|---|---|---|
| API (FastAPI) | `docker/Dockerfile` | 8000 | bezstanowy |
| UI (Streamlit) | `uv run streamlit run ui.py` | 8501 | bezstanowy |
| PostgreSQL + pgvector | `pgvector/pgvector:pg16` | 5432 | **StatefulSet** |

UI jest opcjonalne — API samo w sobie jest kompletne i to ono ma probe'y oraz metryki.

Obraz produkcyjny buduje się z katalogu głównego repo:

```bash
docker build -f docker/Dockerfile -t tsl-rag-api:<tag> .
```

---

## 2. Kontrakt aplikacji

### Endpointy

| Ścieżka | Rola | Uwagi |
|---|---|---|
| `GET /health` | **liveness** | Nie dotyka bazy. Restart poda nie naprawi niedostępnej bazy |
| `GET /ready` | **readiness** | Sprawdza Postgres, retriever i provider embeddingów. 503, gdy którykolwiek zawodzi |
| `GET /metrics` | Prometheus | Bez autoryzacji — scraper nie może zależeć od sekretu aplikacji |
| `POST /query` | zapytanie RAG | Wymaga `X-API-Key`, jeśli `API_PASSWORD` jest ustawione |

**Nie podłączaj `/ready` pod liveness probe.** Zwraca 503 przy chwilowo niedostępnej
bazie, więc jako liveness restartowałby pody w pętli, zamiast tylko wypiąć je z Service.

### Czasy startu

Zmierzone (spany OpenTelemetry, Faza 4):

| Faza | Czas |
|---|---|
| Wczytanie modelu embeddingów | **7.7 s** |
| Budowa indeksu BM25 (438 chunków) | ~0.3 s |
| Zapytanie na rozgrzanym procesie | 94 ms |

`initialDelaySeconds` dla readiness ustaw na **co najmniej 60 s**, a `startupProbe`
z `failureThreshold` pokrywającym ~2 minuty. Model pobiera się z Hugging Face przy
pierwszym starcie, jeśli cache jest pusty — wtedy start trwa dłużej.

### Zasoby

Model `multilingual-e5-base` plus torch CPU. Punkt wyjścia do strojenia:

```yaml
resources:
  requests: { memory: "1Gi", cpu: "500m" }
  limits:   { memory: "2Gi", cpu: "2" }
```

Poniżej 1 Gi pod będzie ubijany przy wczytywaniu modelu.

### Wolumeny

Jeden, opcjonalny, ale mocno zalecany: cache wag modelu.

```yaml
env:
  - name: HF_HOME
    value: /cache/huggingface
volumeMounts:
  - name: hf-cache
    mountPath: /cache
```

Bez niego każdy restart poda pobiera ~1.1 GB z Hugging Face. Obraz **celowo** nie
zawiera wag — inaczej ważyłby ~2 GB i trzeba by go przepychać przy każdej zmianie kodu.

---

## 3. Konfiguracja

Cała przez zmienne środowiskowe, żadnych plików konfiguracyjnych w obrazie.
Pełna lista z opisami: `env.example`. Minimum do uruchomienia:

```yaml
# ConfigMap
POSTGRES_DSN: postgresql+asyncpg://user@postgres.tsl-rag.svc:5432/tsl_rag
EMBEDDING_PROVIDER: local
CHAT_PROVIDER: openrouter
OPENROUTER_CHAT_MODEL: nvidia/nemotron-nano-9b-v2:free
LLM_SYSTEM_PREFIX: /no_think      # 5× szybciej na tym modelu (run_017 vs run_018)
LOG_JSON: "true"                   # logi na stdout jako JSON
OTEL_EXPORTER: otlp
OTEL_ENDPOINT: http://otel-collector.observability.svc:4318/v1/traces

# Secret
OPENROUTER_API_KEY: ...
API_PASSWORD: ...                  # obowiązkowe przy publicznym Ingressie
```

**`Settings` ma `extra="forbid"`.** Zmienna bez odpowiednika w `settings.py` wywala
start komunikatem „Extra inputs are not permitted". To celowe — literówka w ConfigMapie
ma zatrzymać deployment, a nie zostać po cichu zignorowana.

---

## 4. Baza danych

Schemat: `docker/init.sql` (rozszerzenie `vector`, tabela `document_chunks`, indeks HNSW).
Wykonaj go raz przy inicjalizacji StatefulSetu.

**Korpus wgrywa się osobno i nie jest częścią deploymentu aplikacji.** Ingest wymaga
parserów PDF, których obraz API celowo nie zawiera:

```bash
uv sync --extra ingest
POSTGRES_DSN=... uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/
```

Oczekiwany wynik: 13/14 plików, 438 chunków, `failed: 0`.

**Pułapka, która nie zgłasza błędu:** model embeddingów użyty przy ingeście musi być
tym samym, którego używa runtime. Wektor policzony modelem A porównany z zapytaniem
z modelu B daje losowe wyniki. Aplikacja sprawdza to przy starcie (`warmup()`) i przerywa
z komunikatem — nie obchodź tego zabezpieczenia.

Jako job wsadowy ingest ma sens tylko przy zmianie korpusu, nie przy każdym deployu.

---

## 5. Eval-gated promotion

To jest ten element, dla którego aplikacja była projektowana pod klaster.

```bash
uv run python -m evals.run_retrieval_evals
```

- Zwraca **exit code 1**, gdy metryka spadnie poniżej progu z `evals/thresholds.yaml`.
- **Nie wymaga klucza providera ani sieci** — tylko bazy z korpusem. Nadaje się
  na Job w pipelinie promocji.
- Trwa ~40 s.
- Jest **deterministyczny**: te same dane dają te same liczby.

Aktualne progi i pomiar: `recall@5` ≥ 0.938 (jest 0.958), `recall@10` ≥ 0.948 (0.969),
MRR ≥ 0.850 (0.874).

**Nie bramkuj metrykami generacji.** Zmierzony rozrzut między przebiegami identycznego
kodu sięga 0.133 — taka bramka przepuści regresję albo zablokuje poprawę, zależnie
od losu. `evals/run_evals.py` (z LLM-em) jest do obserwacji, nie do promocji.

---

## 6. Observability

Gotowe w kodzie, nic nie trzeba dorabiać po stronie aplikacji.

**Spany** (OpenTelemetry, eksport OTLP po ustawieniu `OTEL_EXPORTER=otlp`):

```
query → retrieve → embed_query | dense_search | bm25_search | rrf_fusion | rerank
      → generate → llm_call (jeden span na PRÓBĘ, nie na zapytanie)
```

`llm_call` per próbę jest istotne przy fallbacku: pokazuje, ile czasu zjadło ogniwo,
które i tak zawiodło.

**Metryki** na `/metrics`:

| Metryka | Etykiety |
|---|---|
| `tsl_rag_stage_duration_seconds` | `stage` |
| `tsl_rag_provider_errors_total` | `provider`, `model`, `kind` |
| `tsl_rag_fallback_switches_total` | `from_target`, `to_target` |
| `tsl_rag_answers_total` | `outcome` (answered / refused / all_providers_failed / cache_hit) |

Kubełki histogramu sięgają 60 s, bo generacja na darmowym modelu potrafi trwać
kilkanaście sekund — domyślne biblioteczne (do 10 s) wrzucałyby wszystko do `+Inf`.

**Logi** — JSON na stdout przy `LOG_JSON=true`, każdy rekord z `trace_id`.

Kandydaci na alerty: wzrost `tsl_rag_provider_errors_total{kind="transient"}`,
niezerowe `outcome="all_providers_failed"`, `p95` etapu `generate` powyżej 30 s.

---

## 7. Rzeczy, które nie są oczywiste

**Aplikacja jest bezstanowa, ale nie całkiem.** `RAGGenerator` trzyma bezpiecznik
łańcucha fallbacku, a cache odpowiedzi żyje w pamięci procesu. Obie rzeczy są
per-replika i to jest w porządku — przy skalowaniu każda replika buduje własny stan,
a jedyny koszt to niższa skuteczność cache'a. Nic nie wymaga sesji przyklejonych.

**Skalowanie w poziomie ma ograniczoną wartość.** Wąskim gardłem jest generacja
u zewnętrznego providera i jego dzienny limit, nie CPU poda. HPA po CPU rozjedzie
liczbę replik bez zysku dla użytkownika. Jeśli chcesz HPA jako element portfolio,
skaluj po `tsl_rag_stage_duration_seconds` albo po liczbie żądań w locie.

**Indeks BM25 jest w pamięci** i budowany przy starcie z całego korpusu. Przy 438
chunkach to 0.3 s, ale rośnie liniowo z korpusem — przy dziesięciokrotnym wzroście
przemyśl przeniesienie go do bazy.

**Chaos engineering ma tu sensowne cele.** Ubicie poda Postgresa powinno dać `/ready`
= 503 i komunikat po polsku zamiast stacktrace'a. Zablokowanie ruchu do OpenRoutera
powinno przełączyć łańcuch na kolejne ogniwo i podbić `tsl_rag_fallback_switches_total`.
Oba zachowania są pokryte testami jednostkowymi, ale w klastrze warto je potwierdzić.

---

## 8. Czego NIE przenosić

- `docker/Dockerfile.spaces` i `ui_backend.py` — tryb jednoprocesowy pod Streamlit
  Community Cloud. W klastrze API i UI są osobnymi deploymentami.
- `docker-compose.yml` — środowisko deweloperskie.
- `data/raw/` — korpus wgrywa się raz do bazy, nie wozi się go w obrazie.
