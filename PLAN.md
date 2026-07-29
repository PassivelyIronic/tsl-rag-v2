# PLAN.md — TSL-RAG v2

## Cel

1. **Cel podstawowy:** nietechniczna osoba na słabym sprzęcie korzysta z systemu
   w przeglądarce, za darmo, o dowolnej porze — bez lokalnego GPU i bez interwencji autora.
2. **Cel portfolio:** repo jest bazą aplikacyjną pod przyszły projekt Kubernetes
   (k3s + eval-gated CI/CD) i demonstruje mierzalną jakość RAG-a oraz odporność na awarie.

Cel drugi jest **odległy i jeszcze nie rozpoczęty** — projekt Kubernetes nie istnieje.
Repo ma być na niego gotowe, ale żadna decyzja w Fazach 0-5 nie może być podejmowana
„pod K8s" na koszt celu podstawowego. Tam, gdzie cele kolidują, rozstrzygnięcie jest jawne
(§ Konflikt celów).

---

## Stan obecny (zweryfikowany 2026-07-26 na uruchomionych usługach)

### Konfiguracja, którą zostawiono na koniec sesji

| Element | Wartość | Skąd |
|---|---|---|
| Embeddingi | `intfloat/multilingual-e5-base`, **lokalnie na CPU** | Faza 2, wymiana z `nomic-embed-text` |
| Prefiksy E5 | `query: ` / `passage: ` | wymagane przez rodzinę E5 |
| Chat (repo) | `nvidia/nemotron-nano-9b-v2:free` | domyślne ustawienie zostaje darmowe |
| Reranking | **wyłączony** (`RERANKER_ENABLED=false`) | pomiar: kosztuje całość latencji retrievalu |
| Model rerankera w rezerwie | `BAAI/bge-reranker-v2-m3`, okno 2048 | najlepszy zmierzony wariant, gdyby wracać |
| Wagi RRF | 0.5 / 0.5 | przegląd czterech wariantów, brak podstaw do zmiany |
| Korpus | **438 chunków, 13 dokumentów**, zero miękkich łączników | po `normalize_pdf_text` |
| Golden dataset | **56 pytań**, min. 6 na kategorię | 42 z NotebookLM + 14 z v1 |

### Zweryfikowane pomiarem

**Retrieval (deterministyczny, bez LLM-a)** — `evals/results/retrieval_009_norerank_bm25_0.5.json`:

| Metryka | Wynik | Próg w `thresholds.yaml` |
|---|---|---|
| `recall@5` | 0.938 | 0.917 |
| `recall@10` | 0.969 | 0.948 |
| `recall@20` | **1.000** | — |
| MRR | 0.874 | 0.850 |
| mediana latencji | **0.1 s** | — |

**Generacja** — `gpt-4o-mini` jako model referencyjny, 56 pytań, `run_014`:
`answer_score` 0.653 keyword-match wobec **0.776 semantycznie**, `citation_precision` 0.929,
`retrieval_recall` 0.946, latencja 1.8 s, koszt całego przebiegu 0.029 USD.

**Korpus:** 438 chunków, 13 dokumentów, zero powtórzonych tekstów i zero miękkich łączników.
Ingest przechodzi 13/14 plików, `failed: 0`, bez tracebacków.

**Czternasty plik nie jest brakiem.** `TARIFF_EMPLOYER_2022.pdf` jest bajtowo identyczny
z `TARIFF_COMPANY_2022.pdf` (md5 `bc6f6cb8…` dla obu). Poprzednia wersja tego planu zalecała
dodanie mu wpisu w rejestrze — to było błędne założenie. Wpis wstrzykuje 15 chunków
identycznych z `tariff_company_2022`, które konkurują w retrievalu i zajmują dwa miejsca
w kontekście zamiast jednego. Plik jest pomijany celowo (`_KNOWN_DUPLICATES`).

**Baseline generacji** — `nvidia/nemotron-nano-9b-v2:free` przez OpenRouter, embeddingi
`nomic-embed-text` przez Ollamę, `top_k=20`, `rerank_top_n=5`, wagi RRF 0.5/0.5,
ocena keyword-match. Trzy przebiegi, `evals/results/run_010`–`run_012`:

| Metryka | `run_010` | `run_011` | `run_012` | Uwaga |
|---|---|---|---|---|
| `answer_score` | 0.633 | 0.600 | 0.633 | rozrzut 0.033 |
| `citation_hit_rate` | 0.733 | 0.667 | 0.800 | **rozrzut 0.133** |
| `citation_precision` | 0.800 | 0.800 | 0.733 | rozrzut 0.067 |
| `retrieval_recall` | 0.867 | 0.867 | 0.867 | **stabilny** |
| porażki retrievalu | 2 | 2 | 2 | **stabilne** |
| porażki generacji | 2 | 3 | 1 | rozrzut 2 pytania |
| `refusal_precision` | 1.000 | 1.000 | 1.000 | stabilny |
| latencja średnia | 19.1 s | 17.5 s | 15.4 s | zależna od obciążenia providera |

`run_011` i `run_012` to **ten sam kod** — między nimi nie ma żadnej zmiany.

### Wniosek, który zmienia strategię bramkowania

Metryki zależne od generacji mają przy 15 pytaniach rozrzut do **0.133 między
przebiegami identycznego kodu** — większy niż efekt, jakiego można oczekiwać od typowej
zmiany w retrievalu. Oznacza to, że `answer_score` i `citation_hit_rate` **nie nadają się
dziś na bramkę CI**: przepuszczą regresję albo zablokują poprawę, w zależności od losu.

Metryki retrievalu są za to w pełni stabilne: `retrieval_recall = 0.867` i te same dwie
porażki w każdym z trzech przebiegów — bo nie ma w nich LLM-a.

**Konsekwencja:** bramka promocji musi stać na metrykach retrievalu (`run_retrieval_evals.py`,
Faza 1), które są deterministyczne i nie wymagają klucza providera. Metryki generacji zostają
jako obserwowane, nie bramkujące, dopóki dataset nie urośnie na tyle, żeby rozrzut zmalał.

**Etap porażki — to jest najważniejszy wynik tej weryfikacji.** Harness rozdziela teraz
„retriever nie podał" od „model zignorował kontekst", więc trzy kategorie, które wcześniej
były nieinterpretowalnym zerem, mają rozpoznanie:

| Kategoria | Etap | Co się faktycznie stało |
|---|---|---|
| `penalty` | **retrieval** | `tariff_driver_2022` nigdy nie wszedł do kontekstu. Model zacytował `tariff_company_2022` i `eu_2016_403` — czyli to, co dostał. Nie jest to „mylenie taryfikatorów" przez model |
| `cross_document` | **retrieval** | Żaden z dwóch oczekiwanych aktów nie wszedł w komplecie |
| `scope` | **generacja** | `aetr` **był** w kontekście (`ret=1.00`), a model zacytował `ec_561_2006` i `eu_1072_2009` |
| `numeric_fact` (1 z 9) | **generacja** | Poprawna treść, brak cytowania |

**Diagnoza dla `penalty` — NIEAKTUALNA, zastąpiona pomiarem z 2026-07-27** (Faza 1 niżej).
Brzmiała: asymetria korpusu, bo taryfikator kierowcy ma 5 chunków, przedsiębiorcy 15,
a klasyfikacja naruszeń `eu_2016_403` — 24. Liczby się zgadzają, ale wniosek był zły:
PDF kierowcy jest kompletny, a filtr po `contains_penalty` nic nie da, bo konkurujące
chunki mają tę samą flagę. Realna przyczyna to dense (MRR 0.352 na tej kategorii wobec
0.619 dla BM25), a mierzona faktami w kontekście kategoria wypada wcale nie najgorzej.

### Naprawione w tej iteracji

- Repozytorium git założone, CI (ruff + format + mypy + pytest) zielone
- Slugi modeli ujednolicone do jednego zweryfikowanego; usunięte trzy martwe
- Martwa konfiguracja podłączona albo usunięta (wagi RRF, `top_k`, limit kontekstu, chunker)
- `HybridRetriever` tworzony raz na proces, nie per request
- `/health` i `/ready` rozdzielone; komunikaty błędów po polsku
- Harness: `retrieval_recall`, `citation_precision`, `failure_stage`, snapshot konfiguracji
- Tokenizer BM25 składa polskie diakrytyki (wcześniej `[a-z0-9]+` rozrywał słowa)
- `ensure_utf8_output()` — ingest wywalał się na cp1250 przy wypisywaniu strzałki
- mypy: 40 błędów → 0; `pytest -m unit` faktycznie uruchamia testy (było 0 z 11)

### Nadal nie działa

- ~~Brak łańcucha fallbacku providerów~~ — **zrobione 2026-07-27** (Faza 3). Domyślnie
  wyłączony (`CHAT_FALLBACK_CHAIN` puste), bo repo jest publiczne; we wdrożeniu dla
  użytkownika końcowego wypełnia się go zgodnie z decyzją o płatnym modelu jako pierwszym
- ~~Brak jakiejkolwiek observability~~ — **zrobione 2026-07-27** (Faza 4): spany OTel
  per etap, metryki Prometheusa na `/metrics`, logi z `trace_id`. Dashboard zostaje
  poza repo, bo Grafana należy do klastra
- Brak testów integracyjnych (`tests/integration`, `tests/e2e` są puste)
- ~~Darmowy model generacji: 17.3 s i puste odpowiedzi~~ — **rozwiązane 2026-07-28**
  przez `LLM_SYSTEM_PREFIX=/no_think`: latencja 25.7 s → 5.0 s (mediana 25.9 → 4.0 s),
  zero pustych odpowiedzi. Patrz Faza 3
- `--use-judge` niesprawny — klucz Gemini ma zerowy limit
- ~~Repo nie ma remote'a~~ — **wypchnięte 2026-07-29** na `github.com/PassivelyIronic/tsl-rag-v2` (38 commitów). Historia sprawdzona przed pushem: `.env` nigdy nie był commitowany, brak kluczy API i hasła bazy

---

## Następna sesja — od czego zacząć

**Przed czymkolwiek:** `docker compose up -d`, Ollama tylko jeśli potrzebna generacja lokalna
(embeddingi jej już nie wymagają). Sanity check bez kosztów i bez limitów:

```powershell
uv run python -m evals.run_retrieval_evals    # ~40 s, ma pokazać "Bramka spełniona"
uv run pytest -m unit                          # 77 testów
```

Kolejność, w mojej ocenie:

1. ~~Push na GitHub~~ — **zrobione 2026-07-29**, `github.com/PassivelyIronic/tsl-rag-v2`.
2. **Nowy baseline generacji na 56 pytaniach** — z `/no_think` i przy `rrf_k=5`.
   56 wywołań przy dziennym limicie 50, więc albo w dwóch dobach, albo na płatnym
   `gpt-4o-mini` za ~0.03 USD.
3. ~~Sprawdzić `LLM_REASONING_EFFORT=none` na nemotronie~~ → **zamknięte**. Parametr
   na tym modelu nie działa, mechanizmem jest `LLM_SYSTEM_PREFIX=/no_think`, a efekt
   został zmierzony parowanym przebiegiem 2026-07-28 (tabela niżej).
4. **Faza 3: łańcuch fallbacku** z płatnym modelem jako pierwszym ogniwem (decyzja niżej).

### Rozumowanie nemotrona: parametr API nie działa, token w promptcie działa

Zmierzone 2026-07-27, jedno pytanie, ten sam kontekst 5 chunków, `LLM_MAX_TOKENS=1024`.
Model **deklaruje** `reasoning` w `supported_parameters` OpenRoutera, więc deklaracja
providera nie jest tu dowodem:

| Dźwignia | tokeny rozumowania | tokeny wyjścia łącznie |
|---|---|---|
| brak parametru | 381, 298 | 467, 398 |
| `reasoning: {"effort": "none"}` | 436, 453 | 508, 570 |
| `reasoning: {"enabled": false}` | 345 | 446 |
| `reasoning: {"exclude": true}` | 354 | 440 |
| **`/no_think` w system promptcie** | **0** | **86** |

Wdrożone jako `LLM_SYSTEM_PREFIX` — jawna konfiguracja, nie zgadywana z nazwy modelu,
tak samo jak prefiksy E5 przy embeddingach (`CLAUDE.md` §3). `LLM_REASONING_EFFORT`
zostaje, bo dotyczy innych providerów, ale ma teraz w komentarzu zapisane, że na modelu
domyślnym jest bezskuteczny.

**Baseline podzbioru zmierzony** — `run_015_nemotron21_baseline.json`, 21 pytań dobranych
warstwowo (wszystkie 6 kategorii), bez prefiksu, 21/21 bez błędu providera:

| Metryka | Wynik |
|---|---|
| `answer_score` (keyword) | 0.508 |
| `semantic_score` | 0.770 |
| `citation_hit_rate` | 0.786 |
| `citation_precision` | 0.810 |
| `retrieval_recall` | 0.929 |
| latencja średnia | 17.3 s |
| `refusal_precision` | 1.000 |
| puste odpowiedzi (faktyczne) | 2 z 21 |
| `has_answer=False` łącznie | 6 z 21 (4 to poprawne odmowy `out_of_scope`) |

Pierwszy przebieg porównawczy (2026-07-27) był **nieważny**: Git Bash zamienił `/no_think`
na `C:/Program Files/Git/no_think`, więc do system promptu poszła ścieżka. Plik zachowany
jako `run_016_NIEWAZNY_prefiks_zmangowany.json` wyłącznie jako ślad pomyłki. Nie cytuj
z niego liczb.

### `/no_think` ZMIERZONE — 5× szybciej i lepiej w każdej metryce treści

Parowany przebieg 2026-07-28, **te same 21 pytań**, ten sam `rrf_k=5`, jedyna różnica
to `LLM_SYSTEM_PREFIX`. Oba 21/21 bez błędu providera
(`run_017_nemotron21_k5_baseline.json` i `run_018_nemotron21_k5_nothink.json`):

| Metryka | bez prefiksu | `/no_think` | delta |
|---|---|---|---|
| latencja średnia | 25.7 s | **5.0 s** | **−20.6 s** |
| latencja mediana | 25.9 s | **4.0 s** | **−21.9 s** |
| `answer_score` (keyword) | 0.532 | **0.627** | +0.095 |
| `semantic_score` | 0.722 | **0.810** | +0.088 |
| `citation_precision` | 0.690 | **0.786** | +0.096 |
| `citation_hit_rate` | 0.786 | 0.762 | −0.024 |
| `retrieval_recall` | 0.929 | 0.929 | 0.000 |
| `refusal_precision` | 1.000 | 1.000 | 0.000 |
| `false_refusal_rate` | 0.000 | 0.000 | 0.000 |
| puste odpowiedzi | 0 | 0 | — |

**Wniosek: `/no_think` to czysty zysk na tym modelu.** Pięciokrotne przyspieszenie
nie kosztuje jakości — trzy z czterech metryk treści rosną o ~0.09, a jedyny spadek
(`citation_hit_rate`, −0.024) to pół pytania z 21, czyli poniżej zmierzonego rozrzutu
metryk generacji (0.133 między przebiegami identycznego kodu). Odmowy zachowują się
identycznie, więc przyspieszenie nie wzięło się z tego, że model przestał myśleć nad
tym, czy w ogóle wie.

**Rekomendacja wdrożeniowa:** `LLM_SYSTEM_PREFIX=/no_think` w `.env` wdrożenia, o ile
modelem generacji jest nemotron. Domyślna wartość w repo zostaje pusta, bo token jest
specyficzny dla tej rodziny modeli i wysłany gdzie indziej zostaje w promptcie jako śmieć.

**Sprostowanie do wcześniejszego zapisu:** raportowałem „6 pustych odpowiedzi na 21 pytań"
w `run_015`. To było błędne — policzyłem `has_answer=False`, a w tym polu mieszczą się
też **poprawne odmowy**. Faktycznie pustych odpowiedzi było **2 z 21**, a pozostałe 4 to
odmowy na pytania `out_of_scope`, punktowane na 1.000. Mechanizm „pustka → przełącz ogniwo"
w Fazie 3 zostaje słuszny, ale częstość była zawyżona trzykrotnie.

**Flaga `--limit` w `run_evals`** dobiera pytania **warstwowo po kategoriach**, nie
„pierwsze N" — pierwsze 21 pytań datasetu to wyłącznie `numeric_fact`, `procedure`
i `scope`, czyli podzbiór bez `out_of_scope` nie mierzyłby `refusal_precision`,
a bez `penalty` pomijałby najsłabszą kategorię retrievalu. Dobór jest deterministyczny,
żeby przebiegi przed/po dotyczyły dokładnie tych samych pytań.

Do rozważenia bez pilności: recall liczony po artykułach zamiast po dokumentach
(obecny zawyża — patrz błędne odmowy przy modelu referencyjnym), podział dużych chunków
tabelarycznych, LLM-as-a-judge za ~0.03 USD za przebieg.

---

## Konflikt celów — rozstrzygnięcia

| Kwestia | Optimum dla użytkownika | Optimum dla portfolio | Decyzja |
|---|---|---|---|
| Vector store | numpy brute-force (444 chunki, <10 ms) | Postgres StatefulSet | **Postgres** — portfolio wygrywa, koszt złożoności akceptowalny |
| API | in-process w Streamlicie | osobny FastAPI (HPA, probes, canary) | **Osobny FastAPI** |
| Frontend | Streamlit | React/Vite/shadcn | **Streamlit** — jeden użytkownik, jeden ekran |
| Kolejka zadań | brak (ingest ręczny) | Celery + Redis | **Brak** — Celery byłby atrapą |
| Hosting | HF Spaces (najprostszy) | k3s Oracle Free Tier | **HF Spaces** jako ścieżka domyślna, k3s jeśli Faza 6 powstanie |

Mapowanie na stack kanoniczny i reguła wyjścia poza niego: `CLAUDE.md` §6.

---

## Faza 0 — Higiena ✅ ZROBIONA (2026-07-26)

- [x] Repozytorium git + CI (`ruff`, `ruff format`, `mypy`, `pytest -m unit`) — zielone
- [x] Usunięte martwe wpisy z `model_comparison.json` (dwa modele zwracające 404)
- [x] `DEFAULT_CANDIDATES` → tylko zweryfikowany slug
- [x] `ingest_all`: jeden `asyncio.run()` na całą pętlę
- [x] Naprawione ścieżki (`Makefile` cel `ui`, `docs/PROVIDERS.md`)
- [x] README przepisany na stan zweryfikowany, bez liczb z pamięci
- [x] ~~Dodać `tariff_employer_2022` do rejestru~~ → **odrzucone po weryfikacji**: plik jest
      duplikatem, nie brakującym dokumentem

**Gate osiągnięty:** ingest przechodzi bez tracebacków, `failed: 0`, 13 dokumentów
(14. pominięty świadomie), CI zielone.

---

## Faza 1 — Metryki retrievalu 🔶 CZĘŚCIOWO ZROBIONA

Zrobione:

- [x] Rozdzielenie porażki retrievalu od generacji (`retrieval_recall`, `failure_stage`)
- [x] `retrieved_docs` w rekordzie wyniku
- [x] `citation_precision` obok recallu
- [x] Snapshot konfiguracji w pliku wyniku
- [x] Wyjaśnione `penalty` (retrieval) i `scope` (generacja)

Zostaje:

- [x] **Miękkie łączniki usunięte** (`normalize_pdf_text`) — zweryfikowane na Neonie 2026-07-29: zero wystąpień U+00AD w 438 chunkach. Historyczny opis problemu: Ekstrakcja z PDF-a zostawia
      U+00AD w miejscach podziału wiersza: **1258 wystąpień w 307 z 444 chunków (69%)**,
      we wszystkich rozporządzeniach UE. Korpus zawiera więc `przynaj­ mniej`, `wyko­ rzystać`,
      `tygodnio­ wego`. Skutki: tokenizer BM25 robi z jednego słowa dwa bezużyteczne tokeny
      (`tygodnio` + `wego`), więc poprawne zapytanie nigdy nie trafia w te miejsca, a model
      dostaje w kontekście tekst z rozerwanymi słowami. **To jest prawdopodobnie większa
      dźwignia niż składanie diakrytyków** i dotyczy dokumentów najczęściej pytanych.
      Wymaga ponownego ingestu i pomiaru przed/po — czyli narzędzia poniżej
- [x] **`evals/run_retrieval_evals.py` — ZROBIONE.** Ewaluacja samego retrievalu, bez wywołania LLM:
      `recall@k` dla `k ∈ {5, 10, 20}`, `MRR`, metryki osobno **przed** i **po** rerankingu.
      **To jest teraz zadanie o najwyższym priorytecie w tej fazie** — pomiar wariancji pokazał,
      że tylko metryki retrievalu są stabilne, więc tylko one mogą bramkować. Dodatkowo:
      przebieg bez kosztu i bez klucza API, czyli nadaje się do CI. Odpowiada też na pytanie,
      czy reranker pomaga, czy szkodzi
- [x] **Progi bramkujące w `evals/thresholds.yaml`** (2026-07-26): `recall@5` ≥ 0.917,
      `recall@10` ≥ 0.948, MRR ≥ 0.850, przy pomiarze 0.938 / 0.969 / 0.874. Margines to
      mniej więcej jedno pytanie z 48. `run_retrieval_evals` sprawdza je domyślnie i zwraca
      exit code 1 poniżej progu, czyli nadaje się na bramkę promocji

### Zmierzone wąskie gardło: cross-encoder, nie fuzja ani embeddingi

Trzy przebiegi ewaluatora retrievalu na 48 pytaniach dają spójny obraz
(`evals/results/retrieval_001`, `retrieval_002`, `retrieval_sweep_*`):

| Wagi BM25/dense | `fused` recall@5 | `fused` MRR | po rerankingu recall@5 |
|---|---|---|---|
| 0.5 / 0.5 (obecne) | 0.823 | 0.719 | 0.854 |
| 0.7 / 0.3 | 0.865 | 0.727 | 0.854 |
| 0.85 / 0.15 | 0.896 | 0.750 | 0.854 |
| 1.0 / 0.0 (BM25 sam) | **0.948** | **0.833** | 0.854 |

Wnioski, wszystkie sprzeczne z założeniami architektury:

1. **Dense nie wnosi nic przy żadnej wadze.** `recall@5` fuzji rośnie
   monotonicznie, im mniejszy udział dense'a, aż do 0.948 przy jego wyłączeniu.
   `nomic-embed-text` na polskim tekście prawnym daje sam z siebie 0.729.
2. **Reranker jest wąskim gardłem, nie fuzja.** Po rerankingu `recall@5`
   wynosi 0.854 **niezależnie od wag** — bo cross-encoder przestawia ten sam
   zbiór 20 kandydatów (`recall@20` = 0.979 wszędzie). Przy BM25-only kosztuje
   to 0.094 recall@5 i 0.097 MRR względem samej fuzji.
3. Prawdopodobna przyczyna: `ms-marco-MiniLM-L-6-v2` to model **angielski**,
   stosowany do polskiego tekstu prawnego.

### Reranking: pięć wariantów zmierzonych, rekomendacja to wyłączenie

Pomiary na korpusie z embeddingami `multilingual-e5-base`, 48 pytań
(`evals/results/retrieval_003`–`008`):

| Wariant | recall@5 | recall@10 | MRR | `penalty` r@5 | mediana |
|---|---|---|---|---|---|
| **bez rerankingu** | 0.938 | 0.969 | 0.874 | 0.857 | **0.1 s** |
| `ms-marco-MiniLM` (512, obecny) | 0.854 | 0.917 | 0.716 | 0.429 | 1.5 s |
| `bge-reranker-base` (512) | 0.865 | 0.938 | 0.746 | 0.429 | 8.9 s |
| `bge-reranker-v2-m3` (512) | 0.948 | 1.000 | 0.873 | 0.857 | 26.2 s |
| `bge-reranker-v2-m3` (2048) | 0.969 | 1.000 | 0.878 | 1.000 | 43.4 s |

Wnioski:

1. **Obecny domyślny reranker jest najgorszym z możliwych wariantów.** Kosztuje
   1.5 s i obniża recall@5 o 0.084 względem niewłączania go w ogóle.
2. **Retrieval bez rerankingu trwa 0.1 s**, więc reranking to praktycznie
   całość kosztu tego etapu.
3. Zysk najlepszego wariantu to +0.031 recall@5 za 434-krotny wzrost latencji.
   Przy generacji trwającej ~13 s daje to blisko minutę na pytanie — nie do
   pogodzenia z celem podstawowym.
4. Znaczenie mają **oba** czynniki: model (v2-m3 przy 512 daje 0.948 wobec
   0.865 dla bge-base) i okno (0.948 → 0.969, `penalty` 0.857 → 1.000).
   Okno tłumaczy się wprost rozmiarem chunków: taryfikatory mają średnio
   516-585 tokenów, czyli powyżej limitu 512 — reranker ocenia obcięty
   fragment, w którym wiersza z karą nie ma.

**Decyzja (2026-07-26): reranking WYŁĄCZONY**, `RERANKER_ENABLED=false`, na wniosek
właściciela repo — czas odpowiedzi ważniejszy niż +0.031 recall@5. Odnotowane
w tabeli §6.3 `CLAUDE.md`. `RERANKER_MODEL` i `RERANKER_MAX_LENGTH` celowo
wskazują najlepszy zmierzony wariant, żeby powrót był jedną zmianą.

- [x] Decyzja o wyłączeniu rerankingu
- [x] Przegląd wag RRF po wyłączeniu — **bez zmian, zostaje 0.5/0.5**:

| Wagi BM25/dense | recall@5 | recall@10 | recall@20 | MRR | `numeric_fact` |
|---|---|---|---|---|---|
| **0.5 / 0.5** | 0.938 | 0.969 | **1.000** | **0.874** | **1.000** |
| 0.7 / 0.3 | 0.938 | 0.979 | 0.979 | 0.853 | 1.000 |
| 0.85 / 0.15 | 0.938 | 0.979 | 0.979 | 0.838 | 1.000 |
| 1.0 / 0.0 | **0.948** | 0.969 | 0.979 | 0.833 | 0.950 |

BM25-only wygrywa recall@5 o 0.010, czyli o jedno pytanie z 48, ale traci
0.041 MRR, psuje największą kategorię (`numeric_fact` 1.000 → 0.950) i traci
`recall@20` = 1.000. MRR ma tu znaczenie praktyczne: pięć chunków taryfikatora
to ~12 290 znaków przy `max_context_chars` = 12 000, więc ostatni bywa obcięty
i kolejność decyduje o tym, co przetrwa. Dodatkowo przy wadze 0 dense stałby
się czystym kosztem — embeddingi liczylibyśmy przy każdym zapytaniu na nic.
- [ ] Wrócić do `v2-m3`, jeśli zmieni się budżet latencji (cache odpowiedzi
      z Fazy 5, mocniejszy sprzęt) — to jest jakość dostępna od ręki za czas
- [ ] Rozważyć podział dużych chunków tabelarycznych. Uwaga: taryfikatory mają
      średnio 2458 znaków, więc przy `max_context_chars=12000` **pięć chunków
      wypełnia cały budżet kontekstu**. Podział musi iść po grupach wierszy
      z powtórzonym nagłówkiem — rozerwanie opisu naruszenia od kwoty to
      gotowa halucynacja o wysokości grzywny
- [x] **Format datasetu** — `questions.json` + walidacja przy wczytaniu i w testach,
      specyfikacja i prompt do generowania w `docs/GOLDEN_DATASET.md`
- [x] **Golden dataset rozszerzony z 15 do 56 pytań** (2026-07-26), min. 6 na kategorię.
      Materiał wygenerowany przez NotebookLM nad korpusem PDF, scalony i zweryfikowany.
      Warianty: 46 `standard`, 6 `bez_ogonkow`, 4 `potoczne`
- [x] **Weryfikacja datasetu względem korpusu** (`evals/verify_dataset.py`) — dla każdego
      pytania sprawdzane, czy fragmenty `expected_answer` faktycznie występują w tekście
      dokumentów z `expected_docs`. Stan: **48/48 pytań z oczekiwaną treścią ma oparcie
      w korpusie**. Narzędzie wykryło trzy wadliwe pytania z datasetu v1, w których oczekiwany
      fragment nie występował we wskazanym dokumencie — patrz niżej
- [ ] Ocena `--use-judge` na rozszerzonym datasecie; keyword-match karze poprawne odpowiedzi
      sformułowane inaczej
- [ ] **Nowy baseline na 56 pytaniach.** Przebiegi `run_010`–`run_012` dotyczą datasetu v1
      (15 pytań) i nie są porównywalne z przyszłymi. Uwaga na limit: 56 pytań = 56 wywołań,
      a darmowy dzienny limit OpenRoutera to 50

### Trzy pytania z datasetu v1 były wadliwe jako narzędzie pomiaru

Weryfikacja wykazała, że dotyczyło to **dokładnie tych trzech kategorii, które raportowałem
jako porażki systemu**. Oczekiwany fragment nie występował we wskazanym dokumencie, więc
pytanie nie mogło dostać punktu niezależnie od jakości odpowiedzi:

| Pytanie | Było | Problem | Jest |
|---|---|---|---|
| `penalty-kierowca-czas-jazdy` | `grzywna` | Taryfikator używa formy „Wysokość grzywny"; mianownik nie występuje | `50, 100` (zał. 1, lp. 5.1) |
| `aetr-zakres-panstwa-trzecie` | `państwa trzecie` | **AETR nie używa tego pojęcia** — mówi o „umawiających się stronach". Fragment występował w innych aktach, więc model cytujący `ec_561_2006` **nie był w błędzie** | `umawiającej się strony, międzynarodowego przewozu drogowego` |
| `cross-561-vs-200215-tygodniowe-limity` | `56 godzin prowadzenia, 48 godzin średni czas pracy` | Sformułowania nieobecne w żadnym akcie | `56 godzin, 48 godzin` |

**Konsekwencja dla wcześniejszej diagnozy:** porażka kategorii `scope` była defektem datasetu,
nie generacji. Zerowe `retrieval_recall` dla `penalty` i `cross_document` pozostaje w mocy,
bo ta metryka porównuje zwrócone dokumenty z oczekiwanymi i nie zależy od `expected_answer`.
- [x] **`penalty` zdiagnozowane (2026-07-27) — obie hipotezy z tego punktu są błędne.**

      **PDF kierowcy jest kompletny:** 5 stron, 12 388 znaków ekstrakcji, z czego 10 830
      w chunkach. To mały dokument, nie urwany skan ani skan bez warstwy tekstowej.

      **Filtr `contains_penalty` nie zadziała**, bo konkurenci mają tę samą flagę. W pytaniu
      `penalty-kierowca-czas-jazdy` pozycje 1-4 to cztery chunki **tego samego** dokumentu
      i artykułu (`eu_2016_403`, Art. 3), wszystkie z `contains_penalty=true`; oczekiwany
      `tariff_driver_2022` stoi na 5. Boost na tę flagę zostawiłby je dokładnie tam, gdzie są.

      **Co jest naprawdę:** to dense ciągnie kategorię w dół. MRR na `penalty`:
      bm25 0.619, fused 0.480, dense 0.352. Pozycje pierwszego trafienia w dense to
      7, 5, 14, 19 albo wcale. Fuzja 50/50 psuje ranking, który BM25 miał poprawny.

      **Skala problemu jest jednak mniejsza, niż mówi MRR.** Mierzone faktami w treści
      kontekstu (metryka niżej) `penalty` ma `fakty@5` = 0.857, czyli tyle co recall
      dokumentowy — treść z odpowiedzią zwykle JEST w piątce, tylko nisko. Najsłabszą
      kategorią w tym ujęciu jest `scope` (0.625 przy recall 0.875), nie `penalty`.

- [x] **Recall po treści, nie po dokumentach — `fact_recall@k`** (2026-07-27).
      `expected_articles` jest w schemacie datasetu, ale puste we wszystkich 56 pytaniach,
      więc metryka nie opiera się na anotacjach: sprawdza, ile fragmentów `expected_answer`
      występuje **w tekście pobranych chunków**, tym samym dopasowaniem co ocena odpowiedzi
      (granica cyfry dla liczb, składanie diakrytyków). Baseline, `retrieval_011`:

| etap | recall@5 | MRR | `fakty@5` | `fakty@20` |
|---|---|---|---|---|
| dense | 0.854 | 0.758 | 0.809 | 0.892 |
| bm25 | 0.948 | 0.833 | 0.858 | 0.878 |
| **fused** | **0.938** | **0.874** | **0.840** | **0.951** |

      Per kategoria `fakty@5`: procedure 1.000, numeric_fact 0.875, penalty 0.857,
      cross_document 0.833, **scope 0.625**.

- [x] **Dywersyfikacja per dokument ODRZUCONA na podstawie pomiaru** (2026-07-27).
      Diagnoza `penalty` sugerowała ograniczenie liczby chunków jednego dokumentu w top-5.
      Symulacja na 48 pytaniach (cap to czysta obróbka listy po fuzji, więc mierzalna
      bez zmiany kodu):

| cap chunków/dokument | doc recall@5 | `fakty@5` |
|---|---|---|
| bez (obecne) | 0.938 | **0.840** |
| 3 | 0.938 | 0.840 |
| 2 | 0.938 | 0.840 |
| 1 | **0.969** | **0.727** |

      `cap=1` kupuje +0.031 recall@5 — dokładnie tyle, ile najlepszy cross-encoder za 43 s —
      i **jednocześnie wyrzuca treść z odpowiedzią z kontekstu**: `fakty@5` spada o 0.113,
      a traci każda kategoria (`numeric_fact` 0.875→0.72, `procedure` 1.000→0.83,
      `scope` 0.625→0.50). Cap 2 i 3 są dokładnie neutralne, więc nie kupują nic.

      **To jest wzorcowy przykład grania pod metrykę:** recall@k liczy RÓŻNE DOKUMENTY
      w top-5, a cap mechanicznie zwiększa ich liczbę — poprawa jest częściowo
      tautologiczna. Bez `fact_recall@k` zmiana zostałaby zaraportowana jako darmowy zysk
      i cicho pogorszyła odpowiedzi. Nie wracaj do tego pomysłu bez pomiaru faktami.

- [x] **`scope` zdiagnozowane i naprawione stałą `rrf_k`** (2026-07-27). Kategoria miała
      recall dokumentowy 0.875 przy `fakty@5` = 0.625, czyli właściwy AKT wchodził bez
      właściwego PRZEPISU. Zawodziły 3 z 8 pytań, w dominującym wzorcu „właściwy dokument,
      niewłaściwy chunk": `aetr::0003` (Art. 2, zakres stosowania) zawiera odpowiedź na dwa
      z tych pytań i nie wchodził do piątki, choć inne chunki AETR wchodziły.

      **Mechanizm — wada fuzji, nie kategorii:**

| zapytanie | dense | bm25 | po fuzji (`k=60`) |
|---|---|---|---|
| „Czy do wojska stosuje się AETR?" | poza | **5** | **11** |
| „definicja przewozu kabotażowego" | **3** | poza | **10** |
| „Co to jest przewoz kabotazowy" | 4 | 3 | **2** |

      RRF **degradował** chunk oceniony wysoko przez jedną listę, gdy druga go nie miała.
      Powód jest arytmetyczny: przy `k=60` i dwóch listach po 20 pozycji ranga 1 dostaje
      1/61, a ranga 20 — 1/80, czyli stosunek 1.31. Pozycja przestaje mieć znaczenie
      i wygrywa sama zgodność list. Chunk z pozycji 3 przegrywał z chunkami, które obie
      listy oceniły na 8 i 9. To samo tłumaczy `penalty`, gdzie dense psuł poprawny
      ranking BM25.

      **Przegląd `k` na 48 pytaniach** (żadna kategoria nie traci przy żadnym `k`):

| `k` | recall@5 | recall@10 | MRR | `fakty@5` |
|---|---|---|---|---|
| 60 (było) | 0.938 | 0.969 | 0.874 | 0.840 |
| 10 | 0.938 | 0.969 | 0.876 | 0.861 |
| **5 (wybrane)** | **0.958** | 0.969 | 0.874 | **0.882** |
| 2 | 0.969 | 0.979 | 0.874 | 0.882 |
| 0 | 0.958 | **1.000** | 0.864 | **0.903** |

      Wybrano 5, a nie szczyt: przy 48 pytaniach różnica 0.02 to jedno pytanie, czyli szum,
      a `k ∈ [1,5]` jest płaskowyżem. Wybieranie maksimum byłoby strojeniem pod zbiór testowy.
      Uzasadnienie jest przy tym niezależne od tych 48 pytań — `k=60` pochodzi z fuzji wielu
      list na korpusie skali TREC, a nie dwóch list po 20 pozycji.

      **Potwierdzone przebiegiem na realnym kodzie** (`retrieval_012_rrf_k5.json`, symulacja
      to nie pomiar): recall@5 0.938 → **0.958**, `fakty@5` 0.840 → **0.882**, MRR bez zmian
      0.874, recall@20 1.000. `scope` 0.875/0.625 → **1.000/0.750**, `numeric_fact` fakty
      0.875 → **0.925**.

      Zostaje: `scope` ma nadal `fakty@5` = 0.750, a wariant potoczny („Czy do wojska stosuje
      się AETR?") to luka słownikowa — korpus mówi „siły zbrojne", użytkownik „wojsko".
      Tego `k` nie naprawi.
- [x] **Model referencyjny — ZMIERZONY** (2026-07-26). Klucz OpenRouter obsługuje modele
      płatne, więc osobny klucz OpenAI nie był potrzebny. `openai/gpt-4o-mini`, 56 pytań,
      ocena keyword-match, **koszt całego przebiegu 0.029 USD**.
      Plik: `evals/results/run_014_reference_gpt4o_mini.json`.

| Metryka | `gpt-4o-mini` (referencja) | `nemotron-nano-9b:free` (v1, 15 pytań) |
|---|---|---|
| `answer_score` | 0.653 | 0.633 |
| `citation_hit_rate` | 0.804 | 0.733 |
| `citation_precision` | **0.929** | 0.800 |
| `retrieval_recall` | 0.946 | 0.867 |
| `false_refusal_rate` | **0.125** | 0.000 |
| latencja średnia | **1.8 s** | 19.1 s |

Kolumny nie są wprost porównywalne (inny dataset), ale rząd wielkości mówi swoje.

**Wniosek dla sufitu: wąskim gardłem nie jest model.** Dziesięciokrotnie mocniejszy
model daje `answer_score` = 0.653, czyli tyle co darmowy. Skoro `retrieval_recall`
wynosi 0.946, a `citation_precision` 0.929, to ograniczeniem jest albo **metryka**
(keyword-match karze poprawne parafrazy), albo **granulacja kontekstu**.

**Uwaga o granicach `retrieval_recall`:** ta metryka mówi, że właściwy DOKUMENT trafił
do kontekstu, nie że trafił właściwy PRZEPIS. Widać to w błędnych odmowach: na 6
przypadków aż 3 miały `retrieval_recall` = 1.00, czyli model dostał właściwy akt
i mimo to odmówił. Prawdopodobnie w pięciu chunkach nie było akurat tego artykułu.
Do rozstrzygnięcia: recall na poziomie artykułu, nie dokumentu.

- [x] **Rozstrzygnięte: to był limit METRYKI, nie systemu.** Dodany scorer semantyczny
      (`evals/semantic_scorer.py`) — porównuje oczekiwane fakty ze zdaniami odpowiedzi
      przez embeddingi. Darmowy i deterministyczny, bo używa modelu wczytanego już do
      retrievalu. Przeliczenie odpowiedzi z przebiegu referencyjnego:

| Metoda oceny | Wynik na tych samych odpowiedziach |
|---|---|
| keyword-match | 0.596 |
| **semantyczna** | **0.776** |

      Wartość semantyczna jest **zaniżona** — 24 z 56 odpowiedzi było w zapisie obciętych
      do 200 znaków. Rekord wyniku zapisuje teraz pełną treść, więc kolejny przebieg da
      liczbę bez tego zastrzeżenia.

      **Fakty liczbowe są w tym scorerze sprawdzane DOSŁOWNIE**, nie semantycznie. „Dzienny
      czas jazdy to 11 godzin" jest semantycznie niemal identyczne z „9 godzin" — te same
      słowa, ten sam przepis, różnica jednej cyfry. W systemie o limitach czasu pracy
      i wysokości kar metryka, która tego nie łapie, byłaby gorsza niż bezużyteczna.

- [ ] **LLM-as-a-judge — do decyzji, nie jest już konieczny.** Scorer semantyczny odpowiedział
      na pytanie o sufit. Sędzia wnosi jedno, czego embeddingi nie zrobią: wykrycie odpowiedzi
      płynnej i błędnej, bo semantycznie bliska bzdura dostaje wysoką ocenę. Koszt przy
      `gpt-4o-mini` przez OpenRouter: ~0.03 USD za przebieg. Sędzia na Gemini pozostaje
      niesprawny (zerowy limit klucza)
- [ ] **`--use-judge` jest dziś niesprawny** — ten sam zerowy limit klucza Gemini. Dokumentacja
      opisuje ocenę semantyczną jako działającą, a nie jest. Do rozstrzygnięcia razem z modelem
      referencyjnym: albo klucz z limitem, albo sędzia na innym providerze. Dopóki to nie
      zadziała, jedyną dostępną oceną jest keyword-match, który karze poprawne odpowiedzi
      sformułowane inaczej niż `expected_answer`

**Gate:** dla każdego pytania wiadomo, czy porażka leży w retrievalu, czy w generacji
(✅ osiągnięte dla obecnych 15), przy min. 5 pytaniach na kategorię (❌), ze znanym sufitem
jakości z modelu referencyjnego (❌).

---

## Faza 2 — Odcięcie od lokalnego GPU

Blocker celu podstawowego. Embedding liczony jest przy **każdym** zapytaniu, nie tylko
przy ingeście — to najbardziej wrażliwy na awarię punkt pipeline'u.

Faza 1 dała tej fazie drugie, mocniejsze uzasadnienie: `nomic-embed-text` osiąga
`recall@5` = 0.729 na polskim tekście prawnym i **nie wnosi nic do fuzji przy żadnej
wadze**. Wymiana embeddingów przestała być wyłącznie kwestią odcięcia od GPU — jest
też kandydatem na realną poprawę jakości. Narzędzie do zmierzenia tego istnieje
(`run_retrieval_evals.py`), a `bge-m3` jest już pobrany lokalnie w Ollamie (1.16 GB),
więc porównanie da się zrobić bez pobierania wag.

- [x] **Rozszerzone o `"local"`** (`sentence-transformers`, in-process, CPU) — domyślne od Fazy 2
- [ ] A/B kandydatów metrykami z Fazy 1, nie „na oko":

| Model | Wymiary | Re-ingest? | Uwaga |
|---|---|---|---|
| `nomic-embed-text` (obecny, Ollama) | 768 | baseline | — |
| `intfloat/multilingual-e5-base` | 768 | nie — zgodne wymiary | jedyny bez migracji schematu |
| `BAAI/bge-m3` | 1024 | **tak** + migracja `vector(n)` | **już pobrany w Ollamie (1.16 GB)**, więc da się porównać szybciej niż zakładano |

- [x] **Czas embeddingu zmierzony spanami (Faza 4): 84 ms na rozgrzanym procesie**, 7.7 s przy zimnym starcie (wczytanie wag). Cel <1 s spełniony z zapasem
- [ ] Migracja `docker/init.sql` + skrypt migracyjny, jeśli wygra `bge-m3`
- [ ] Opcjonalnie Cloudflare BGE jako `embedding_provider="cloudflare"` — zapas przy
      ograniczonej pamięci targetu

**Gate:** pełne zapytanie end-to-end przy **zatrzymanej Ollamie**, `recall@5` nie gorszy
od baseline o więcej niż 5 pp.

---

### Pytanie, które otworzył model referencyjny: czy runtime musi być darmowy

Przebieg referencyjny kosztował **0.029 USD za 56 pytań**, czyli około **0.0005 USD
za zapytanie**, i był przy tym **dziesięciokrotnie szybszy** od darmowego modelu
(1.8 s wobec 19.1 s) oraz lepszy w cytowaniach (precyzja 0.929 wobec 0.800).

Przy profilu użycia „mama pyta kilka razy w tygodniu" — powiedzmy 100 zapytań
miesięcznie — daje to **0.05 USD miesięcznie**. Pierwotne ograniczenie brzmiało
„bez karty i za darmo"; karta jest już podpięta do OpenRoutera, a saldo wystarcza
na kilkanaście lat takiego użycia.

**Decyzja (2026-07-26), oddana asystentowi do rozstrzygnięcia:**

1. **Domyślna konfiguracja repo zostaje darmowa** (`OPENROUTER_CHAT_MODEL` =
   `nvidia/nemotron-nano-9b-v2:free`). Repo ma być publiczne — ustawienie domyślne,
   które wydaje pieniądze każdego, kto je sklonuje, jest złe niezależnie od kwoty.
2. **We wdrożeniu dla użytkownika końcowego płatny model jest pierwszy w łańcuchu,
   darmowy zapasowy.** 1.8 s wobec 19 s to różnica między „działa" a „chyba się
   zawiesiło", a 0.05 USD miesięcznie nie jest kosztem. Cel brzmiał „mama korzysta
   bez frustracji", nie „nigdy ani grosza" — a wyczerpanie salda przy takim układzie
   nie psuje systemu, tylko go spowalnia.
3. Realizacja: łańcuch fallbacku w Fazie 3. Do tego czasu nic się nie zmienia.

Konsekwencja dla Fazy 5: streaming i cache przestają być sposobem na maskowanie
19 sekund, a stają się zwykłą optymalizacją. To zdejmuje presję z tamtej fazy.

---

## Faza 3 — Model generacji i odporność providerów

Uzasadnienie fallbacku jest empiryczne: w jednej sesji testowej wystąpiły **trzy różne klasy
awarii** OpenRoutera (404 wycofany model, 400 zły slug, 429 przeciążenie upstream w 3/3 próbach
z backoffem). System używany bez nadzoru autora nie może zależeć od jednego darmowego endpointu.

- [ ] Cloudflare Workers AI jako `chat_provider="cloudflare"` (10k neuronów/dobę, reset 00:00 UTC,
      brak rotacji modeli — wagi na GPU Cloudflare)
- [ ] Benchmark przez `compare_models.py`: kandydaci Cloudflare vs `nemotron-nano-9b-v2:free`
- [x] **Łańcuch fallbacku — ZROBIONY** (2026-07-27). `CHAT_FALLBACK_CHAIN` w formacie
      `provider:model,provider:model`; pierwszym ogniwem jest zawsze `CHAT_PROVIDER`,
      więc konfiguracja bez łańcucha zachowuje się jak przedtem. Domyślnie **puste**,
      bo repo jest publiczne — łańcuch z płatnym modelem wydawałby pieniądze każdego,
      kto je sklonuje.

      Klasyfikacja awarii w `generation/fallback.py`: `400`/`404` jako deterministyczne
      (ten sam request da ten sam błąd, więc bez retry), `429`/`5xx`/timeout jako
      przejściowe, `401`/`403` jako konfiguracja. Przejście dalej następuje we wszystkich
      przypadkach — także przy **pustej odpowiedzi**, bo nemotron zwracał ją w 2 z 21
      pytań (`run_015`), a oddanie pustki użytkownikowi przy skonfigurowanym modelu
      zapasowym marnowałoby posiadaną odporność.

      **Odmowa NIE jest traktowana jak awaria.** Przełączanie w jej wyniku szukałoby
      modelu skłonnego halucynować, czyli odwrotnie do zasady, że halucynacja jest
      gorsza niż odmowa.

      Bezpiecznik: N porażek z rzędu odcina ogniwo na `CHAT_BREAKER_COOLDOWN_S`, sukces
      zeruje licznik, po cooldownie ogniwo dostaje czystą kartę. Trzyma stan, więc
      `RAGGenerator` jest tworzony **raz na proces** w lifespanie aplikacji — wcześniej
      router tworzył go per request, co zerowałoby licznik i bezpiecznik nie chroniłby
      przed niczym.

      **Gate osiągnięty na realnym providerze:** pierwsze ogniwo z nieistniejącym slugiem
      zwróciło `400`, nastąpiło jedno przełączenie, odpowiedź przyszła z drugiego ogniwa
      z dwoma poprawnymi cytowaniami (`ec_561_2006 | Art. 6(1)`). 33 testy jednostkowe
      pokrywają klasyfikację, budowę łańcucha, bezpiecznik i samą pętlę przełączania
      na atrapach — ścieżki awaryjne nie występują w normalnym przebiegu, więc muszą
      być wymuszone testem, a nie wypatrywane w produkcji.
- [x] **Puste odpowiedzi — NAPRAWIONE** przez `LLM_SYSTEM_PREFIX=/no_think` (zmierzone 2026-07-28, tabela w tej fazie). Historia diagnozy: `nemotron-nano-9b-v2:free`
      jest modelem rozumującym: w zmierzonym wywołaniu **455 z 621 tokenów wyjścia poszło
      na reasoning**. Przy `LLM_MAX_TOKENS=1024` długi łańcuch rozumowania zjada cały budżet
      i zwraca pustą treść — to wyjaśnia „pustą odpowiedź mimo 37 s", odnotowaną jeszcze
      w pierwotnym opisie kategorii `cross_document`. Doraźnie: generator nie przepuszcza
      już pustki do użytkownika, tylko zwraca komunikat po polsku i `has_answer=False`.
      Do zrobienia: podnieść `LLM_MAX_TOKENS`, rozważyć ograniczenie reasoningu parametrem
      OpenRoutera, i **zmierzyć** — to zmiana generacji, więc wymaga przebiegu przed/po
- [ ] Adresować porażki generacji z baseline: brak cytowania przy poprawnej treści
      (`numeric_fact`) i cytowanie niewłaściwego aktu przy właściwym kontekście (`scope`).
      To są problemy promptu i modelu, nie retrievalu — wiemy to teraz z pomiaru

**Gate:** przy sztucznie zepsutym pierwszym providerze zapytanie kończy się poprawną odpowiedzią
z drugiego. `answer_score` ≥ baseline, `failure_stage=generation` maleje.

---

## Faza 4 — Observability

- [x] **OpenTelemetry, spany per etap** (2026-07-27): `query` → `retrieve` →
      `embed_query` / `dense_search` / `bm25_search` / `rrf_fusion` / `rerank`,
      oraz `generate` → `llm_call` (span per PRÓBĘ, nie per zapytanie — przy fallbacku
      ślad pokazuje, ile czasu zjadło ogniwo, które i tak zawiodło).
      Eksporter konfigurowalny: `none` (domyślny — spany powstają, nigdzie nie lecą),
      `console`, `otlp`. Brak kolektora nie może psuć aplikacji dla jednego użytkownika.
- [x] **Metryki Prometheus na `/metrics`** (2026-07-27): histogram
      `tsl_rag_stage_duration_seconds{stage}`, liczniki
      `tsl_rag_provider_errors_total{provider,model,kind}`,
      `tsl_rag_fallback_switches_total{from_target,to_target}`,
      `tsl_rag_answers_total{outcome}` (answered / refused / all_providers_failed).
      Kubełki histogramu dobrane pod ZMIERZONE latencje (do 60 s), nie domyślne do 10 s —
      te wrzucałyby każdą generację na darmowym modelu do `+Inf`, czyli 15 s i 60 s
      byłyby nierozróżnialne. Żadna etykieta nie zawiera treści zapytania ani `chunk_id`.
- [x] **Logi strukturalne z `trace_id`** (2026-07-27), `LOG_JSON=true` → jedna linia
      JSON na stdout. `trace_id` wchodzi przez `patcher` loguru, nie ręcznie w każdym
      wywołaniu — inaczej jeden log bez identyfikatora zrywa ciągłość śladu dokładnie
      tam, gdzie coś poszło nie tak. Przy okazji `LOG_LEVEL` w końcu cokolwiek robi:
      istniał w `Settings`, ale nic go nie czytało.
- [x] `/health` i `/ready` — zrobione w Fazie 0
- [ ] Dashboard: latencja, error rate per provider, wykorzystanie darmowych limitów.
      **Nie w tym repo** — Grafana należy do klastra, nie do aplikacji (§7)

**Gate: OSIĄGNIĘTY.** Jedno zapytanie widoczne jako kompletny ślad, wszystkie etapy
w jednym `trace_id`, metryki obecne na `/metrics`.

### Co pokazał pierwszy ślad — embedding to 89% latencji retrievalu

Rozbicie na etapy było w tym projekcie niedostępne, więc te liczby są nowe:

| etap | zimny start | rozgrzane |
|---|---|---|
| `embed_query` | **7673 ms** | **84 ms** |
| `dense_search` | 52 ms | 7.6 ms |
| `bm25_search` | 236 ms | 2.0 ms |
| `rrf_fusion` | 0.0 ms | 0.0 ms |
| **całość** | **7962 ms** | **94 ms** |

Dwa wnioski:

1. Na rozgrzanym procesie **embedding zapytania to ~89% latencji retrievalu**.
   Deklarowane „retrieval trwa 0.1 s" jest prawdziwe, ale ta setna sekundy to
   praktycznie w całości jeden model na CPU.
2. **`warmup()` nie rozgrzewał modelu embeddingów** — ładował indeks BM25 i reranker
   (wyłączony), a wagi e5 wczytywały się leniwie przy pierwszym zapytaniu użytkownika.
   Docstring obiecywał, że pierwsze pytanie nie płaci za wczytanie modeli, i pomijał
   dokładnie ten koszt, który dominuje. Naprawione: warmup liczy jeden wektor.
   Ma to znaczenie dla Fazy 5, bo cold start na HF Spaces trafiał dotąd w pierwsze
   pytanie użytkownika ośmioma sekundami.

---

## Faza 5 — Deployment dla użytkownika końcowego

- [x] **`docker/Dockerfile` produkcyjny** (2026-07-28): multi-stage, `uv sync --no-dev`,
      użytkownik bez roota (uid 1001), `HEALTHCHECK` na `/health` (nie `/ready` — ten
      odpytuje bazę, więc jako healthcheck Dockera restartowałby kontener przy chwilowo
      niedostępnej bazie; readiness należy do orkiestratora). Wagi modelu embeddingów
      **nie są wbudowane w obraz** — pobierają się do wolumenu `HF_HOME`, bo obraz z wagami
      to ~2 GB do wypchnięcia przy zmianie jednej linijki kodu.

      **Zależności ingestu wydzielone do extra `ingest`.** `unstructured[pdf]` wciągał
      do obrazu API `torchvision`, `opencv` i `spacy` — stos wizyjny, którego API nie
      uruchamia, bo ingest jest ręczny i odpalany lokalnie. Wymagało to przeniesienia
      `DOCUMENT_REGISTRY` z `ingestion/cli.py` do `core/documents.py`, bo `/query/documents`
      importował go stamtąd i ciągnął cały moduł ingestu razem z parserami PDF.
      Ingest wymaga teraz `uv sync --extra ingest`.

      **`torch` przypięty do wariantu CPU na Linuksie.** Domyślne koło z PyPI to build CUDA:
      w obrazie dawało 2725 MB pakietów `nvidia/*` i 691 MB `triton`, przy `torch.cuda
      .is_available() == False`. Musi być zadeklarowany **wprost** w `dependencies` —
      `[tool.uv.sources]` działa tylko na zależności bezpośrednie, a torch wchodzi
      tranzytywnie przez `sentence-transformers`, więc bez tego wpisu przypięcie jest
      po cichu ignorowane.

      Stan końcowy, zweryfikowany uruchomieniem kontenera: `/health` 200, `/ready` zielone
      na wszystkich trzech sprawdzeniach (postgres, retriever, embeddings), `/metrics` 200,
      `/query` bez hasła 401 z komunikatem po polsku, proces jako uid 1001, logi JSON.
      Obraz 2.59 GB wg `docker images`, `site-packages` 1805 MB, zero pakietów
      `nvidia/*`, `triton`, `torchvision`, `opencv`, `spacy`, `unstructured`.

      **Pułapka pomiarowa:** `docker image inspect .Size` na Docker Desktop pokazuje
      rozmiar SKOMPRESOWANY (530 MB) i nie zgadza się z kolumną SIZE w `docker images`
      (2.59 GB). Przy raportowaniu trzymaj się jednego źródła.

      **Weryfikuj URUCHOMIENIEM, nie buildem.** Obraz budował się poprawnie i wywracał
      przy starcie na `RuntimeError: operator torchvision::nms does not exist` —
      `torchvision` z PyPI był skompilowany pod inny build torcha. Sam `docker build`
      tego nie wykrywa.
- [ ] Streaming odpowiedzi albo informacja o postępie — 19 s bez sygnału zwrotnego
      wygląda jak zawieszenie
- [ ] Cache odpowiedzi na powtarzające się pytania (oszczędza limity i skraca latencję)
- [x] **Podstawowa autoryzacja** (2026-07-28): `API_PASSWORD` + nagłówek `X-API-Key`
      na `/query`. Puste = wyłączona (uruchomienie lokalne, testy). Porównanie przez
      `secrets.compare_digest`, nie `==` — zwykłe porównanie kończy się na pierwszym
      różniącym się bajcie, więc czas odpowiedzi zdradza, ile znaków się zgadza.
      `/health`, `/ready` i `/metrics` zostają otwarte: probe'y i scraper nie mogą
      zależeć od sekretu aplikacji. UI wysyła ten sam sekret ze zmiennej `API_PASSWORD`
      i tłumaczy 401 na komunikat po polsku.
- [x] **Deployment: Streamlit Community Cloud + Neon** (2026-07-28, przygotowany).
      **HF Spaces odpadło**: wariant z Dockerem jest tam płatny (sprawdzone przez
      właściciela repo na żywym ekranie), a Space i tak nie utrzymałby bazy — jeden
      kontener z ulotnym dyskiem, więc Postgres musiałby stać osobno tak czy inaczej.

      Streamlit Cloud uruchamia JEDEN proces, więc UI nie ma dokąd wysłać żądania HTTP.
      Stąd `UI_MODE=inprocess` i wspólna warstwa `tsl_rag.service.answer_query`, którą
      wołają teraz oba wejścia — router API i UI. Bez tej warstwy `ui.py` musiałby mieć
      własną kopię logiki zapytania, która rozjechałaby się przy pierwszej zmianie,
      i to cicho, bo obie „działają". Rozdział API/UI zostaje w mocy: `docker/Dockerfile`
      z osobnym FastAPI jest dalej w repo i to on idzie do K8s.

      `requirements.txt` wygenerowany z `uv.lock` — Streamlit Cloud nie czyta ani
      locka, ani `pyproject.toml`, więc po zmianie zależności trzeba go wygenerować
      od nowa, inaczej wdrożenie zostaje w tyle bez ostrzeżenia. Wariant CPU torcha
      przypięty jawnie przez `--extra-index-url`.

      **Zweryfikowane uruchomieniem** trybu in-process na lokalnej bazie: poprawna
      treść, cytowanie, 5 chunków w kontekście, 3.7 s, drugie pytanie w tym samym
      procesie też przechodzi (pętla zdarzeń i pool asyncpg przeżywają kolejne
      wywołania — `asyncio.run()` per pytanie zamykałby pool razem z pętlą).

      Instrukcja krok po kroku: `docs/DEPLOY.md`. Zostaje do wykonania po stronie
      właściciela repo: konto Neon, ingest do Neona, push na GitHub, konfiguracja
      sekretów na Streamlit Cloud.

- [ ] ~~HF Spaces~~ — odrzucone, patrz wyżej
- [x] **Instrukcja dla użytkownika końcowego** — `docs/INSTRUKCJA.md`, po polsku,
      bez żargonu: jak pytać, jak czytać cytowania, dlaczego odmowa jest poprawnym
      zachowaniem, i tabela „co widzisz → co zrobić" dla komunikatów błędów

**Gate:** osoba nietechniczna otwiera URL i uzyskuje poprawną odpowiedź z cytowaniem,
bez obecności autora. **Po tej fazie cel podstawowy jest spełniony** — niezależnie od Fazy 6.

---

## Faza 6 — Kubernetes / LLMOps (osobny projekt, odległy, nierozpoczęty)

TSL-RAG będzie w tym klastrze **jednym z tenantów**, a nie jego tematem. To nie jest projekt
„TSL-RAG na k8s" — to klaster, w którym ten system jest jedną z uruchomionych aplikacji.

Nie zaczynaj tego w tym repo i nie projektuj niczego „pod multi-tenancy": namespace'y, quoty
i polityki sieciowe są sprawą klastra. Jedyne, co ma z tego wynikać dla aplikacji, to żeby
była **dobrze zachowującym się tenantem** — konfiguracja ze zmiennych środowiskowych, stan
wyłącznie w bazie, uczciwe probe'y, logi na stdout. To i tak jest dobra praktyka niezależnie
od k8s, więc nic tu nie robimy „na zapas".

Lista poniżej należy do tamtego projektu, nie do tego:

- [ ] Manifesty: `Deployment` (API), `StatefulSet` (Postgres+pgvector), `Service`, `Ingress`
- [ ] cert-manager + TLS, HPA na API
- [ ] ArgoCD GitOps
- [ ] **Eval-gated promotion:** `Job` uruchamia eval; `exit != 0` poniżej progu blokuje promocję.
      Wymaga wcześniej: rozszerzonego datasetu (Faza 1) i `run_retrieval_evals.py`, który
      działa bez klucza providera
- [ ] Progi w wersjonowanym `evals/thresholds.yaml`; CI odrzuca commit obniżający próg
      bez jawnego override (zasada #1 w `CLAUDE.md`)
- [ ] Argo Rollouts — canary po metrykach z Fazy 4
- [ ] Chaos engineering: ubicie poda Postgresa, symulacja awarii providera

---

## Ryzyka

| Ryzyko | Prawdopodobieństwo | Mitygacja |
|---|---|---|
| Kolejne darmowe modele znikają lub są przeciążone | **Wysokie** — wystąpiło 3× | Łańcuch fallbacku (Faza 3), min. 2 niezależne platformy |
| 15 pytań to za wąska podstawa do bramkowania | **Pewne** | Rozszerzenie datasetu w Fazie 1 przed użyciem progów w CI |
| `bge-m3` (~2.2 GB) nie mieści się w limitach pamięci targetu | Średnie | `multilingual-e5-base` jako lżejsza alternatywa, decyzja na danych z Fazy 2 |
| Latencja 19 s frustruje użytkownika | **Wysokie** — zmierzone | Streaming, cache, komunikat o postępie (Faza 5) |
| Korpus zawiera kolejne duplikaty lub niekompletne PDF-y | Średnie — jeden już znaleziony | `md5sum data/raw/*.pdf` przed dodaniem dokumentu; audyt kompletności w Fazie 1 |
| Warunki NVIDIA wykluczają obsługę użytkowników końcowych | Znane | NVIDIA tylko do ewaluacji; runtime na Cloudflare/OpenRouter — `docs/PROVIDERS.md` |
| Oracle Cloud Free Tier odbiera bezczynne instancje | Średnie `[WERYFIKUJ]` | Nie wiązać celu podstawowego z Fazą 6; HF Spaces jako ścieżka domyślna |

---

## Kolejność i zależności

```
Faza 0 (higiena) ✅
   └→ Faza 1 (metryki retrievalu) 🔶  ← narzędzie pomiarowe PRZED zmianami
        └→ Faza 2 (embedding — odblokowanie celu podstawowego)
             └→ Faza 3 (generacja + fallback)
                  └→ Faza 4 (observability)
                       ├→ Faza 5 (deployment)  ← cel podstawowy OSIĄGNIĘTY
                       └→ Faza 6 (Kubernetes — osobny projekt, później)
```

Najbliższy krok: domknąć Fazę 1 (`run_retrieval_evals.py` + rozszerzenie datasetu),
bo bez niej A/B embeddingów z Fazy 2 mierzyłoby jakość LLM-a zamiast jakości retrievalu.
