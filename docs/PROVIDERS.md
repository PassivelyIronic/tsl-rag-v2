# Providery inference — analiza pod TSL_RAG_reimplemented

Stan wiedzy: **lipiec 2026**. Limity darmowych tierów zmieniają się bez ostrzeżenia —
każdy wiersz oznaczony `[WERYFIKUJ]` sprawdź przed podjęciem decyzji, nie ufaj temu dokumentowi.

## 0. Kontekst decyzyjny

Dwa niezależne zapotrzebowania, często mylone:

| Zapotrzebowanie | Kiedy wywoływane | Wymagania |
|---|---|---|
| **Embedding** | ingest (raz) + **każde zapytanie użytkownika** (dense retrieval) | Niski koszt, niska latencja, stabilność wymiarów |
| **Generacja (chat)** | każde zapytanie | Jakość po polsku, długi kontekst, stabilna dostępność |

Kluczowe: embedding jest potrzebny **w runtime**, nie tylko przy ingest. Nie da się
"zembedować raz lokalnie i zapomnieć" — zapytanie mamy też musi zostać zembedowane.
To jest właściwy powód, dla którego `embedding_provider` musi mieć ścieżkę bez lokalnego GPU.

---

## 1. OpenRouter

**Status: obecnie zintegrowany, zweryfikowany empirycznie w tym projekcie.**

| Cecha | Wartość |
|---|---|
| Karta wymagana | Nie |
| Limit (bez doładowania) | 20 req/min, **50 req/dobę** — wspólne dla wszystkich modeli `:free` |
| Limit (po jednorazowym $10) | 1000 req/dobę, limit nie wygasa |
| API | OpenAI-compatible, `https://openrouter.ai/api/v1` |
| Embeddingi | Endpoint istnieje, ale **brak darmowych** — pass-through cena dostawcy |

### Zaobserwowane tryby awarii (dane z tego projektu, nie spekulacja)

1. **`404 — "This model is unavailable for free. The paid version is available now"`**
   Dotknęło `deepseek/deepseek-chat-v3.1:free` i `qwen/qwen3-235b-a22b:free`.
   Model został przeniesiony z darmowej puli do płatnej. Slug przestaje działać z dnia na dzień.

2. **`400 — "is not a valid model ID"`**
   `google/gemma-4-31b:free` — brakujący sufiks `-it`. Nazwa wyświetlana w katalogu
   ≠ slug API. Zawsze kopiuj slug z zakładki API konkretnego modelu.

3. **`429 — "Provider returned error"`**
   Dotknęło `meta-llama/llama-3.3-70b-instruct:free` i `google/gemma-4-31b-it:free`,
   3/3 próby z backoffem, o poranku. **To nie jest limit konta** — to przeciążenie
   upstream providera (OpenRouter routuje darmowe modele przez zewnętrzną infrastrukturę).
   Popularne modele są przeciążone niezależnie od Twojego limitu.

**Wniosek:** OpenRouter nadaje się do ewaluacji i jako *jeden z* providerów w łańcuchu
fallbacku. **Nie nadaje się jako jedyny provider** dla systemu, z którego ktoś ma korzystać
bez nadzoru — trzy różne klasy awarii w jednej sesji testowej to wystarczający dowód.

---

## 2. NVIDIA build.nvidia.com (NIM)

| Cecha | Wartość |
|---|---|
| Karta wymagana | Nie (wystarczy NVIDIA Developer Program) |
| API | OpenAI-compatible, `https://integrate.api.nvidia.com/v1` `[WERYFIKUJ]` |
| Rate limit | ~40 RPM — liczba powtarzana przez community i personel NVIDIA na forach `[WERYFIKUJ]` |
| Model rozliczeń | **Sprzeczne źródła** — patrz niżej |
| Embeddingi | Tak, w katalogu (m.in. modele NV-Embed / Nemotron Embed) `[WERYFIKUJ]` |

### Sprzeczność w źródłach — nie rozstrzygaj z pamięci

- Forum NVIDIA (starsze): trial ograniczony do 5000 kredytów, 1000 przy rejestracji,
  reszta po podaniu firmowego maila.
- Nowsze relacje: limity kredytowe **zniesione**, obowiązuje wyłącznie rate limit ~40 RPM.
- Personel NVIDIA (cytowany, maj 2026): użycie trialowe *nie jest* kredytowe, tylko
  rate-limitowane, a limit zależy od modelu i bieżącego ruchu.

Twoje realne ograniczenie widać w dashboardzie konta. Sprawdź tam, nie w artykułach.

### Ograniczenie licencyjne — istotne dla architektury

Warunki NVIDIA definiują produkcję jako użycie inne niż development, testy, badania lub
ewaluacja — **w tym obsługę realnych użytkowników końcowych**. Produkcja wymaga licencji
NVIDIA AI Enterprise.

Konsekwencja praktyczna dla tego projektu:
- **Faza ewaluacji modeli (`compare_models.py`) — mieści się w "evaluation" bez zastrzeżeń.**
- **Deployment obsługujący mamę — formalnie jest "realnym użytkownikiem końcowym".**

To ten sam typ zastrzeżenia co darmowy tier Gemini w EOG: ryzyko egzekwowania przy
prywatnym, niekomercyjnym użyciu jest znikome, ale decyzja jest Twoja i lepiej ją podjąć
świadomie niż odkryć zapis później. **Rekomendacja: NVIDIA jako provider ewaluacyjny
(benchmark modeli), niekoniecznie jako runtime dla mamy.**

Uwaga: `nvidia/nemotron-nano-9b-v2:free` — najlepszy dotąd wynik w Twoim eval — to model
NVIDII serwowany **przez OpenRouter**. To osobna ścieżka niż build.nvidia.com i nie podlega
powyższemu zastrzeżeniu w ten sam sposób.

---

## 3. Cloudflare Workers AI

**Najciekawszy kandydat, bo jako jedyny pokrywa embedding + generację w jednym darmowym tierze.**

| Cecha | Wartość |
|---|---|
| Karta wymagana | Nie (Workers Free) |
| Limit | **10 000 neuronów/dobę**, reset 00:00 UTC, twardy cap na planie Free |
| Szacunek Cloudflare | ~1 300 odpowiedzi LLM **lub** ~12 500 embeddingów na dobę |
| Po przekroczeniu | $0.011 / 1000 neuronów (wymaga planu Paid) |
| API | REST + warstwa OpenAI-compatible `[WERYFIKUJ dokładny base URL i kształt endpointu embeddings]` |
| Modele | ~50-80 open-weight: Llama, Mistral, Qwen, Gemma, GPT-OSS, destylaty DeepSeek-R1, **BGE embeddings** |
| Warunki komercyjne | **`[WERYFIKUJ]`** — jedno ze źródeł opisuje je jako niejasno udokumentowane |

### Dlaczego to jest mocny kandydat

- Pula neuronów jest **wspólna** dla wszystkich modeli — ale embeddingi kosztują ułamek
  tego co generacja. Przy Twoim profilu użycia (rzadkie zapytania, ~10 dokumentów)
  10k neuronów/dobę to z dużym zapasem więcej niż potrzeba.
- Limit dobowy **resetuje się**, w przeciwieństwie do jednorazowej puli kredytów.
- Brak problemu "model zniknął z darmowej puli" — Cloudflare hostuje wagi na własnym GPU,
  nie routuje do zewnętrznych dostawców. To bezpośrednio adresuje tryb awarii #1 i #3 z OpenRoutera.

### Ryzyka

- Neurony to jednostka znormalizowana — duże modele wypalają pulę nieproporcjonalnie szybko.
  Wybór modelu ma wpływ nie tylko na jakość, ale na to, ile zapytań zmieści się w dobie.
- Zestaw modeli mniejszy niż na OpenRouterze; nie ma tam największych open-weightów.

---

## 4. Lokalny `sentence-transformers` na CPU (tylko embedding)

Nie jest to "provider", ale realna alternatywa dla embeddingów — i prawdopodobnie
**najlepsza opcja dla tego konkretnego projektu**.

| Cecha | Wartość |
|---|---|
| Koszt | $0, na zawsze |
| Rate limit | Brak |
| Ryzyko rotacji modeli | Brak — wagi masz u siebie |
| Zależność sieciowa | Brak (po pierwszym pobraniu wag) |
| GPU | Niepotrzebne — embedding jednego krótkiego zapytania na CPU to rząd 50-500 ms |

Kandydaci:

| Model | Wymiary | Rozmiar | Uwagi |
|---|---|---|---|
| `BAAI/bge-m3` | 1024 | ~2.2 GB | Silny multilingual, dobry dla PL; najcięższy |
| `intfloat/multilingual-e5-base` | 768 | ~1.1 GB | Kompromis; wymiary zgodne z obecną kolumną pgvector |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB | Najlżejszy, najsłabszy jakościowo |

**Kluczowa uwaga:** zmiana modelu embeddingów = zmiana wymiarów = **konieczny re-ingest
całego korpusu** i migracja kolumny `vector(n)` w pgvector. Przy 444 chunkach to kilka minut,
ale musi być świadomą decyzją, nie przypadkiem. `multilingual-e5-base` ma 768 wymiarów,
czyli tyle co obecny `nomic-embed-text` — jedyny kandydat niewymagający migracji schematu.

Ważne: **cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) już teraz działa lokalnie na CPU**
w tym projekcie i nie sprawia problemów. To dowód, że model tej klasy jest akceptowalny
na docelowym sprzęcie — argument za opcją lokalną.

---

## 5. Google Gemini

Odnotowane dla kompletności. Warunki Google wymagają płatnych usług przy udostępnianiu
klienta API użytkownikom w EOG — a Polska jest w EOG. Dotyczy lokalizacji użytkownika,
nie charakteru komercyjnego projektu.

Obecne wykorzystanie w projekcie (**Gemini jako LLM-as-a-judge w `evals/judge.py`**) to
narzędzie deweloperskie używane przez Ciebie, nie usługa udostępniana użytkownikowi —
inna kategoria niż runtime obsługujący mamę.

---

## 6. Rekomendacja

### Embedding: **lokalny `sentence-transformers` na CPU**

Uzasadnienie: to jedyna opcja, która nie ma rate limitu, nie może zniknąć z darmowej puli
i nie wymaga klucza API w runtime. Embedding zapytania jest wywoływany przy **każdym**
pytaniu — czyli jest to najbardziej wrażliwy na awarię punkt całego pipeline'u.
Uzależnianie go od zewnętrznego darmowego tieru, który — jak zmierzyliśmy — potrafi zwrócić
429 trzy razy pod rząd, jest złym kompromisem.

Cloudflare BGE jako *opcja zapasowa* w abstrakcji providera, gdyby target deployment miał
ograniczoną pamięć.

### Generacja: **łańcuch fallbacku, nie jeden provider**

Kolejność do ustalenia empirycznie przez `compare_models.py`, wstępnie:

1. Cloudflare Workers AI (stabilny hosting, dobowy reset, brak rotacji modeli)
2. OpenRouter (`nvidia/nemotron-nano-9b-v2:free` — jedyny zweryfikowany działający wynik: answer 0.70, citation 0.733)
3. Ollama lokalnie (tylko na Twojej maszynie, do developmentu i jako ostatnia deska ratunku)

Zaobserwowane trzy klasy awarii OpenRoutera w jednej sesji to wystarczające uzasadnienie,
by fallback był wymaganiem funkcjonalnym, a nie "nice to have".
