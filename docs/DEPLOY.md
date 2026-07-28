# Wdrożenie: Streamlit Community Cloud + Neon

Ścieżka darmowa i bez karty. Trzy elementy:

```
GitHub (kod)  →  Streamlit Community Cloud (UI + retrieval + generacja, 1 proces)
                          │
                          ├─→ Neon      (Postgres + pgvector, 438 chunków)
                          └─→ OpenRouter (generacja odpowiedzi)
```

**Dlaczego nie Hugging Face Spaces:** wariant z Dockerem jest tam płatny (stan
2026-07-28), a Space i tak nie utrzymałby bazy — jeden kontener z ulotnym dyskiem.

**Dlaczego API działa w tym samym procesie co UI:** Streamlit Cloud uruchamia
jeden proces, więc nie ma dokąd wysłać żądania HTTP. Steruje tym `UI_MODE=inprocess`.
Rozdział API/UI z `PLAN.md` zostaje w mocy — obie ścieżki wołają tę samą funkcję
`tsl_rag.service.answer_query`, a `docker/Dockerfile` z osobnym FastAPI dalej jest
w repo i to on pójdzie do K8s.

---

## Krok 1 — baza na Neonie

1. Załóż konto na <https://neon.tech> (GitHub OAuth, bez karty).
2. Utwórz projekt, region **Europe (Frankfurt)** — najbliżej.
3. W zakładce **SQL Editor** wykonaj zawartość `docker/init.sql`.
   Zaczyna się od `CREATE EXTENSION IF NOT EXISTS vector;` — Neon ma pgvector
   dostępny, ale rozszerzenie trzeba włączyć jawnie.
4. Skopiuj connection string z **Connection Details**. Wygląda tak:

   ```
   postgresql://user:haslo@ep-cos-tam-123456.eu-central-1.aws.neon.tech/neondb?sslmode=require
   ```

5. Przerób go na format, którego oczekuje `POSTGRES_DSN` — wystarczy zmienić schemat
   na `postgresql+asyncpg://`:

   ```
   postgresql+asyncpg://user:haslo@ep-cos-tam-123456.eu-central-1.aws.neon.tech/neondb
   ```

   **Weź endpoint BEZ `-pooler` w nazwie hosta.** Neon podaje domyślnie adres puli
   (PgBouncer w trybie transakcyjnym), a `asyncpg` używa prepared statements, które
   się z tym trybem gryzą. Zwykłe zapytania przechodzą, więc problem ujawnia się
   nierównomiernie — przy ingeście i pod obciążeniem. Endpoint bezpośredni to ten
   sam adres z usuniętym członem `-pooler`.

   `?sslmode=require&channel_binding=require` **można zostawić** — sprawdzone
   2026-07-29, `asyncpg` przyjmuje oba parametry i łączy się poprawnie. (Wcześniejsza
   wersja tej instrukcji kazała je usuwać; to było twierdzenie z pamięci, nie z pomiaru.)

## Krok 2 — ingest korpusu do Neona

Z Twojego komputera, bo ingest potrzebuje parserów PDF, których wdrożenie nie ma:

```powershell
uv sync --extra ingest
$env:POSTGRES_DSN="postgresql+asyncpg://user:haslo@ep-....neon.tech/neondb"
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/
```

Oczekiwany wynik: **13 z 14 plików, `failed: 0`, 438 chunków**. Czternasty jest
pomijany celowo (duplikat, patrz `CLAUDE.md` §8).

Sprawdź w SQL Editorze Neona:

```sql
SELECT count(*) FROM document_chunks;                 -- 438
SELECT DISTINCT metadata->>'embedding_model' FROM document_chunks;
-- intfloat/multilingual-e5-base
```

Ten drugi zapytanie jest istotne: jeśli korpus zaindeksujesz innym modelem niż
ten, którego użyje wdrożenie, retrieval zwróci losowe wyniki. System wykrywa to
przy starcie i przerywa z komunikatem.

## Krok 3 — repozytorium na GitHubie

Streamlit Cloud wdraża wyłącznie z GitHuba, więc to już nie jest opcjonalne.

1. Utwórz **publiczne** repo `tsl-rag-v2` na koncie `PassivelyIronic`.
2. Z katalogu projektu:

   ```powershell
   git remote add origin https://github.com/PassivelyIronic/tsl-rag-v2.git
   git push -u origin main
   ```

**Zanim wypchniesz — sprawdź, czy `.env` nie wchodzi do repo:**

```powershell
git status --short         # .env NIE może się tu pojawić
git ls-files | Select-String "^\.env$"    # ma nic nie zwrócić
```

## Krok 4 — aplikacja na Streamlit Cloud

1. <https://share.streamlit.io> → zaloguj się GitHubem (`PassivelyIronic`).
2. **New app** → repozytorium `tsl-rag-v2`, branch `main`, main file **`ui.py`**.
3. **Advanced settings → Python version: 3.11.**
4. W **Secrets** wklej (to jest format TOML, bez `export`):

   ```toml
   UI_MODE = "inprocess"
   POSTGRES_DSN = "postgresql+asyncpg://user:haslo@ep-....neon.tech/neondb"

   EMBEDDING_PROVIDER = "local"
   CHAT_PROVIDER = "openrouter"
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_CHAT_MODEL = "nvidia/nemotron-nano-9b-v2:free"

   # Zmierzone: 5× szybciej i lepiej w każdej metryce treści (PLAN.md Faza 3).
   LLM_SYSTEM_PREFIX = "/no_think"

   # Fallback: gdy darmowy model padnie, pytanie i tak dostanie odpowiedź.
   # Kosztuje ~0.0005 USD za zapytanie i wchodzi dopiero, gdy darmowy zawiedzie.
   CHAT_FALLBACK_CHAIN = "openrouter:openai/gpt-4o-mini"
   ```

5. **Deploy**. Pierwsze uruchomienie trwa kilka minut — instalują się zależności
   i pobiera model embeddingów (~1.1 GB).

## Krok 5 — sprawdzenie

Zadaj pytanie: *„Ile godzin dziennie może jechać kierowca?"*

Poprawna odpowiedź zawiera **9 godzin** i cytowanie — `[ec_561_2006 | Art. 6(1)]`
albo `[aetr | Art. 6(1)]`. Oba akty zawierają ten sam limit, przy czym 561/2006 jest
właściwsze dla przewozów w UE, a AETR dla międzynarodowych poza nią. W teście
z 2026-07-28 system zacytował AETR i to jest znana słabość kategorii `scope`,
nie usterka wdrożenia.

**Odpowiedź bez żadnego cytowania traktuj jako porażkę** (`CLAUDE.md` §5.6),
nawet jeśli treść brzmi sensownie.

W panelu bocznym powinny świecić 🟢 przy PostgreSQL i embeddingach.

---

## Czego się spodziewać

| Rzecz | Jak jest |
|---|---|
| Pierwsze pytanie po przerwie | Wolne. Streamlit Cloud usypia aplikację po bezczynności, a po przebudzeniu dochodzi wczytanie modelu embeddingów — zmierzone lokalnie 7.7 s, na Cloudzie więcej |
| Kolejne pytania | ~5 s przy `/no_think` na darmowym modelu (zmierzone: mediana 4.0 s) |
| Limit darmowego modelu | ~50 zapytań na dobę, reset o północy UTC. Po wyczerpaniu wchodzi ogniwo zapasowe z łańcucha fallbacku |
| Pamięć | Model e5-base plus torch CPU to ok. 0.5-1 GB RSS. Streamlit Cloud daje ograniczony limit — jeśli aplikacja będzie się restartować, to jest pierwsze podejrzenie |

## Jeśli coś nie działa

| Objaw | Przyczyna | Co zrobić |
|---|---|---|
| `Extra inputs are not permitted` | Sekret bez odpowiednika w `settings.py` | Usuń nadmiarowy klucz z Secrets albo dodaj pole w `Settings` |
| `invalid dsn` przy starcie | Schemat `postgresql://` zamiast `postgresql+asyncpg://` | Popraw zgodnie z krokiem 1.5 |
| `prepared statement "__asyncpg_…" already exists` | DSN wskazuje endpoint `-pooler` | Użyj adresu bez `-pooler` |
| `Korpus zaindeksowano modelem embeddingów …` | Ingest zrobiony innym modelem niż `EMBEDDING_PROVIDER` wdrożenia | Powtórz ingest (krok 2) |
| Retrieval zwraca pustkę, brak błędu | Ingest poszedł do lokalnej bazy, nie do Neona | Sprawdź `count(*)` w Neonie |
| Aplikacja restartuje się w kółko | Przekroczony limit pamięci | Patrz tabela wyżej |
| `429` przy każdym pytaniu | Wyczerpany dzienny limit darmowego modelu | Poczekaj do północy UTC albo ustaw `CHAT_FALLBACK_CHAIN` |

Instrukcja dla osoby korzystającej z gotowego systemu: `docs/INSTRUKCJA.md`.
