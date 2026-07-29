from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Nazwany alias, bo ta sama lista providerów jest potrzebna w llm_client
# do budowy łańcucha fallbacku. Powielony Literal rozjechałby się przy
# dodaniu providera i rozjazd wyszedłby dopiero w runtime.
ChatProvider = Literal["ollama", "openai", "openrouter", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # extra="forbid" jest domyślne, ale zapisane jawnie, bo to źródło
        # częstego błędu: zmienna w .env bez odpowiednika tutaj wywala start
        # komunikatem "Extra inputs are not permitted". Dodając pole tutaj,
        # dodaj je też do env.example (CLAUDE.md §9).
        extra="forbid",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Provider switches ---
    # Rozdzielone celowo: embedding i chat mogą iść przez różnych providerów
    # jednocześnie, np. embeddingi lokalnie przez Ollamę (retrieval bez zmian),
    # a generacja przez darmowy model na OpenRouter (brak lokalnego GPU).
    # "local" = sentence-transformers w tym samym procesie, na CPU. Jedyny
    # wariant bez zależności sieciowej i bez rate limitu w runtime — embedding
    # zapytania liczy się przy każdym pytaniu.
    embedding_provider: Literal["ollama", "openai", "local"] = "ollama"
    chat_provider: ChatProvider = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_dimensions: int = 768

    # OpenAI (opcjonalne)
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072

    # OpenRouter (opcjonalne, wyłącznie chat/generacja — free tier)
    # Klucz: https://openrouter.ai/keys
    # Darmowe modele (":free") — lista zmienia się bez ostrzeżenia, sprawdź:
    # https://openrouter.ai/models?max_price=0
    openrouter_api_key: SecretStr | None = None
    # Domyślny slug = jedyny model zweryfikowany empirycznie w tym projekcie
    # (patrz evals/results/model_comparison.json). Poprzedni domyślny
    # llama-3.3-70b-instruct:free zwracał 429 w 3/3 próbach z backoffem.
    # Slug zmieniaj tylko po sprawdzeniu w zakładce API modelu (CLAUDE.md §5.3).
    openrouter_chat_model: str = "nvidia/nemotron-nano-9b-v2:free"

    # --- Lokalne embeddingi (EMBEDDING_PROVIDER=local) ---
    # Prefiksy są JAWNE, nie zgadywane z nazwy modelu. Modele z rodziny E5
    # wymagają "query: " i "passage: ", a ich pominięcie nie powoduje błędu —
    # tylko obniża jakość, niewidocznie. bge-m3 prefiksów nie używa, więc przy
    # porównywaniu modeli trzeba je wyzerować, inaczej porównuje się model
    # z prefiksami do modelu bez nich.
    local_embed_model: str = "intfloat/multilingual-e5-base"
    local_embed_dimensions: int = 768
    local_embed_query_prefix: str = "query: "
    local_embed_passage_prefix: str = "passage: "
    local_embed_batch_size: int = 16

    # --- Gemini ---
    # Dwa zastosowania, oba WYŁĄCZNIE deweloperskie:
    #   1. LLM-as-a-judge w evals/judge.py
    #   2. chat_provider="gemini" jako MODEL REFERENCYJNY — pomiar sufitu jakości
    #      przy obecnym retrievalu, nie runtime dla użytkownika końcowego
    # Warunki Google wymagają płatnych usług przy udostępnianiu klienta API
    # użytkownikom w EOG, więc jako runtime dla mamy to się nie kwalifikuje.
    gemini_api_key: SecretStr | None = None
    gemini_chat_model: str = "gemini-2.0-flash"

    @property
    def embedding_dimensions(self) -> int:
        if self.embedding_provider == "openai":
            return self.openai_embedding_dimensions
        if self.embedding_provider == "local":
            return self.local_embed_dimensions
        return self.ollama_embed_dimensions

    @property
    def active_embedding_model(self) -> str:
        """Nazwa modelu embeddingów dla aktywnego providera — do snapshotów i logów."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        if self.embedding_provider == "local":
            return self.local_embed_model
        return self.ollama_embed_model

    @property
    def active_llm_model(self) -> str:
        if self.chat_provider == "openai":
            return self.openai_chat_model
        if self.chat_provider == "openrouter":
            return self.openrouter_chat_model
        if self.chat_provider == "gemini":
            return self.gemini_chat_model
        return self.ollama_llm_model

    # --- Łańcuch fallbacku generacji ---
    # Format: "provider:model,provider:model". Puste = brak fallbacku, czyli
    # zachowanie sprzed wprowadzenia łańcucha. Pierwszym ogniwem jest ZAWSZE
    # chat_provider + jego model, więc tu wpisuje się wyłącznie zapasowe.
    #
    # Po co: w jednej sesji testowej wystąpiły trzy różne klasy awarii
    # OpenRoutera (404 wycofany model, 400 zły slug, 429 przeciążenie upstream
    # w 3/3 próbach z backoffem). System używany bez nadzoru autora nie może
    # zależeć od jednego darmowego endpointu.
    #
    # Domyślnie PUSTE także dlatego, że repo jest publiczne: łańcuch wskazujący
    # płatny model wydawałby pieniądze każdego, kto sklonuje repo (PLAN.md,
    # decyzja o darmowej konfiguracji domyślnej).
    chat_fallback_chain: str = ""

    # Bezpiecznik: po tylu porażkach z rzędu ogniwo jest pomijane przez
    # chat_breaker_cooldown_s sekund. Bez tego każde zapytanie płaci pełny
    # timeout na providerze, o którym już wiadomo, że nie odpowiada.
    chat_breaker_failures: int = 3
    chat_breaker_cooldown_s: float = 60.0

    # LLM params
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Sterowanie rozumowaniem modeli typu "reasoning". Puste = nie wysyłamy
    # parametru wcale (zachowanie domyślne providera).
    #
    # Wysyłane jako `reasoning: {"effort": ...}` przez OpenRouter i jako
    # `reasoning_effort` przez pozostałych. Nie każdy model to obsługuje —
    # Gemma odpowiada wtedy 400 "Thinking budget is not supported".
    #
    # UWAGA, zmierzone 2026-07-27: na nvidia/nemotron-nano-9b-v2:free ten
    # parametr NIE DZIAŁA, mimo że model deklaruje `reasoning`
    # w `supported_parameters` OpenRoutera. Tokeny rozumowania bez parametru:
    # 381 i 298; przy `effort=none`: 436 i 453; przy `reasoning.enabled=false`:
    # 345. Czyli żadnej redukcji. Na tym modelu dźwignią jest
    # llm_system_prefix poniżej, nie to pole.
    llm_reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high"] = ""

    # Tekst wstawiany PRZED system prompt generatora. Puste = nic nie doklejamy.
    #
    # Istnieje, bo część modeli rozumujących sterowana jest tokenem w promptcie,
    # a nie parametrem API. Zmierzone na nvidia/nemotron-nano-9b-v2:free
    # (jedno pytanie, ten sam kontekst 5 chunków): `/no_think` zbija tokeny
    # rozumowania z ~350-450 do 0, a całe wyjście z ~446 tokenów do 86.
    # To usuwa jednocześnie źródło pustych odpowiedzi — przy zerowym reasoningu
    # łańcuch rozumowania nie ma jak wyczerpać llm_max_tokens.
    #
    # Wartość jest JAWNĄ KONFIGURACJĄ, nie jest zgadywana z nazwy modelu —
    # tak samo jak prefiksy E5 przy embeddingach (CLAUDE.md §3). `/no_think`
    # wysłane do modelu, który go nie zna, zostaje w promptcie jako śmieć.
    llm_system_prefix: str = ""

    # --- Cache odpowiedzi ---
    # Powtórzone pytanie nie ma powodu zjadać dziennego limitu providera ani
    # czekać kilku sekund na generację. Klucz zawiera konfigurację, więc zmiana
    # modelu albo stałej RRF unieważnia wpisy automatycznie.
    answer_cache_enabled: bool = True
    answer_cache_max_entries: int = 128
    answer_cache_ttl_s: float = 86_400.0

    # --- Dostęp do API ---
    # Puste = autoryzacja WYŁĄCZONA (uruchomienie lokalne, testy). Ustawione =
    # /query wymaga nagłówka X-API-Key. Publiczny URL bez hasła to zaproszenie
    # do wypalenia dziennych limitów providera przez pierwszego bota, który go
    # znajdzie — a limity są dzienne, nie kwotowe, więc nie widać tego w rachunku.
    api_password: SecretStr | None = None

    # --- Observability ---
    # Domyślnie "none": spany powstają, ale nigdzie nie lecą. Aplikacja dla
    # jednego użytkownika nie może wymagać działającego Jaegera do startu,
    # a instrumentacja ma być w kodzie od początku, nie doklejana później
    # (PLAN.md Faza 4). "console" służy do podejrzenia śladu bez kolektora.
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "tsl-rag"

    # Logi jako JSON — wymagane, gdy logi zbiera agregator (stdout w K8s).
    # W developmencie czytelniejszy jest format domyślny loguru.
    log_json: bool = False

    postgres_dsn: PostgresDsn

    # --- Retrieval ---
    # Wszystkie te wartości są faktycznie odczytywane w ścieżce zapytania
    # (RetrievalRequest, _reciprocal_rank_fusion). Do commita 92ab634 były
    # martwe: kod miał własne hardkody, a strojenie przez .env nie robiło nic.
    retrieval_top_k: int = 20
    retrieval_rerank_top_n: int = 5
    # Wagi RRF. 0.5/0.5 = obie listy rankingowe równoważne, czyli dokładnie
    # to, co robił nieważony RRF przed podłączeniem tych pól. Zmiana proporcji
    # zmienia ranking, więc wymaga przebiegu evalu przed/po (CLAUDE.md §9).
    bm25_weight: float = 0.5
    dense_weight: float = 0.5

    # Stała k w RRF: score(d) = Σ waga / (k + rank). Rozstrzyga spór między
    # ZGODNOŚCIĄ obu list a MOCNYM DOWODEM z jednej.
    #
    # Było 60 (zaszyte w kodzie, nigdy nie mierzone) — wartość z oryginalnej
    # pracy o RRF, gdzie łączono wiele przebiegów na korpusie skali TREC.
    # Tutaj łączymy DWIE listy po 20 pozycji, a przy k=60 ranga 1 dostaje 1/61,
    # ranga 20 — 1/80. Stosunek 1.31 oznacza, że informacja o pozycji jest
    # praktycznie wyrzucana i zostaje sama zgodność list. Skutek zmierzony:
    # chunk z pozycji 3 w jednej liście, nieobecny w drugiej, lądował na 10 —
    # poniżej chunków, które obie listy oceniły przeciętnie.
    #
    # Zmierzone na 48 pytaniach (przegląd k ∈ {60,30,20,10,5,2,1,0}):
    # k=60 → recall@5 0.938, fakty@5 0.840;  k=5 → 0.958 i 0.882.
    # Żadna kategoria nie traci; scope idzie z 0.875/0.625 na 1.000/0.750.
    # Wybrano 5, a nie 2 (recall@5 0.969), bo różnica to jedno pytanie z 48,
    # a k ∈ [1,5] jest płaskowyżem — wybieranie szczytu byłoby strojeniem
    # pod zbiór testowy.
    rrf_k: int = 5

    # --- Reranker ---
    # reranker_max_length: ile tokenów pary (zapytanie, chunk) widzi cross-encoder.
    # Chunki taryfikatorów to duże tabele, więc przy 512 tokenach właściwy wiersz
    # bywa poza obcięciem i model ocenia fragment, w którym odpowiedzi nie ma.
    # ms-marco-MiniLM i bge-reranker-base obsługują 512; bge-reranker-v2-m3 do 8192.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 2048
    # WYŁĄCZONY DOMYŚLNIE na podstawie pomiaru, nie dla uproszczenia
    # (tabela pięciu wariantów w PLAN.md Faza 1):
    #   bez rerankingu            recall@5 0.938, mediana 0.1 s
    #   ms-marco-MiniLM (512)     recall@5 0.854, mediana 1.5 s
    #   bge-reranker-v2-m3 (2048) recall@5 0.969, mediana 43.4 s
    # Retrieval bez rerankingu trwa 0.1 s, więc ten etap to praktycznie całość
    # kosztu. Najlepszy wariant kupuje +0.031 recall@5 za 434-krotny wzrost
    # latencji — nie do pogodzenia z celem, w którym odpowiedź ma przyjść
    # w rozsądnym czasie na słabym sprzęcie.
    #
    # Model i okno zostawione na wartościach NAJLEPSZEGO zmierzonego wariantu,
    # żeby włączenie tego z powrotem (np. gdy dojdzie cache odpowiedzi z Fazy 5)
    # było jedną zmianą, a nie odtwarzaniem konfiguracji z commit message.
    reranker_enabled: bool = False

    # --- Generation ---
    # Limit kontekstu liczony w ZNAKACH, nie tokenach — generator przycina
    # kontekst po długości stringa. Poprzednie max_context_tokens=4096 nie
    # było nigdzie używane i sugerowało limit tokenowy, którego nie ma.
    max_context_chars: int = 12_000

    # --- Ingestion ---
    # Parametry chunkera. Wartości odpowiadają temu, co LegalChunker robił
    # dotąd na swoich stałych modułowych. Poprzednie chunk_size=400 /
    # chunk_overlap=50 były nie tylko martwe, ale też niezgodne z faktycznym
    # zachowaniem (450/60) — strojenie ich wprowadzało w błąd.
    # Zmiana tych wartości wymaga ponownego ingestu całego korpusu.
    chunker_max_tokens: int = 450
    chunker_min_tokens: int = 60
    chunker_overlap_tokens: int = 60
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    @model_validator(mode="after")
    def validate_weights(self) -> "Settings":
        total = round(self.bm25_weight + self.dense_weight, 6)
        if total != 1.0:
            raise ValueError(f"bm25_weight + dense_weight musi = 1.0, got {total}")
        return self

    @model_validator(mode="after")
    def validate_system_prefix(self) -> "Settings":
        """
        Odrzuca prefiks, który wygląda na ścieżkę systemu plików.

        Git Bash na Windowsie stosuje konwersję ścieżek MSYS: argument
        zaczynający się od `/` jest tłumaczony na ścieżkę Windows, więc
        `LLM_SYSTEM_PREFIX=/no_think uv run ...` dociera do procesu jako
        `C:/Program Files/Git/no_think`. Zdarzyło się to 2026-07-27
        i unieważniło cały przebieg evalu: do system promptu poszła ścieżka,
        a wynik wyglądał jak pomiar `/no_think`. Cichy błąd tej klasy jest
        groźniejszy niż wyjątek, bo produkuje liczbę, której nikt nie kwestionuje.
        """
        prefix = self.llm_system_prefix
        if ":/" in prefix or ":\\" in prefix or "\\" in prefix:
            raise ValueError(
                f"LLM_SYSTEM_PREFIX wygląda na ścieżkę systemu plików: {prefix!r}. "
                "Prawdopodobnie Git Bash przetłumaczył wartość zaczynającą się od '/'. "
                "Ustaw ją w pliku .env (dotenv nie tłumaczy ścieżek) albo poprzedź "
                "komendę zmienną MSYS_NO_PATHCONV=1."
            )
        return self

    @model_validator(mode="after")
    def validate_openai_key(self) -> "Settings":
        needs_openai = self.chat_provider == "openai" or self.embedding_provider == "openai"
        if needs_openai and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY wymagany gdy CHAT_PROVIDER=openai lub EMBEDDING_PROVIDER=openai"
            )
        return self

    @model_validator(mode="after")
    def validate_openrouter_key(self) -> "Settings":
        if self.chat_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY wymagany gdy CHAT_PROVIDER=openrouter. "
                "Klucz: https://openrouter.ai/keys"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Wyciszenie poniżej: mypy nie wie, że pydantic-settings wypełnia pola
    # wymagane (postgres_dsn) ze zmiennych środowiskowych i .env, więc widzi
    # brakujący argument nazwany. Brak wartości w środowisku i tak skończy
    # się ValidationError w runtime, czyli błąd nie znika, tylko jest
    # raportowany tam, gdzie faktycznie występuje.
    return Settings()  # type: ignore[call-arg]
