from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # zapytania liczy się przy każdym pytaniu (docs/PROVIDERS.md).
    embedding_provider: Literal["ollama", "openai", "local"] = "ollama"
    chat_provider: Literal["ollama", "openai", "openrouter", "gemini"] = "ollama"

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
    # Dwa zastosowania, oba WYŁĄCZNIE deweloperskie (docs/PROVIDERS.md §5):
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

    # LLM params
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

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
