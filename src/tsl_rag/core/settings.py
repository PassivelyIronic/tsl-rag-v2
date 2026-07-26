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
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    chat_provider: Literal["ollama", "openai", "openrouter"] = "ollama"

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

    # Gemini (opcjonalne, pod ewaluację — LLM-as-a-judge)
    gemini_api_key: SecretStr | None = None

    @property
    def embedding_dimensions(self) -> int:
        if self.embedding_provider == "openai":
            return self.openai_embedding_dimensions
        return self.ollama_embed_dimensions

    @property
    def active_llm_model(self) -> str:
        if self.chat_provider == "openai":
            return self.openai_chat_model
        if self.chat_provider == "openrouter":
            return self.openrouter_chat_model
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

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
