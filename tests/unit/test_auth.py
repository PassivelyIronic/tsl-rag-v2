import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from tsl_rag.api.auth import verify_api_key
from tsl_rag.core.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db", **overrides)


def test_disabled_when_no_password_configured():
    """Uruchomienie lokalne i testy nie mogą wymagać ustawiania sekretu."""
    verify_api_key(x_api_key=None, settings=_settings())


def test_disabled_ignores_provided_key():
    verify_api_key(x_api_key="cokolwiek", settings=_settings())


def test_correct_password_passes():
    settings = _settings(api_password=SecretStr("tajne"))
    verify_api_key(x_api_key="tajne", settings=settings)


def test_missing_header_is_rejected():
    settings = _settings(api_password=SecretStr("tajne"))
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key=None, settings=settings)
    assert exc.value.status_code == 401
    # Użytkownik jest nietechniczny — komunikat po polsku i mówi, co zrobić
    assert "X-API-Key" in exc.value.detail
    assert "administratora" in exc.value.detail


def test_wrong_password_is_rejected():
    settings = _settings(api_password=SecretStr("tajne"))
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="zle", settings=settings)
    assert exc.value.status_code == 401


def test_error_message_does_not_leak_expected_password():
    """
    Komunikat błędu nie może zawierać oczekiwanej wartości — trafia do klienta,
    a przy 401 klientem bywa ktoś, kto właśnie zgaduje.
    """
    settings = _settings(api_password=SecretStr("bardzo-tajne-haslo"))
    with pytest.raises(HTTPException) as exc:
        verify_api_key(x_api_key="proba", settings=settings)
    assert "bardzo-tajne-haslo" not in exc.value.detail


def test_password_is_not_exposed_by_repr():
    """
    SecretStr, nie str: powtórzony wyjątek pydantica albo zalogowany snapshot
    konfiguracji nie może wypisać hasła w czystej postaci.
    """
    settings = _settings(api_password=SecretStr("bardzo-tajne-haslo"))
    assert "bardzo-tajne-haslo" not in repr(settings)
    assert "bardzo-tajne-haslo" not in str(settings.api_password)
