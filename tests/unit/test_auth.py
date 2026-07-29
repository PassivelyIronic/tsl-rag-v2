import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from tsl_rag.api.auth import check_api_key

pytestmark = pytest.mark.unit


def test_disabled_when_no_password_configured():
    """Uruchomienie lokalne i testy nie mogą wymagać ustawiania sekretu."""
    check_api_key(None, None)


def test_disabled_ignores_provided_key():
    check_api_key("cokolwiek", None)


def test_correct_password_passes():
    check_api_key("tajne", SecretStr("tajne"))


def test_missing_header_is_rejected():
    with pytest.raises(HTTPException) as exc:
        check_api_key(None, SecretStr("tajne"))
    assert exc.value.status_code == 401
    # Użytkownik jest nietechniczny — komunikat po polsku i mówi, co zrobić
    assert "X-API-Key" in exc.value.detail
    assert "administratora" in exc.value.detail


def test_wrong_password_is_rejected():
    with pytest.raises(HTTPException) as exc:
        check_api_key("zle", SecretStr("tajne"))
    assert exc.value.status_code == 401


def test_error_message_does_not_leak_expected_password():
    """
    Komunikat błędu nie może zawierać oczekiwanej wartości — trafia do klienta,
    a przy 401 klientem bywa ktoś, kto właśnie zgaduje.
    """
    with pytest.raises(HTTPException) as exc:
        check_api_key("proba", SecretStr("bardzo-tajne-haslo"))
    assert "bardzo-tajne-haslo" not in exc.value.detail


def test_password_is_not_exposed_by_repr():
    """
    SecretStr, nie str: powtórzony wyjątek pydantica albo zalogowany snapshot
    konfiguracji nie może wypisać hasła w czystej postaci.
    """
    secret = SecretStr("bardzo-tajne-haslo")
    assert "bardzo-tajne-haslo" not in repr(secret)
    assert "bardzo-tajne-haslo" not in str(secret)
