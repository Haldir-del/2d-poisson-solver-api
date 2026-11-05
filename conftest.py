import pytest


@pytest.fixture(autouse=True)
def _clear_rapidapi_secret(monkeypatch):
    """Keep local tests independent from any RAPIDAPI_PROXY_SECRET in the shell."""
    monkeypatch.delenv("RAPIDAPI_PROXY_SECRET", raising=False)
