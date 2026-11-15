import pytest


@pytest.fixture(autouse=True)
def isolated_photo_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-service-key-not-a-secret-12345678")
    monkeypatch.setenv("PHOTO_CACHE_PATH", str(tmp_path / "photo.sqlite3"))
