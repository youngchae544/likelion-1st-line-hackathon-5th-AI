import pytest


@pytest.fixture(autouse=True)
def isolated_photo_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_CACHE_PATH", str(tmp_path / "photo.sqlite3"))
