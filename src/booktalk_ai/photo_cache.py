"""Stable photo results; persist explanations and source indices, never OCR text/images."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Callable

_LOCKS = [Lock() for _ in range(64)]


def cache_key(image: str, language: str) -> str:
    version = os.environ.get("PHOTO_CACHE_VERSION", "v1")
    return hashlib.sha256(json.dumps([version, image, language], ensure_ascii=False).encode()).hexdigest()


def get_or_create(key: str, create: Callable[[], dict]) -> dict:
    path = Path(os.environ.get("PHOTO_CACHE_PATH", "./data/photo-cache.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCKS[int(key[:8], 16) % len(_LOCKS)]:
        with sqlite3.connect(path, timeout=5) as db:
            db.execute("CREATE TABLE IF NOT EXISTS photo_results (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
            row = db.execute("SELECT result FROM photo_results WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        result = create()
        # A concurrent worker can finish first. Every worker returns the winning row.
        with sqlite3.connect(path, timeout=5) as db:
            db.execute("INSERT OR IGNORE INTO photo_results(key,result) VALUES (?,?)",
                       (key, json.dumps(result, ensure_ascii=False)))
            row = db.execute("SELECT result FROM photo_results WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0])
