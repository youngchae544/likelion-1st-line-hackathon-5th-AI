import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import time

import pytest

from booktalk_ai import learning, photo_cache
from booktalk_ai.google_api import ModelResponseInvalid

PHOTO = {"image_base64": "aW1hZ2U=", "language": "vi"}
ITEM = {"expression": "두근두근", "type": "mimetic", "definition": "가슴이 뛰는 모양",
        "translation": "heartbeat", "source_index": 0}


def setup(monkeypatch):
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "토끼의 가슴이 두근두근했어요.")


def test_repeat_and_module_reload_reuse_first_result_without_saving_source(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: json.dumps({"expressions": [ITEM]}))
    first = learning.analyze_photo(learning.PhotoRequest(**PHOTO))
    importlib.reload(photo_cache)
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: pytest.fail("Must reuse stored results"))
    assert learning.analyze_photo(learning.PhotoRequest(**PHOTO)) == first
    with sqlite3.connect(os.environ["PHOTO_CACHE_PATH"]) as db:
        raw = db.execute("SELECT result FROM photo_results").fetchone()[0]
    assert "source_sentence" not in raw and "토끼" not in raw
    assert PHOTO["image_base64"] not in raw


def test_language_and_image_are_independent(monkeypatch):
    setup(monkeypatch)
    calls = []
    def generate(*a, **kw):
        calls.append(1)
        return json.dumps({"expressions": [ITEM]})
    monkeypatch.setattr(learning, "generate_json", generate)
    for data in [PHOTO, {**PHOTO, "language": "en"}, {**PHOTO, "image_base64": "b3RoZXI="}, PHOTO]:
        learning.analyze_photo(learning.PhotoRequest(**data))
    assert len(calls) == 3


def test_failed_generation_not_cached(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: "bad json")
    with pytest.raises(ModelResponseInvalid):
        learning.analyze_photo(learning.PhotoRequest(**PHOTO))
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: json.dumps({"expressions": [ITEM]}))
    assert learning.analyze_photo(learning.PhotoRequest(**PHOTO)).status == "ok"


def test_simultaneous_requests_share_generation(monkeypatch):
    setup(monkeypatch)
    calls = []
    def generate(*a, **kw):
        calls.append(1)
        time.sleep(0.03)
        return json.dumps({"expressions": [ITEM]})
    monkeypatch.setattr(learning, "generate_json", generate)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: learning.analyze_photo(learning.PhotoRequest(**PHOTO)), range(4)))
    assert all(r == results[0] for r in results) and len(calls) == 1


def test_ocr_changes_do_not_trigger_new_expression_generation(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: json.dumps({"expressions": [ITEM]}))
    first = learning.analyze_photo(learning.PhotoRequest(**PHOTO))
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "다음 장면이에요. 토끼의 가슴이 두근두근했어요.")
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: pytest.fail("Must not regenerate"))
    assert learning.analyze_photo(learning.PhotoRequest(**PHOTO)) == first
