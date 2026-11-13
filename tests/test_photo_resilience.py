import json

import pytest
from fastapi.testclient import TestClient

from booktalk_ai import learning
from booktalk_ai.api import app
from booktalk_ai.google_api import ModelResponseInvalid

client = TestClient(app)
PHOTO = {"image_base64": "aW1hZ2U=", "language": "vi"}
ITEM = {"expression": "두근두근", "type": "mimetic", "definition": "가슴이 자꾸 세게 뛰는 모양",
        "translation": "Tim đập thình thịch", "source_index": 0}


@pytest.mark.parametrize("invalid", [
    "not json", '{"expressions":null}', '{"expressions":[{"expression":"두근두근"}]}',
    json.dumps({"expressions": [{**ITEM, "source_index": 99}]}),
    json.dumps({"expressions": [{**ITEM, "expression": "깡충깡충"}]}),
])
def test_invalid_generation_retries_using_same_ocr(monkeypatch, caplog, invalid):
    reads, calls = [], []

    def ocr(*args, **kwargs):
        reads.append(1)
        return "가슴이\n두근두근했어요."

    def generate(instructions, prompt, schema, **kwargs):
        calls.append(kwargs["timeout"])
        assert schema["properties"]["expressions"]["items"]["required"] == [
            "expression", "type", "definition", "translation", "source_index"]
        assert "source_sentence" not in schema["properties"]["expressions"]["items"]["properties"]
        return invalid if len(calls) == 1 else json.dumps({"expressions": [ITEM]})

    monkeypatch.setattr(learning, "read_photo", ocr)
    monkeypatch.setattr(learning, "generate_json", generate)
    response = client.post("/v1/photo-analyses", json=PHOTO)
    assert response.status_code == 200
    assert response.json()["expressions"][0]["source_sentence"] == "가슴이 두근두근했어요."
    assert len(reads) == 1 and len(calls) == 2
    assert calls[1] <= calls[0] <= 20
    assert "두근두근" not in caplog.text


def test_second_invalid_result_returns_error_not_empty_success(monkeypatch):
    calls = []
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "가슴이 두근두근했어요.")

    def generate(*args, **kwargs):
        calls.append(1)
        raise ModelResponseInvalid

    monkeypatch.setattr(learning, "generate_json", generate)
    result = client.post("/v1/photo-analyses", json=PHOTO)
    assert result.status_code == 502
    assert result.json() == {"detail": {"code": "MODEL_RESPONSE_INVALID"}}
    assert len(calls) == 2


def test_photo_budget_prevents_regeneration_after_slow_ocr(monkeypatch):
    clock = iter([0, 45, 50, 51])
    monkeypatch.setattr(learning.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "가슴이 두근두근했어요.")
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise ModelResponseInvalid

    monkeypatch.setattr(learning, "generate_json", generate)
    with pytest.raises(TimeoutError):
        learning.analyze_photo(learning.PhotoRequest(**PHOTO))
    assert calls == [5]


def test_no_text_skips_llm(monkeypatch):
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "")
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: pytest.fail("Empty OCR must skip Gemini"))
    response = client.post("/v1/photo-analyses", json=PHOTO)
    assert response.json()["status"] == "no_text"


def test_selected_sources_preserve_correct_context(monkeypatch):
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "토끼는 두근두근했어요.\n돌이 퐁당 빠졌어요.")
    other = {**ITEM, "expression": "퐁당", "source_index": 1}
    monkeypatch.setattr(learning, "generate_json",
                        lambda *a, **kw: json.dumps({"expressions": [other]}))
    response = client.post("/v1/photo-analyses", json=PHOTO)
    assert response.json()["expressions"][0]["source_sentence"] == "돌이 퐁당 빠졌어요."
