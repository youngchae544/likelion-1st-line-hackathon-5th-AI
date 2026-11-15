import io
import json

import pytest
from fastapi.testclient import TestClient

from booktalk_ai import google_api
from booktalk_ai.api import app

client = TestClient(app)
SAMPLE = {
    "sentence": "아이가 넘어져서 두근두근 울었어요.",
    "expression": "두근두근",
    "definition": "가슴이 자꾸 세게 뛰는 모양",
    "source_sentence": "토끼는 당근을 보고 가슴이 두근두근했어요.",
    "language": "en",
    "translation": "A pounding heartbeat",
    "related": [{"expression": "콩닥콩닥", "example": "마음이 콩닥콩닥해요."}],
}


@pytest.mark.parametrize("result", [
    {"kind": "meaning", "hint": "선물을 열기 직전의 마음을 떠올려 보세요."},
    {"kind": "polish", "good_point": "뜻을 잘 전달했어요.", "correction": "조사를 빼 보세요.",
     "suggested_sentence": "가슴이 두근두근해요."},
    {"kind": "natural"},
    {"kind": "missing", "hint": "선물을 열기 전 가슴이 어떻게 뛰는지 말해 보세요."},
])
def test_structured_review(monkeypatch, result):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def model(request, timeout, context):
        payload = json.loads(request.data)
        assert timeout == 30
        assert request.full_url.endswith("/models/gemini-3.5-flash-lite:generateContent")
        assert request.get_header("X-goog-api-key") == "test-key"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert json.loads(payload["contents"][0]["parts"][0]["text"])["source_sentence"] == SAMPLE["source_sentence"]
        candidate = {
            "finishReason": "STOP",
            "content": {"parts": [{"text": json.dumps({"result": result})}]},
        }
        return io.BytesIO(json.dumps({"candidates": [candidate]}).encode())

    monkeypatch.setattr(google_api, "urlopen", model)
    sample = {**SAMPLE, "sentence": "선물을 기다려요."} if result["kind"] == "missing" else SAMPLE
    response = client.post("/v1/sentence-reviews", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == result
    assert body["original"] == sample["sentence"]
    assert body["translation"] == (SAMPLE["translation"] if result["kind"] == "meaning" else None)
    assert body["related"] == (SAMPLE["related"] if result["kind"] == "natural" else [])


def test_errors_are_distinguished(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.post("/v1/sentence-reviews", json=SAMPLE).status_code == 503

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(google_api, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"{}"))
    assert client.post("/v1/sentence-reviews", json=SAMPLE).status_code == 502

    def timeout(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(google_api, "urlopen", timeout)
    assert client.post("/v1/sentence-reviews", json=SAMPLE).status_code == 504
    assert client.post("/v1/sentence-reviews", json={**SAMPLE, "sentence": " "}).status_code == 422


def test_model_override_and_incomplete_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "another-model")

    def model(request, timeout, context):
        assert request.full_url.endswith("/models/another-model:generateContent")
        return io.BytesIO(b'{"candidates":[{"finishReason":"MAX_TOKENS"}]}')

    monkeypatch.setattr(google_api, "urlopen", model)
    assert client.post("/v1/sentence-reviews", json=SAMPLE).status_code == 502


@pytest.mark.parametrize("http_status,expected_code,expected_status", [
    (400, "MODEL_REQUEST_INVALID", 503),
    (401, "MODEL_AUTH_FAILED", 503),
    (403, "MODEL_ACCESS_DENIED", 503),
    (404, "MODEL_NOT_FOUND", 503),
    (429, "MODEL_QUOTA_EXCEEDED", 429),
])
def test_provider_errors_do_not_expose_keys(monkeypatch, http_status, expected_code, expected_status):
    from urllib.error import HTTPError

    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")

    def failed_request(request, **kwargs):
        raise HTTPError(request.full_url, http_status, "sensitive provider message", {}, None)

    monkeypatch.setattr(google_api, "urlopen", failed_request)
    response = client.post("/v1/sentence-reviews", json=SAMPLE)
    assert response.status_code == expected_status
    assert response.json() == {"detail": {"code": expected_code}}
    assert "private-test-key" not in response.text


@pytest.mark.parametrize("sentence", ["구불구불하다", "길이 구불구불해요.", "길이 구불 구불했어요."])
def test_literal_expression_cannot_be_reported_missing(monkeypatch, sentence):
    from booktalk_ai import sentence_review as review
    calls = []
    def generate(instructions, prompt, schema, **kwargs):
        assert json.loads(prompt)["has_literal_target"] is True
        assert "missing" not in schema["properties"]["result"]["properties"]["kind"]["enum"]
        calls.append(1)
        return json.dumps({"result": {"kind": "missing", "hint": "힌트"}}) if len(calls) == 1 else json.dumps({"result": {"kind": "natural"}})
    monkeypatch.setattr(review, "generate_json", generate)
    response = client.post("/v1/sentence-reviews", json={**SAMPLE, "expression": "구불구불", "sentence": sentence})
    assert response.status_code == 200 and response.json()["result"]["kind"] == "natural"
    assert len(calls) == 2
    assert "missing" in review.GEMINI_REVIEW_SCHEMA["properties"]["result"]["properties"]["kind"]["enum"]


def test_repeated_false_missing_is_error_not_feedback(monkeypatch):
    from booktalk_ai import sentence_review as review
    monkeypatch.setattr(review, "generate_json", lambda *a, **kw: json.dumps({"result": {"kind": "missing", "hint": "힌트"}}))
    r = client.post("/v1/sentence-reviews", json={**SAMPLE, "expression": "구불구불", "sentence": "구불구불하다"})
    assert r.status_code == 502


def test_decomposed_hangul_is_still_present(monkeypatch):
    import unicodedata
    from booktalk_ai import sentence_review as review
    def generate(instructions, prompt, schema):
        assert json.loads(prompt)["has_literal_target"] is True
        return '{"result":{"kind":"natural"}}'
    monkeypatch.setattr(review, "generate_json", generate)
    r = client.post("/v1/sentence-reviews", json={**SAMPLE, "expression": "구불구불", "sentence": unicodedata.normalize("NFD", "구불구불하다")})
    assert r.status_code == 200
