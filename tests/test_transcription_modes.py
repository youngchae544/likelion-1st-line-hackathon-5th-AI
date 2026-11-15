import pytest
from fastapi.testclient import TestClient
from booktalk_ai.api import app
from booktalk_ai import speech

client = TestClient(app)
AUDIO = {"mime_type": "audio/wav", "audio_base64": "UklGRg=="}


@pytest.mark.parametrize("mode", [None, "DICTATION", "PRONUNCIATION"])
def test_preserves_nonword_transcript_and_does_not_provide_answer_hint(monkeypatch, mode):
    def generate(instructions, prompt, schema, **kwargs):
        assert "비단어" in instructions and "자동 보정" not in prompt
        assert "두근두근" not in instructions + prompt
        assert "두군두군" not in instructions + prompt
        assert ("발음 관찰용 전사" in instructions) == (mode == "PRONUNCIATION")
        assert kwargs["media"] == {"mimeType": "audio/wav", "data": AUDIO["audio_base64"]}
        return '{"text":"두군두군"}'
    monkeypatch.setattr(speech, "generate_json", generate)
    payload = AUDIO if mode is None else {**AUDIO, "mode": mode}
    response = client.post("/v1/transcriptions", json=payload)
    assert response.status_code == 200
    assert response.json() == {"text": "두군두군", "status": "ok"}


def test_silence_does_not_invent_a_target(monkeypatch):
    monkeypatch.setattr(speech, "generate_json", lambda *a, **kw: '{"text":""}')
    assert client.post("/v1/transcriptions", json={**AUDIO, "mode": "PRONUNCIATION"}).json() == {
        "text": "", "status": "no_speech"}


def test_unsupported_mode_rejected():
    assert client.post("/v1/transcriptions", json={**AUDIO, "mode": "SCORING"}).status_code == 422
