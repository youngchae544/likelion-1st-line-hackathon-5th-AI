import base64
import io
import json
import wave

from fastapi.testclient import TestClient

from booktalk_ai import conversation, learning, speech
from booktalk_ai.api import app

client = TestClient(app)
CONTEXT = {
    "expression": "두근두근", "definition": "가슴이 빨리 뛰는 모양",
    "source_sentence": "가슴이 두근두근했어요.", "language": "vi",
}
PHOTO = {"image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode(), "language": "vi"}
DIALOGUE = {**CONTEXT, "scene": "토끼가 선물을 열어요."}
EXCHANGE = {"child_line": "무슨 선물일까?", "parent_answer": "가슴이 두근두근해."}


def test_vision_ocr_and_evidence_filter(monkeypatch):
    def vision(url, key, payload, **kwargs):
        assert url == "https://vision.googleapis.com/v1/images:annotate"
        assert payload["requests"][0]["features"] == [{"type": "DOCUMENT_TEXT_DETECTION"}]
        return {"responses": [{"fullTextAnnotation": {"text": "가슴이\n두근두근했어요."}}]}

    def gemini(instructions, text, schema, **kwargs):
        assert "image_base64" not in text
        item = {"expression": "두근두근", "type": "mimetic", "definition": CONTEXT["definition"],
                "translation": "Tim đập thình thịch", "source_index": 0}
        return json.dumps({"expressions": [
            item, item,
        ]})

    monkeypatch.setattr(learning, "post_json", vision)
    monkeypatch.setattr(learning, "generate_json", gemini)
    body = client.post("/v1/photo-analyses", json=PHOTO).json()
    assert body["status"] == "ok"
    assert [item["expression"] for item in body["expressions"]] == ["두근두근"]
    assert "ocr_text" not in body


def test_photo_empty_states_and_invalid_media(monkeypatch):
    monkeypatch.setattr(learning, "post_json", lambda *a, **kw: {"responses": [{}]})
    assert client.post("/v1/photo-analyses", json=PHOTO).json()["status"] == "no_text"
    monkeypatch.setattr(learning, "read_photo", lambda *a, **kw: "안녕하세요.")
    monkeypatch.setattr(learning, "generate_json", lambda *a, **kw: '{"expressions":[]}')
    assert client.post("/v1/photo-analyses", json=PHOTO).json()["status"] == "no_expressions"


def test_lesson_keeps_verified_context(monkeypatch):
    lesson = {"situations": ["소풍 전날"], "similar_expression": "콩닥콩닥", "difference": "조금 가벼운 느낌",
              "child_explanation": "심장이 빨리 뛰는 느낌이야.", "practice_example": "내일이 기대돼서 두근두근해."}
    monkeypatch.setattr(learning, "generate_json", lambda *a: json.dumps(lesson))
    response = client.post("/v1/word-lessons", json={**CONTEXT, "translation": "Tim đập thình thịch"})
    assert response.status_code == 200
    assert response.json()["source_sentence"] == CONTEXT["source_sentence"]


def test_dialogue_retry_and_fourth_accepted_turn(monkeypatch):
    monkeypatch.setattr(conversation, "generate_json", lambda *a: '{"child_line":"어떤 선물일까?","hint":"선물을 생각해 보세요."}')
    assert client.post("/v1/dialogues/openings", json=DIALOGUE).json()["child_line"] == "어떤 선물일까?"
    payload = {**DIALOGUE, "previous": [EXCHANGE] * 3, "current": EXCHANGE}
    assessment = {"decision": "retry", "child_line": "", "hint": "다시 생각해 보세요.", "feedback": "다시 말해 봐요."}
    monkeypatch.setattr(conversation, "generate_json", lambda *a: json.dumps(assessment))
    body = client.post("/v1/dialogues/turns", json=payload).json()
    assert (body["status"], body["accepted_turns"], body["child_line"]) == ("retry", 3, None)
    assessment["decision"] = "accept"
    body = client.post("/v1/dialogues/turns", json=payload).json()
    assert (body["status"], body["accepted_turns"], body["child_line"]) == ("complete", 4, None)


def test_transcription_and_wav(monkeypatch):
    monkeypatch.setattr(speech, "generate_json", lambda *a, **kw: '{"text":"가슴이 두근두근해요."}')
    response = client.post("/v1/transcriptions", json={"mime_type": "audio/wav", "audio_base64": "UklGRg=="})
    assert response.json() == {"status": "ok", "text": "가슴이 두근두근해요."}
    response = client.post("/v1/transcriptions", json={"mime_type": "audio/wav", "audio_base64": "!!!!"})
    assert response.json()["detail"]["code"] == "INVALID_MEDIA"
    monkeypatch.setattr(speech, "generate_parts", lambda *a, **kw: [{"inlineData": {"data": "AAABAA=="}}])
    response = client.post("/v1/speech", json={"text": "두근두근", "playback_rate": 0.8})
    assert response.status_code == 200
    assert response.headers["x-playback-rate"] == "0.8"
    with wave.open(io.BytesIO(response.content)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (24000, 1, 2)


def test_learning_without_saved_source(monkeypatch):
    context = {key: value for key, value in CONTEXT.items() if key != "source_sentence"}
    lesson = {"situations": ["소풍 전날"], "similar_expression": "콩닥콩닥", "difference": "가벼운 느낌",
              "child_explanation": "심장이 빨리 뛰는 느낌이야.", "practice_example": "소풍이 기대돼서 두근두근해."}
    monkeypatch.setattr(learning, "generate_json", lambda *a: json.dumps(lesson))
    response = client.post("/v1/word-lessons", json={**context, "translation": "Tim đập thình thịch"})
    assert response.status_code == 200
    assert response.json()["source_sentence"] is None
    monkeypatch.setattr(conversation, "generate_json", lambda *a: '{"child_line":"어떤 선물일까?","hint":"기대하는 마음을 말해 보세요."}')
    assert client.post("/v1/dialogues/openings", json={**context, "scene": "선물을 열기 전"}).status_code == 200
