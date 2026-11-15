import pytest
from fastapi.testclient import TestClient

from booktalk_ai import api, sentence_review as review
from booktalk_ai.google_api import ModelResponseInvalid

KEY = 'test-service-key-not-a-secret-12345678'
client = TestClient(api.app)
SAMPLE = dict(sentence='구불구불하다', expression='구불구불', definition='여러 번 굽은 모양', language='en', translation='winding')


@pytest.mark.parametrize('path', [
    '/v1/speech', '/v1/transcriptions', '/v1/photo-analyses', '/v1/sentence-reviews',
    '/v1/word-lessons', '/v1/dialogues/openings', '/v1/dialogues/turns', '/v1/answers', '/v1/routes/preview',
])
def test_missing_key_cannot_reach_any_v1_endpoint(path):
    r = client.post(path, json={})
    assert r.status_code == 401
    assert r.json() == {'detail': {'code': 'AI_UNAUTHORIZED'}}


def test_correct_key_allows_call_and_wrong_key_does_not(monkeypatch):
    calls = []
    monkeypatch.setattr(api, 'synthesize', lambda request: calls.append(request.text) or b'RIFFtest')
    assert client.post('/v1/speech', json={'text':'안녕'}, headers={'X-AI-Service-Key':'wrong'}).status_code == 401
    assert not calls
    r = client.post('/v1/speech', json={'text':'안녕'}, headers={'X-AI-Service-Key':KEY})
    assert r.status_code == 200 and calls == ['안녕']
    assert KEY not in r.text


@pytest.mark.parametrize('key', ['', 'too-short'])
def test_missing_server_configuration_fails_closed_but_health_is_open(monkeypatch, key):
    monkeypatch.setenv('AI_SERVICE_API_KEY', key)
    assert client.get('/health').status_code == 200
    r = client.post('/v1/speech', json={'text':'안녕'}, headers={'X-AI-Service-Key':KEY})
    assert r.status_code == 503
    assert r.json()['detail']['code'] == 'AI_AUTH_NOT_CONFIGURED'


def test_all_photo_context_fields_accept_500_characters(monkeypatch):
    monkeypatch.setattr(review, 'generate_json', lambda *a, **kw: '{"result":{"kind":"natural"}}')
    payload = {**SAMPLE, 'definition':'가'*500, 'translation':'a'*500, 'source_sentence':'나'*500}
    assert client.post('/v1/sentence-reviews', json=payload, headers={'X-AI-Service-Key':KEY}).status_code == 200
    for field in ['definition','translation','source_sentence']:
        r = client.post('/v1/sentence-reviews', json={**payload,field:'가'*501}, headers={'X-AI-Service-Key':KEY})
        assert r.status_code == 422


def test_retry_uses_only_remaining_time(monkeypatch):
    clock = [0]
    calls = []
    monkeypatch.setattr(review.time, 'monotonic', lambda: clock[0])
    def generate(*args, **kw):
        calls.append(kw['timeout'])
        if len(calls) == 1:
            clock[0] = 35
            raise ModelResponseInvalid
        clock[0] = 40
        return '{"result":{"kind":"natural"}}'
    monkeypatch.setattr(review, 'generate_json', generate)
    assert review.review_sentence(review.SentenceReviewRequest(**SAMPLE)).result.kind == 'natural'
    assert calls == [20,10]


def test_late_model_result_is_not_returned(monkeypatch):
    clock = [0]
    monkeypatch.setattr(review.time, 'monotonic', lambda: clock[0])
    def generate(*args, **kw):
        clock[0] = 46
        return '{"result":{"kind":"natural"}}'
    monkeypatch.setattr(review, 'generate_json', generate)
    with pytest.raises(TimeoutError):
        review.review_sentence(review.SentenceReviewRequest(**SAMPLE))


def test_expired_budget_does_not_start_another_request(monkeypatch):
    clock = [0]
    calls = []
    monkeypatch.setattr(review.time, 'monotonic', lambda: clock[0])
    def generate(*args, **kw):
        calls.append(1)
        clock[0] = 46
        raise ModelResponseInvalid
    monkeypatch.setattr(review, 'generate_json', generate)
    with pytest.raises(TimeoutError):
        review.review_sentence(review.SentenceReviewRequest(**SAMPLE))
    assert len(calls) == 1


def test_swagger_documents_auth_for_v1_but_not_health():
    schema = client.get('/openapi.json').json()
    assert schema['paths']['/v1/speech']['post']['security'] == [{'ServiceKey': []}]
    assert 'security' not in schema['paths']['/health']['get']
    assert schema['components']['securitySchemes']['ServiceKey']['name'] == 'X-AI-Service-Key'
