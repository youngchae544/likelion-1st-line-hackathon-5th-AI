from __future__ import annotations

import json
import os
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import truststore


class ModelUnavailable(Exception):
    def __init__(self, code: str = "MODEL_UNAVAILABLE"):
        super().__init__(code)
        self.code = code


class ModelResponseInvalid(Exception):
    pass


def post_json(url: str, key: str, payload: dict, *, timeout: int = 15, limit: int = 128_000) -> dict:
    if not key:
        raise ModelUnavailable("MODEL_KEY_MISSING")
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urlopen(request, timeout=timeout, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as response:
            content = response.read(limit + 1)
    except HTTPError as exc:
        code = {
            400: "MODEL_REQUEST_INVALID",
            401: "MODEL_AUTH_FAILED",
            403: "MODEL_ACCESS_DENIED",
            404: "MODEL_NOT_FOUND",
            429: "MODEL_QUOTA_EXCEEDED",
        }.get(exc.code, "MODEL_UNAVAILABLE")
        raise ModelUnavailable(code) from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError from exc
        raise ModelUnavailable from exc
    except ConnectionError as exc:
        raise ModelUnavailable from exc

    if len(content) > limit:
        raise ModelResponseInvalid
    try:
        body = json.loads(content)
    except ValueError as exc:
        raise ModelResponseInvalid from exc
    if not isinstance(body, dict):
        raise ModelResponseInvalid
    return body


def generate_parts(payload: dict, *, model: str | None = None, timeout: int = 15,
                   limit: int = 128_000) -> list[dict]:
    model = model or os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite"
    body = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent",
        os.environ.get("GEMINI_API_KEY", "").strip(), payload, timeout=timeout, limit=limit,
    )
    try:
        candidate = body["candidates"][0]
        if candidate["finishReason"] != "STOP":
            raise ModelResponseInvalid
        parts = candidate["content"]["parts"]
        if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
            raise ModelResponseInvalid
        return [part for part in parts if not part.get("thought")]
    except (IndexError, KeyError, TypeError) as exc:
        raise ModelResponseInvalid from exc


def generate_json(instructions: str, text: str, schema: dict, *, media: dict | None = None,
                  timeout: int = 15) -> str:
    parts = [{"text": text}]
    if media is not None:
        parts.append({"inlineData": media})
    result = generate_parts({
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
    }, timeout=timeout)
    try:
        return "".join(part.get("text", "") for part in result)
    except TypeError as exc:
        raise ModelResponseInvalid from exc
