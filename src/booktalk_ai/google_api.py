from __future__ import annotations

import json
import logging
import os
import random
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import truststore

logger = logging.getLogger(__name__)


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
    deadline = time.monotonic() + timeout
    for attempt in range(2):
        try:
            remaining = timeout if attempt == 0 else max(0.001, deadline - time.monotonic())
            with urlopen(request, timeout=remaining,
                         context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as response:
                content = response.read(limit + 1)
            break
        except (HTTPError, URLError, ConnectionError, TimeoutError) as exc:
            status = exc.code if isinstance(exc, HTTPError) else None
            reason = exc.reason if isinstance(exc, URLError) else exc
            timed_out = isinstance(reason, TimeoutError)
            code = {
                400: "MODEL_REQUEST_INVALID", 401: "MODEL_AUTH_FAILED",
                403: "MODEL_ACCESS_DENIED", 404: "MODEL_NOT_FOUND",
                429: "MODEL_QUOTA_EXCEEDED",
            }.get(status, "MODEL_UNAVAILABLE")
            transient = status in {500, 502, 503, 504} or (
                status is None and not isinstance(reason, ssl.SSLError)
            )
            delay = 0.2 + random.random() * 0.1
            retry = attempt == 0 and transient and time.monotonic() + delay < deadline
            # Never log exception text, URLs, payloads, credentials or provider bodies.
            logger.warning("provider_request_failed code=%s category=%s status=%s attempt=%d retry=%s",
                           code, "timeout" if timed_out else ("http" if status else "connection"),
                           status, attempt + 1, retry)
            if retry:
                time.sleep(delay)
                continue
            if timed_out:
                raise TimeoutError from exc
            raise ModelUnavailable(code) from exc

    if len(content) > limit:
        logger.warning("provider_response_invalid category=size_limit")
        raise ModelResponseInvalid
    try:
        body = json.loads(content)
    except ValueError as exc:
        logger.warning("provider_response_invalid category=json")
        raise ModelResponseInvalid from exc
    if not isinstance(body, dict):
        logger.warning("provider_response_invalid category=object_required")
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
            logger.warning("provider_response_invalid category=incomplete_generation")
            raise ModelResponseInvalid
        parts = candidate["content"]["parts"]
        if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
            raise ModelResponseInvalid
        return [part for part in parts if not part.get("thought")]
    except (IndexError, KeyError, TypeError) as exc:
        logger.warning("provider_response_invalid category=candidate_structure")
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
