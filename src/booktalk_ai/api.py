from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .models import Document, GraphEdge, GraphNode, Route
from .orchestrator import RagOrchestrator
from .conversation import DialogueContext, DialogueTurn, Opening, TurnResult, answer_dialogue, start_dialogue
from .google_api import ModelResponseInvalid, ModelUnavailable
from .learning import (
    InvalidMedia, OcrInvalid, OcrTimeout, OcrUnavailable, PhotoRequest, PhotoResult,
    WordLessonRequest, WordLessonResult, analyze_photo, make_lesson,
)
from .rag_generation import GeminiAnswerGenerator
from .sentence_review import SentenceReview, SentenceReviewRequest, review_sentence
from .speech import SpeechRequest, TranscriptionRequest, TranscriptionResult, synthesize, transcribe

load_dotenv(Path.cwd() / ".env", override=False)

app = FastAPI(title="Booktalk AI Service", version="0.1.0")
orchestrator = RagOrchestrator(
    generator=GeminiAnswerGenerator() if os.getenv("RAG_GENERATOR", "").lower() == "gemini" else None
)


def error_handler(status: int, code: str):
    async def handle(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": {"code": code}})
    return handle


async def provider_error_handler(request: Request, error: ModelUnavailable) -> JSONResponse:
    status = 429 if error.code.endswith("QUOTA_EXCEEDED") else 503
    return JSONResponse(status_code=status, content={"detail": {"code": error.code}})


for exception, status, code in (
    (InvalidMedia, 422, "INVALID_MEDIA"),
    (OcrTimeout, 504, "OCR_TIMEOUT"),
    (OcrInvalid, 502, "OCR_RESPONSE_INVALID"),
    (TimeoutError, 504, "MODEL_TIMEOUT"),
    (ModelResponseInvalid, 502, "MODEL_RESPONSE_INVALID"),
):
    app.add_exception_handler(exception, error_handler(status, code))

app.add_exception_handler(ModelUnavailable, provider_error_handler)
app.add_exception_handler(OcrUnavailable, provider_error_handler)


class DocumentInput(BaseModel):
    id: str
    text: str
    title: str = ""
    kind: str = "CONTENT"
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeInput(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class EdgeInput(BaseModel):
    source: str
    target: str
    type: str
    weight: float = Field(default=1.0, ge=0, le=1)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    route: Route | None = None
    documents: list[DocumentInput] = Field(default_factory=list, max_length=200)
    nodes: list[NodeInput] = Field(default_factory=list, max_length=500)
    edges: list[EdgeInput] = Field(default_factory=list, max_length=1500)


def convert(request: AnswerRequest):
    documents = [Document(**item.model_dump()) for item in request.documents]
    nodes = [GraphNode(**item.model_dump()) for item in request.nodes]
    edges = [GraphEdge(**item.model_dump()) for item in request.edges]
    return documents, nodes, edges


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/routes/preview")
def preview(request: AnswerRequest) -> dict:
    documents, _, _ = convert(request)
    return orchestrator.preview_route(request.query, documents)


@app.post("/v1/answers")
def answer(request: AnswerRequest) -> dict:
    documents, nodes, edges = convert(request)
    result = orchestrator.answer(request.query, documents, nodes, edges, request.route)
    return {
        "answer": result.answer,
        "route": result.route.value,
        "citations": list(result.citations),
        "confidence": result.confidence,
        "abstained": result.abstained,
        "plan": [{"order": step.order, "tool": step.tool.value, "reason": step.reason} for step in result.plan],
        "evidence": [{"id": item.id, "text": item.text, "source": item.source, "score": item.score,
                      "metadata": item.metadata} for item in result.evidence],
        "verification": {
            "passed": result.verification.passed,
            "citationValidity": result.verification.citation_validity,
            "evidenceCoverage": result.verification.evidence_coverage,
            "unsupportedClaims": list(result.verification.unsupported_claims),
        },
        "diagnostics": result.diagnostics,
    }


@app.post("/v1/sentence-reviews", response_model=SentenceReview)
def sentence_review(request: SentenceReviewRequest) -> SentenceReview:
    return review_sentence(request)


@app.post("/v1/photo-analyses", response_model=PhotoResult)
def photo_analysis(request: PhotoRequest) -> PhotoResult:
    return analyze_photo(request)


@app.post("/v1/word-lessons", response_model=WordLessonResult)
def word_lesson(request: WordLessonRequest) -> WordLessonResult:
    return make_lesson(request)


@app.post("/v1/dialogues/openings", response_model=Opening)
def dialogue_opening(request: DialogueContext) -> Opening:
    return start_dialogue(request)


@app.post("/v1/dialogues/turns", response_model=TurnResult)
def dialogue_turn(request: DialogueTurn) -> TurnResult:
    return answer_dialogue(request)


@app.post("/v1/transcriptions", response_model=TranscriptionResult)
def transcription(request: TranscriptionRequest) -> TranscriptionResult:
    return transcribe(request)


@app.post("/v1/speech", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
def speech(request: SpeechRequest) -> Response:
    return Response(synthesize(request), media_type="audio/wav",
                    headers={"X-Playback-Rate": str(request.playback_rate)})
