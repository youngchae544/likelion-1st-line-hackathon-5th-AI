# Booktalk AI Service

검증된 백엔드 데이터를 근거로 답변하는 독립 AI 오케스트레이션 서비스다.

## 처리 방식

1. 질문 복잡도와 의도를 분류한다.
2. 단순 질문은 Hybrid(BM25 + deterministic semantic hashing) 검색으로 처리한다.
3. 관계 질문은 GraphRAG Local Search를 사용한다.
4. 전체 주제 질문은 GraphRAG Global Search를 사용한다.
5. 아이 정보·학습 이력·표현 관계가 함께 필요한 질문은 제한형 PlanRAG로 처리한다.
6. 생성 답변의 evidence coverage와 citation을 검증한다.
7. 근거가 부족하면 답변을 추측하지 않고 보류한다.

그래프는 LLM이 자동 생성하지 않는다. Spring 백엔드가 소유권 검증을 마친 `아이–페이지–표현–유형–언어`
관계를 요청에 넣는다. 따라서 그래프 추출 환각이 영구 저장되는 것을 막는다.

## 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn booktalk_ai.api:app --host 127.0.0.1 --port 8090
```

## 테스트

핵심 RAG 엔진 테스트는 외부 패키지 없이 실행할 수 있다.

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## API

- `GET /health`
- `POST /v1/answers`
- `POST /v1/routes/preview`

기본 생성기는 검색 근거에서 문장을 추출합니다. Gemini 생성기는 아래 RAG 설정에서 활성화할 수 있습니다.

## 그림책 표현 문장 평가

`POST /v1/sentence-reviews`는 부모의 문장을 검토해 `polish`, `meaning`,
`natural`, `missing` 중 하나를 반환합니다. 호출자는 목표 표현과 함께 검증된
한국어 뜻, 출처 문장, 설정 언어의 번역을 넘깁니다. 관련 표현과 예문은 선택 사항입니다.
뜻 혼동 피드백의 정의·번역과 자연스러운 문장의 관련 표현은 요청의 검증된 자료를 그대로 사용합니다.

`ai-service` 폴더에서 서비스를 시작하면 해당 폴더의 `.env`를 읽습니다.
파일이 없다면 `cp .env.example .env`로 만듭니다. [Google AI Studio](https://aistudio.google.com/app/apikey)에서
발급한 키를 `GEMINI_API_KEY`에 넣고 서비스를 재시작합니다.
기본 모델은 `gemini-3.5-flash-lite`이며 `GEMINI_MODEL`로 변경할 수 있습니다.
배포 환경에서는 `GEMINI_API_KEY`를 플랫폼의 비밀 환경 변수에 등록합니다.
이미 설정된 환경 변수가 `.env` 값보다 우선합니다.
문장과 근거 텍스트는 Gemini로 전송되며, 무료 티어 검증에는 샘플 문장을 사용하세요.

```json
{
  "sentence": "아이가 넘어져서 두근두근 울었어요.",
  "expression": "두근두근",
  "definition": "가슴이 자꾸 세게 뛰는 모양",
  "source_sentence": "토끼는 당근을 보고 가슴이 두근두근했어요.",
  "language": "en",
  "translation": "A pounding heartbeat"
}
```

`result.kind`가 판정입니다. 교정 결과에는 `good_point`, `correction`,
`suggested_sentence`가 있고 뜻 혼동과 표현 누락에는 `hint`가 있습니다.
키 미설정·Gemini 요청 실패는 HTTP 503, 응답 지연은 504, 모델 응답 형식 오류는 502입니다.

## 사진·단어 학습·대화·음성 API

모든 요청은 JSON입니다. `/docs`에서 필드와 결과 형식을 확인할 수 있습니다.
배포 환경에 `GEMINI_API_KEY`와 Cloud Vision이 활성화된 프로젝트의
`GOOGLE_VISION_API_KEY`를 비밀 환경 변수로 등록합니다.

| 경로 | 요청 | 결과 |
|---|---|---|
| `POST /v1/photo-analyses` | `image_base64`와 `language`. JPEG·PNG·WebP, 최대 4 MiB | Cloud Vision이 글자를 읽고 Gemini가 표현 유형, 뜻, 번역, 출처 문장을 반환. `no_text`와 `no_expressions`를 구분 |
| `POST /v1/word-lessons` | `expression`, `definition`, `source_sentence`, `language`, `translation` | 쓰는 상황, 비슷한 표현과 차이, 아이에게 설명할 문장, 연습 예시 |
| `POST /v1/dialogues/openings` | `expression`, `definition`, `source_sentence`, `language`, `scene` | 아이의 첫 발화와 힌트 |
| `POST /v1/dialogues/turns` | 시작 요청의 필드, 수락된 교환 목록 `previous`, 현재 교환 `current`. 각 교환은 `child_line`과 `parent_answer` | `retry`, `continue`, `complete`, 수락된 턴 수와 피드백 |
| `POST /v1/transcriptions` | `audio_base64`와 `mime_type`. 최대 8 MiB | 한국어 전사 `text` 또는 `no_speech` |
| `POST /v1/speech` | `text`와 선택 값 `playback_rate` | `audio/wav` 및 `X-Playback-Rate` 헤더 |

`language`는 `vi`, `zh-CN`, `en`을 지원합니다. STT 결과의 `text`는
기존 문장 평가 API나 대화 턴 API에 전달할 수 있습니다. WAV 자체는 기본 속도이며
재생하는 쪽에서 `X-Playback-Rate`를 적용합니다. TTS 모델은 `GEMINI_TTS_MODEL`로 지정합니다.

사진과 OCR 전문은 서비스에서 디스크에 쓰지 않습니다. 사진은 Vision으로,
인식한 글은 Gemini로 전달됩니다. 결과에는 표현별 출처 문장만 포함됩니다.
`OCR_UNAVAILABLE`, `OCR_TIMEOUT`, `OCR_RESPONSE_INVALID`는
글자가 없거나 표현이 없다는 정상 결과와 구분합니다.

대화는 수락된 부모 답변 네 개에서 완료합니다. 백엔드는 세션과 수락된 교환을 보관하고,
`retry`이면 이전 목록에 추가하지 않은 채 같은 아이 발화에 대한 새 답변을 받습니다.
네트워크 재시도의 요청 식별과 중복 저장 방지는 백엔드가 담당합니다.

## RAG와 직접 조회

표현 ID로 뜻과 출처 문장을 이미 조회한 문장 평가·단어 학습·대화에서는
호출자가 그 근거를 직접 제공합니다. 다른 장면이나 학습 이력까지 검색하는 요청은
기존 `POST /v1/answers`를 사용합니다. `RAG_GENERATOR=gemini`를 설정하면
검색된 근거를 바탕으로 Gemini가 답변을 만들고 기존 인용·근거 검증을 거칩니다.
이 변수를 생략하면 기존 추출형 생성기를 사용합니다.


## 배포 전 보완 사항

출처 문장을 보관하지 않는 학습 이력을 지원하도록 단어 학습, 문장 평가, 대화 API의
`source_sentence`는 생략하거나 `null`로 전달할 수 있습니다. 이때 전달된 뜻을
기준으로 판단하며 생성 예문을 그림책 원문으로 소개하지 않습니다. 대화에는 별도의
연습 상황인 `scene`을 전달합니다.

공급자 오류는 `MODEL_KEY_MISSING`, `MODEL_AUTH_FAILED`,
`MODEL_ACCESS_DENIED`, `MODEL_NOT_FOUND`, `MODEL_REQUEST_INVALID`,
`MODEL_QUOTA_EXCEEDED`로 구분합니다. Vision 오류에는 `OCR_` 접두사가 붙습니다.
한도 초과는 HTTP 429이며 나머지 공급자 설정·연결 오류는 HTTP 503입니다.
응답에 공급자의 원문 오류나 API 키를 포함하지 않습니다.

업로드용 백엔드 변경은 페이지 업로드를 OCR과 같은 4 MiB로 제한하고,
분석이 끝난 사진의 삭제 실패 시 파일 경로를 유지합니다. 백그라운드에서 1분마다
삭제를 재시도하며, 실제 삭제가 성공하면 삭제 완료를 기록합니다.

### 동일 사진 결과 재사용

`/v1/photo-analyses`는 이미지 내용의 SHA-256 키와 설명 언어별로 첫 정상 추출 결과를
SQLite에 저장합니다. 반복 요청에서는 Gemini를 다시 호출하지 않으며 표현 순서·분류·뜻·번역을
재사용합니다. OCR은 다시 수행하여 출처 문장을 복원합니다. 저장 내용은 표현·설명·번역·문장
번호뿐이며 이미지, 이미지 base64, OCR 전체 텍스트, 출처 문장 원문은 저장하지 않습니다.
실패 응답은 캐시하지 않습니다. OCR에서 기존 표현의 출처를 확인할 수 없으면 새 표현 목록을
생성하지 않고 OCR 오류를 반환합니다. 최초 추출의 누락을 자동으로 보완하는 기능은 아닙니다.

- `PHOTO_CACHE_PATH`: 기본 `./data/photo-cache.sqlite3`. 여러 인스턴스는 같은 영구 저장소를
  공유해야 합니다. Render 무료 인스턴스의 임시 파일은 재시작·재배포 시 유실될 수 있으므로,
  배포 간에도 동일 결과를 보장하려면 영구 디스크 또는 백엔드 DB 기반 보관이 필요합니다.
- `PHOTO_CACHE_VERSION`: 기본 `v1`. 의도적으로 전체 결과를 다시 생성할 때만 변경합니다.
- 동일성 기준은 같은 이미지 파일 내용과 같은 설명 언어입니다. 다시 촬영하거나 다시 압축한
  파일은 다른 이미지로 취급합니다. 같은 파일도 다른 설명 언어는 별도 결과로 저장합니다.
