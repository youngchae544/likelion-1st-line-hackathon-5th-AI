# AI 서버 간 인증 배포 안내

AI 서버는 `/health`와 `/docs`를 공개하고, 모든 `/v1/*` 요청은 인증합니다.

- AI 환경변수: `AI_SERVICE_API_KEY` (32자 이상의 임의 비밀값)
- 요청 헤더: `X-AI-Service-Key: <같은 비밀값>`
- 키 누락/불일치: HTTP 401 `AI_UNAUTHORIZED`
- AI 서버 키 미설정/32자 미만: HTTP 503 `AI_AUTH_NOT_CONFIGURED` (유료 호출 차단)

이 키는 Gemini/Google Vision 키와 다른 내부 통신용 키입니다. 프론트의 `VITE_*`, GitHub,
로그, 채팅에 키를 넣지 마세요. Render 두 서비스의 비밀 환경변수로만 공유합니다.

## 백엔드 변경

다음 세 클라이언트가 AI 서버로 보내는 모든 요청에 헤더를 붙여야 합니다.

- `LearningAiClient`: 문장 첨삭·단어 설명·대화·STT·TTS
- `RemotePageAnalyzer`: 사진 분석
- `AiOrchestrationClient`: RAG 답변

Spring 설정 예시:

```yaml
app:
  ai:
    api-key: ${AI_SERVICE_API_KEY:}
```

각 클라이언트 생성자에 `@Value("${app.ai.api-key}") String apiKey`를 추가하고
`RestClient` 생성에 아래 설정을 붙입니다. 기존 URL·타임아웃 설정은 유지합니다.

```java
this.restClient = RestClient.builder()
        .baseUrl(baseUrl)
        .requestFactory(requestFactory)
        .defaultHeader("X-AI-Service-Key", apiKey)
        .build();
```

## 배포 순서

1. 충분히 긴 임의 키를 생성하고 AI와 백엔드의 `AI_SERVICE_API_KEY`에 같은 값을 설정합니다.
2. 위 헤더를 보내도록 백엔드를 수정·배포합니다. 이전 AI 코드는 추가 헤더를 무시하므로 호환됩니다.
3. 인증을 적용한 새 AI 코드를 배포합니다. 백엔드를 먼저 바꾸지 않으면 기존 호출이 401로 막힙니다.
4. 외부에서 키 없는 요청은 401, 백엔드를 통한 사진·첨삭·음성·RAG 요청은 성공하는지 확인합니다.

새 코드에는 인증을 끄는 배포 옵션이 없습니다. 키를 누락한 채 AI만 재배포하지 마세요.
Swagger `/docs`에서는 Authorize 버튼에 내부 키를 입력할 수 있지만 키를 저장소에 기록하지 마세요.
