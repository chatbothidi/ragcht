# Medical & Event RAG Chatbot

의료 문서와 행사 문서를 기반으로 한 RAG 챗봇 시스템입니다.

## 기술 스택

- **LLM**: Vertex AI Gemini 2.5 Flash
- **임베딩**: text-multilingual-embedding-002 (768차원)
- **벡터 스토어**: Vertex AI Vector Search
- **한국어 형태소**: kiwipiepy
- **검색**: 하이브리드 (벡터 + BM25 + RRF)
- **API**: FastAPI + SSE 스트리밍
- **대화 메모리**: Redis

## 사전 요구사항

- Python 3.12+
- GCP 프로젝트 (Vertex AI API 활성화)
- `gcloud auth application-default login` 완료
- Docker (Redis 실행용)

## 시작하기

```bash
# 1. 의존성 설치
make install

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에서 GCP_PROJECT_ID 수정

# 3. Redis 실행
make docker-up

# 4. Vector Search 셋업 (최초 1회)
make setup

# 5. 문서 인덱싱
# data/documents/medical/ 또는 data/documents/events/ 에 문서 추가 후:
make index

# 6. 서버 실행
make run
```

## API 엔드포인트

### `POST /chat` - 챗봇 대화

```json
{
  "query": "고혈압 치료 방법은?",
  "session_id": "optional-session-id",
  "source_type_filter": "medical",
  "stream": false
}
```

### `POST /search` - 직접 검색

```json
{
  "query": "건강검진 행사 일정",
  "top_k": 5,
  "source_type_filter": "event"
}
```

### `GET /admin/health` - 헬스체크

### `DELETE /admin/session/{session_id}` - 세션 삭제

## 문서 구조

```
data/documents/
├── medical/    # 의료 문서 (PDF, DOCX, TXT, MD)
└── events/     # 행사 문서 (PDF, DOCX, TXT, MD)
```

디렉토리에 따라 자동으로 `source_type`이 지정됩니다.

## 개발

```bash
make dev      # 개발 의존성 설치
make test     # 테스트 실행
make lint     # 린트 검사
make format   # 코드 포맷팅
```
