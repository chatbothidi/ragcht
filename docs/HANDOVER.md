# Medical & Event RAG Chatbot - 인수인계서

> 최종 업데이트: 2026-04-16

---

## 1. 프로젝트 개요

의료 문서(지침, 공문 등)와 행사 문서(학술대회, 워크샵 등)를 기반으로 질의응답하는 **RAG(Retrieval-Augmented Generation) 챗봇**입니다.

### 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | FastAPI + Uvicorn |
| LLM | Gemini (google-genai SDK, Vertex AI) |
| Embedding | Vertex AI `text-multilingual-embedding-002` (768차원) |
| Vector DB | Vertex AI Vector Search (Matching Engine) |
| Reranking | Discovery Engine Ranking API |
| BM25 | rank-bm25 + kiwipiepy (한국어 형태소 분석) |
| OCR | Document AI (스캔 PDF), Gemini Vision (이미지) |
| Memory | 인메모리 딕셔너리 (서버 재시작 시 초기화) |
| Frontend | Vanilla JS + marked.js (단일 HTML) |

---

## 2. 디렉토리 구조

```
medical-event-rag/
├── api/                          # FastAPI 애플리케이션
│   ├── main.py                   # 앱 초기화, 라우트/미들웨어 등록
│   ├── middleware.py             # CORS, 로깅 미들웨어
│   ├── dependencies.py           # 싱글턴 의존성 팩토리 (@lru_cache)
│   └── routes/
│       ├── chat.py               # POST /chat (SSE 스트리밍)
│       └── admin.py              # GET /admin/health, DELETE /admin/session, GET /admin/download
│
├── src/                          # 핵심 비즈니스 로직
│   ├── config.py                 # pydantic-settings 기반 설정
│   ├── models.py                 # 데이터 모델 (Request/Response, DocumentChunk 등)
│   ├── rag_pipeline.py           # RAG 파이프라인 오케스트레이터
│   ├── hybrid_retriever.py       # 2단계 검색: 하이브리드(RRF) → 리랭킹
│   ├── generator.py              # Gemini LLM 응답 생성 (스트리밍 지원)
│   ├── embeddings.py             # 임베딩 서비스 (재시도 로직 포함)
│   ├── vectorstore.py            # Vector Search 인덱스 관리
│   ├── bm25_index.py             # BM25 인덱스 (한국어 형태소 토큰화)
│   ├── chunk_store.py            # 청크 ID → 메타데이터 매핑 (JSON)
│   ├── chunker.py                # 문서 청킹 (의료: 계층적, 행사: 재귀적)
│   ├── document_loader.py        # 문서 로더 (PDF/DOCX/MD/TXT/이미지)
│   └── memory.py                 # 대화 히스토리 (인메모리)
│
├── scripts/                      # CLI 스크립트
│   ├── setup_gcp.py              # Vector Search 인덱스/엔드포인트 생성 (최초 1회)
│   ├── index_documents.py        # 문서 인덱싱 (전체/증분/개별)
│   └── build_bm25_index.py       # BM25 인덱스 재빌드
│
├── static/
│   └── index.html                # 웹 UI (단일 페이지)
│
├── tests/                        # 테스트
│   ├── conftest.py               # 픽스처
│   ├── test_chunker.py           # 청킹 로직
│   ├── test_hybrid_retriever.py  # RRF 퓨전
│   └── test_pipeline.py          # 대화 메모리
│
├── data/                         # 데이터
│   ├── documents/{post_id}/      # 게시글별 문서 폴더
│   ├── bm25_index/               # BM25 직렬화 인덱스
│   └── chunk_store/              # 청크 메타데이터 JSON
│
├── .env                          # 환경변수 (git 제외)
├── .env.example                  # 환경변수 템플릿
├── Makefile                      # 개발 커맨드
├── requirements.txt              # 프로덕션 의존성
└── requirements-dev.txt          # 개발 의존성
```

---

## 3. 환경 설정

### 3.1 사전 요구사항

- Python 3.12+
- GCP 프로젝트 (Vertex AI, Document AI, Discovery Engine API 활성화)
- `gcloud` CLI 인증: `gcloud auth application-default login`

### 3.2 .env 설정

`.env.example`을 `.env`로 복사 후 값을 채웁니다:

```bash
cp .env.example .env
```

| 변수 | 필수 | 설명 |
|------|------|------|
| `GCP_PROJECT_ID` | O | GCP 프로젝트 ID |
| `GCP_LOCATION` | - | GCP 리전 (기본: asia-northeast3) |
| `EMBEDDING_MODEL` | - | 임베딩 모델 (기본: text-multilingual-embedding-002) |
| `LLM_MODEL` | - | LLM 모델 (기본: gemini-2.0-flash) |
| `LLM_LOCATION` | - | LLM 전용 리전 (기본: GCP_LOCATION과 동일) |
| `VERTEX_INDEX_ID` | O* | Vector Search 인덱스 리소스 ID |
| `VERTEX_ENDPOINT_ID` | O* | Vector Search 엔드포인트 리소스 ID |
| `VERTEX_COLLECTION_NAME` | - | 컬렉션 이름 (기본: medical_event_docs) |
| `CHUNK_SIZE` | - | 의료 문서 청크 크기 (기본: 800) |
| `CHUNK_OVERLAP` | - | 청크 오버랩 (기본: 120) |
| `TOP_K` | - | 최종 검색 결과 수 (기본: 7) |
| `BM25_TOP_K` | - | BM25 후보 수 (기본: 10) |
| `HYBRID_ALPHA` | - | 벡터 검색 가중치, 0~1 (기본: 0.6) |
| `RERANK_CANDIDATES` | - | 리랭킹 전 후보 수 (기본: 50) |
| `RERANK_TOP_K` | - | 리랭킹 후 상위 결과 수 (기본: 10) |
| `MAX_CONTEXT_TOKENS` | - | LLM 컨텍스트 최대 토큰 (기본: 4000) |
| `MAX_CONVERSATION_TURNS` | - | 대화 히스토리 유지 턴 수 (기본: 5) |

> *`VERTEX_INDEX_ID`, `VERTEX_ENDPOINT_ID`는 최초 `make setup` 실행 후 생성된 값을 넣어야 합니다.

---

## 4. 아키텍처

### 4.1 전체 요청 흐름

```
사용자 (브라우저)
  │
  ▼
POST /chat  ─────────────────────────────────────────────────────┐
  │                                                              │
  ▼                                                              │
RAGPipeline.query_stream()                                       │
  │                                                              │
  ├─ 1. 세션 관리: session_id 생성 또는 조회                        │
  │                                                              │
  ├─ 2. 대화 히스토리 로드 (인메모리)                                │
  │                                                              │
  ├─ 3. 쿼리 리라이팅 (멀티턴인 경우)                                │
  │     "그거 자세히" → "결핵 진료지침 4판 내용을 자세히 알려줘"         │
  │                                                              │
  ├─ 4. 검색 (HybridRetriever) ──────────────────────┐           │
  │     ├─ Stage 1: 하이브리드 검색                     │           │
  │     │   ├─ Vector Search (임베딩 유사도, 50개)      │           │
  │     │   ├─ BM25 (키워드 매칭, 50개)                │           │
  │     │   └─ RRF 퓨전 (가중 합산, 50개)              │           │
  │     ├─ Stage 2: 리랭킹                             │           │
  │     │   └─ Discovery Engine (시맨틱, 10개)         │           │
  │     └─ Stage 3: 게시글 확장                        │           │
  │         └─ 같은 post_id의 추가 청크 포함            │           │
  │                                                    │           │
  ├─ 5. 응답 생성 (Gemini 스트리밍)  ◄─────────────────┘           │
  │     컨텍스트 + 질문 → LLM → 토큰 스트림                        │
  │                                                              │
  ├─ 6. 대화 히스토리 저장                                         │
  │                                                              │
  └─ 7. 출처 정보 반환                                             │
       ▼                                                         │
  SSE 스트림 (토큰 → 출처 → 완료)  ◄─────────────────────────────┘
       │
       ▼
  브라우저: marked.js로 마크다운 렌더링
```

### 4.2 의존성 초기화 (`api/dependencies.py`)

모든 컴포넌트는 `@lru_cache`로 싱글턴 관리됩니다. 첫 요청 시 지연 초기화됩니다:

```
get_pipeline()
  ├── get_retriever()
  │     ├── get_embedding_service()  → Vertex AI 임베딩 클라이언트
  │     ├── get_vectorstore()        → Vector Search 인덱스/엔드포인트 로드
  │     ├── get_bm25_index()         → BM25 인덱스 파일 로드
  │     └── get_chunk_store()        → chunks.json 로드
  ├── get_generator()                → Gemini 클라이언트
  └── get_memory()                   → 인메모리 딕셔너리
```

---

## 5. 핵심 컴포넌트 상세

### 5.1 HybridRetriever (`src/hybrid_retriever.py`)

**3단계 검색 파이프라인:**

| 단계 | 메서드 | 설명 |
|------|--------|------|
| Stage 1 | `_hybrid_search()` | Vector + BM25 → RRF 퓨전. `HYBRID_ALPHA`(기본 0.6)로 벡터 검색에 더 높은 가중치 |
| Stage 2 | `_rerank()` | Discovery Engine Ranking API로 시맨틱 리랭킹. 실패 시 RRF 순서 폴백 |
| Stage 3 | `_expand_by_post()` | 검색된 청크와 같은 `post_id`를 가진 추가 청크를 포함 (게시글당 최대 5개) |

**RRF(Reciprocal Rank Fusion) 공식:**
```
score(doc) = α × 1/(K + rank_vector) + (1-α) × 1/(K + rank_bm25)
K = 60, α = HYBRID_ALPHA
```

### 5.2 LLMGenerator (`src/generator.py`)

- **시스템 프롬프트**: 한국어 전용, 컨텍스트 기반 답변, 첨부파일 다운로드 링크 형식 `[[다운로드:파일명]]`
- **스트리밍**: `generate_stream()` — Gemini의 `generate_content_stream()` 사용
- **연속 생성**: `generate()` — MAX_TOKENS로 끊기면 최대 3회 이어서 생성
- **쿼리 리라이팅**: `rewrite_query()` — 대명사 해소, 맥락 보존

### 5.3 DocumentChunker (`src/chunker.py`)

| 카테고리 | 전략 | 청크 크기 | 특징 |
|---------|------|----------|------|
| `medical` | 계층적 (섹션 헤더 → 문장 경계) | ~1600자 | 진단/소견/처방 등 의료 섹션 헤더로 분할, kiwipiepy 문장 분리 |
| `event` (및 기타) | 재귀적 문자 분할 | 5000자 | `RecursiveCharacterTextSplitter` 사용 |

새 카테고리 추가 시: `chunk_document()` 메서드에 분기를 추가하거나, 기본값(`_chunk_event`)이 범용적이므로 그대로 사용 가능.

### 5.4 DocumentLoader (`src/document_loader.py`)

| 포맷 | 처리 방식 |
|------|----------|
| PDF (텍스트) | PyMuPDF로 텍스트 추출 |
| PDF (스캔) | Document AI OCR (15페이지씩 분할 처리) |
| DOCX | python-docx 단락 추출 |
| TXT/MD | 인코딩 자동 감지 + 이미지 URL 자동 추출 (Gemini Vision) |
| 이미지 (JPG/PNG 등) | Gemini Vision API로 텍스트 추출 |

- 이미지/OCR 결과는 `.image_cache.json`에 MD5 해시 기반으로 캐싱
- MuPDF 경고는 `fitz.TOOLS.mupdf_display_errors(False)`로 stderr 출력 억제, `fitz.TOOLS.mupdf_warnings()`로 파일별 경고 수집 후 출력

### 5.5 ConversationMemory (`src/memory.py`)

- 인메모리 딕셔너리로 세션별 대화 히스토리 관리
- 최대 `MAX_CONVERSATION_TURNS`(기본 5) 턴 유지
- 서버 재시작 시 초기화됨

---

## 6. 데이터 파이프라인

### 6.1 문서 데이터 형식

각 게시글은 `data/documents/{post_id}/` 폴더에 저장됩니다:

```
data/documents/9888/
├── data.json                    # 메타데이터 (필수)
├── 본문파일.md                   # 메인 문서
├── 첨부파일1.pdf                 # 첨부
├── 첨부파일2.pdf                 # 첨부
└── .image_cache.json            # 이미지 OCR 캐시 (자동 생성)
```

**data.json 구조:**
```json
{
  "id": 9888,
  "title": "게시글 제목",
  "category": "medical",         // "medical" 또는 "event"
  "year": 2024,
  "main_file": "본문파일.md",
  "attachments": ["첨부파일1.pdf", "첨부파일2.pdf"]
}
```

### 6.2 인덱싱 명령어

```bash
# 증분 인덱싱 (새로 추가/삭제된 게시글만 처리)
make index

# 전체 재인덱싱
python scripts/index_documents.py --full

# 특정 게시글 추가
python scripts/index_documents.py --add 12345

# 특정 게시글 삭제
python scripts/index_documents.py --delete 12345

# BM25 인덱스만 재빌드
make bm25
```

### 6.3 인덱싱 흐름

```
data/documents/{post_id}/
        │
        ▼
  DocumentLoader.load_file()
  (PDF→텍스트/OCR, DOCX→단락, MD→텍스트+이미지)
        │
        ▼
  DocumentChunker.chunk_documents()
  (medical: 섹션→문장 / event: 재귀적 분할)
        │
        ▼
  EmbeddingService.embed_batch()
  (Vertex AI, 배치 5개, 재시도 6회)
        │
        ├──────────────────────┐
        ▼                      ▼
  VectorStore.upsert()    ChunkStore.save_chunks()
  (Vector Search 업로드)   (chunks.json 저장)
        │
        ▼
  BM25Index.build()
  (전체 청크로 재빌드, bm25_index.pkl 저장)
```

### 6.4 저장소 구조

| 저장소 | 위치 | 형식 | 용도 |
|--------|------|------|------|
| Vector Search | GCP (클라우드) | 768차원 벡터 + 메타데이터 | 시맨틱 유사도 검색 |
| BM25 인덱스 | `data/bm25_index/bm25_index.pkl` | pickle | 키워드 검색 |
| 청크 스토어 | `data/chunk_store/chunks.json` | JSON | 청크 ID → 텍스트/메타데이터 매핑 |

---

## 7. API 엔드포인트

### POST /chat

채팅 질의응답 (SSE 스트리밍).

**요청:**
```json
{
  "query": "결핵 치료 원칙을 알려줘",
  "session_id": "uuid (선택, 미전송 시 자동 생성)",
  "source_type_filter": "medical | event | null",
  "stream": true
}
```

**응답 (SSE 스트림):**
```
data: {"type": "token", "content": "결핵의"}
data: {"type": "token", "content": " 치료 원칙은..."}
data: {"type": "sources", "data": [{"source_file": "...", "source_type": "medical", ...}]}
data: {"type": "done", "session_id": "uuid"}
```

### GET /admin/health

헬스체크. `{"status": "ok"}` 반환.

### DELETE /admin/session/{session_id}

세션 대화 기록 삭제.

### GET /admin/download/{filename}

첨부파일 다운로드. `data/documents/` 하위 전체에서 `rglob`으로 파일을 검색하여 반환.

---

## 8. 프론트엔드 (`static/index.html`)

단일 HTML 파일, 프레임워크 없이 Vanilla JS로 구현.

- **마크다운 렌더링**: marked.js (del 태그 비활성화하여 `~` 기호 보존)
- **스트리밍**: `fetch` + `ReadableStream`으로 SSE 파싱 (항상 스트리밍 모드 사용)
- **다운로드 링크**: LLM이 `[[다운로드:파일명]]` 형식으로 출력 → `rewriteDownloadMarkers()` 함수가 `encodeURIComponent` 적용한 마크다운 링크로 변환 → marked.js가 렌더링
- **소스 필터**: 드롭다운으로 전체/의료/행사 필터링
- **소스 뱃지**: 의료(초록)/행사(파랑) 뱃지 + 관련도 점수 표시

---

## 9. 개발 환경 세팅

```bash
# 1. 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate
make dev    # 프로덕션 + 개발 의존성 모두 설치

# 2. GCP 인증
gcloud auth application-default login

# 3. 환경변수 설정
cp .env.example .env
# .env 편집: GCP_PROJECT_ID, VERTEX_INDEX_ID, VERTEX_ENDPOINT_ID 등

# 4. (최초 1회) Vector Search 인덱스 생성
make setup
# 출력된 index_id, endpoint_id를 .env에 입력

# 5. 문서 인덱싱
make index

# 6. 서버 실행
fuser -k 8000/tcp 2>/dev/null; sleep 2
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
# → http://localhost:8000 에서 접속
```

### 주요 Make 커맨드

| 커맨드 | 설명 |
|--------|------|
| `make install` | 프로덕션 의존성 설치 |
| `make dev` | 프로덕션 + 개발 의존성 설치 |
| `make setup` | Vector Search 인덱스/엔드포인트 생성 |
| `make index` | 증분 인덱싱 |
| `make bm25` | BM25 인덱스 재빌드 |
| `make test` | 테스트 실행 |
| `make lint` | 린트 검사 |
| `make format` | 코드 포맷팅 |
| `make clean` | 캐시 파일 정리 |

---

## 10. 테스트

```bash
# 전체 테스트
make test

# 특정 파일
pytest tests/test_chunker.py -v

# 커버리지
pytest tests/ --cov=src --cov-report=term-missing
```

| 테스트 파일 | 대상 | 외부 API |
|------------|------|---------|
| `test_chunker.py` | 청킹 로직 (의료/행사, 섹션 분할) | 없음 |
| `test_hybrid_retriever.py` | RRF 퓨전 공식 | 없음 |
| `test_pipeline.py` | 대화 메모리 관리 | 없음 |

> 현재 테스트는 외부 API 호출 없이 순수 로직만 검증합니다. 통합 테스트는 미구현 상태입니다.

---

## 11. 알려진 이슈 및 주의사항

### 에러 핸들링 부족

- **`_stream_response()` (chat.py:49-55)**: try/except 없음. 스트리밍 중 예외 발생 시 200 OK 후 스트림이 끊기며 프론트엔드에 "답변을 생성하지 못했습니다"가 표시됨. 실제 에러는 uvicorn 터미널에서만 확인 가능.
- **`query_stream()` (rag_pipeline.py:93-147)**: 동일하게 에러 핸들링 없음.

### 파일명 특수문자

- 다운로드 링크에서 괄호/공백이 포함된 파일명 처리를 위해 `[[다운로드:파일명]]` 마커 형식을 사용합니다. 기존 마크다운 링크 형식 `[텍스트](URL)`은 괄호가 포함된 파일명에서 깨짐.

### MuPDF 경고

- 일부 PDF에서 "syntax error: invalid key in dict" 경고가 발생합니다. PDF 내부 메타데이터 문제로, 텍스트 추출에는 영향 없습니다. 경고는 파일명과 함께 출력되도록 처리되어 있음.

### Vertex AI 일시 장애

- 임베딩 API (`embeddings.py`)에는 지수 백오프 재시도(최대 6회)가 구현되어 있음.
- Gemini 스트리밍(`generator.py`)에는 재시도가 없음.

---

## 12. 고도화 시 참고사항

### 새 문서 카테고리 추가

1. `data/documents/{post_id}/data.json`의 `category`에 새 값 사용
2. `static/index.html`: 필터 드롭다운에 `<option>` 추가, CSS 뱃지 색상 추가, 소스 뱃지 조건 추가
3. (선택) `src/chunker.py`: 새 카테고리 전용 청킹 전략 추가. 미추가 시 `_chunk_event`(범용)로 처리됨
4. 나머지 (vectorstore, bm25, retriever, pipeline): 문자열 기반이므로 수정 불필요

### 스트리밍 에러 가시화

`chat.py`의 `_stream_response()`에 try/except 추가하여 에러를 SSE 이벤트로 전달:

```python
async def _stream_response(pipeline, request):
    try:
        for event in pipeline.query_stream(...):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
```

프론트엔드에서 `event.type === 'error'` 처리 추가.

### 검색 품질 튜닝 포인트

| 파라미터 | 설정 위치 | 영향 |
|---------|----------|------|
| `HYBRID_ALPHA` | .env | 벡터(시맨틱) vs BM25(키워드) 가중치. 높을수록 시맨틱 우선 |
| `RERANK_CANDIDATES` | .env | 리랭킹 전 후보 수. 높을수록 recall↑ 비용↑ |
| `RERANK_TOP_K` | .env | 리랭킹 후 결과 수. LLM 컨텍스트 크기와 연관 |
| `MAX_CONTEXT_TOKENS` | .env | LLM에 전달할 컨텍스트 길이 제한 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | .env | 의료 문서 청크 크기 (행사는 고정 5000자) |

### 주요 GCP 서비스 콘솔 경로

| 서비스 | 콘솔 경로 |
|--------|----------|
| Vector Search | Vertex AI → Vector Search → 인덱스/엔드포인트 |
| Document AI | Document AI → 프로세서 |
| Discovery Engine Ranking | 콘솔 UI 없음. API만 활성화하면 사용 가능 (`discoveryengine.googleapis.com`) |