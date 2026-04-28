# Medical & Event RAG Chatbot - 인수인계서

> 최종 업데이트: 2026-04-24

---

## 1. 프로젝트 개요

의료 문서(지침, 공문 등)와 행사 문서(학술대회, 워크샵 등)를 기반으로 질의응답하는 **RAG(Retrieval-Augmented Generation) 챗봇**입니다.

### 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | FastAPI + Uvicorn (전 구간 async I/O) |
| LLM | Gemini (google-genai SDK, Vertex AI, `client.aio` 비동기 호출) |
| Embedding | Vertex AI `text-multilingual-embedding-002` (768차원) |
| Vector DB | **Firestore Vector Search (serverless, 쿼리당 과금)** |
| Reranking | Discovery Engine Ranking API |
| BM25 | rank-bm25 + kiwipiepy (한국어 형태소 분석) |
| OCR | Document AI (스캔 PDF), Gemini Vision (이미지) |
| Session / Cache | **Redis (필수)** — 세션 대화 히스토리 + 임베딩 Redis 캐시 |
| Frontend | Vanilla JS + marked.js (단일 HTML) |
| Infra | docker-compose (app + Redis), Cloud Run 배포 가능 |

---

## 2. 디렉토리 구조

```
medical-event-rag/
├── api/                          # FastAPI 애플리케이션
│   ├── main.py                   # lifespan(Redis ping + 워밍업), 라우트/미들웨어 등록
│   ├── middleware.py             # CORS, 로깅 미들웨어
│   ├── dependencies.py           # 싱글턴 의존성 팩토리 (@lru_cache, get_redis 포함)
│   └── routes/
│       ├── chat.py               # POST /chat (SSE 스트리밍)
│       ├── search.py             # POST /search (리트리버 단독 호출)
│       └── admin.py              # GET /admin/health, DELETE /admin/session, GET /admin/download
│
├── src/                          # 핵심 비즈니스 로직
│   ├── config.py                 # pydantic-settings 기반 설정
│   ├── models.py                 # 데이터 모델 (Request/Response, DocumentChunk 등)
│   ├── rag_pipeline.py           # RAG 파이프라인 오케스트레이터 (async, fallback 처리)
│   ├── hybrid_retriever.py       # 3단계 검색: 하이브리드 → 리랭킹 → 확장 + 정렬 인텐트 (async)
│   ├── generator.py              # Gemini LLM 응답 생성 (async, 재시도 + 429 백오프)
│   ├── embeddings.py             # 임베딩 서비스 (async, Redis 캐시 + 재시도)
│   ├── vectorstore.py            # Firestore Vector Search (async, find_nearest 기반)
│   ├── bm25_index.py             # BM25 인덱스 (한국어 형태소 토큰화)
│   ├── chunk_store.py            # 청크 ID → 메타데이터 매핑 (JSON)
│   ├── chunker.py                # 문서 청킹 (의료: 계층적, 행사: 재귀적)
│   ├── document_loader.py        # 문서 로더 (PDF/DOCX/MD/TXT/이미지)
│   └── memory.py                 # 대화 히스토리 (Redis 필수, async)
│
├── scripts/                      # CLI 스크립트
│   ├── setup_gcp.py              # (레거시) Vector Search 인덱스/엔드포인트 생성 — Firestore 전환 후 미사용
│   ├── index_documents.py        # 문서 인덱싱 (Firestore upsert 기반, async)
│   ├── migrate_to_firestore.py   # chunks.json → Firestore 일회성 마이그레이션 (OCR 건너뛰기)
│   ├── build_bm25_index.py       # BM25 인덱스 재빌드
│   └── load_test.py              # 부하/격리 테스트 스크립트
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
├── infra/                        # 인프라/배포
│   ├── Dockerfile                # 앱 이미지
│   ├── docker-compose.yml        # app + redis + healthcheck + gcloud ADC 마운트
│   └── cloudbuild.yaml           # Cloud Run 배포 구성
│
├── docs/                         # 문서
│   ├── HANDOVER.md               # 본 문서
│   └── CODE_REFERENCE.md         # 코드 레퍼런스
│
├── .env                          # 환경변수 (git 제외)
├── .env.example                  # 환경변수 템플릿
├── pytest.ini                    # pytest 설정 (asyncio_mode=auto)
├── Makefile                      # 개발 커맨드
├── requirements.txt              # 프로덕션 의존성
└── requirements-dev.txt          # 개발 의존성 (pytest-asyncio, fakeredis 포함)
```

---

## 3. 환경 설정

### 3.1 사전 요구사항

- Python 3.12+
- GCP 프로젝트 (**Vertex AI, Document AI, Discovery Engine, Firestore API** 활성화)
- `gcloud` CLI 인증: `gcloud auth application-default login`
- **Redis** — 로컬 개발은 `docker compose up` 이 자동 제공, 운영은 Memorystore 등 관리형 Redis 필요
- **Firestore Native DB** — asia-northeast3 에 1회 생성 (`gcloud firestore databases create`). 벡터 인덱스 2개 (vector-only, source_type+vector) 도 함께 설정.
- Docker / Docker Compose (로컬 실행 권장 경로)

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
| `DOCAI_PROCESSOR` | **O** | Document AI OCR 프로세서 리소스 경로 (`projects/{number}/locations/{loc}/processors/{id}`) |
| `FIRESTORE_DATABASE_ID` | - | Firestore DB ID (기본: `(default)`) |
| `FIRESTORE_COLLECTION_NAME` | - | 벡터 저장 컬렉션 이름 (기본: medical_event_chunks) |
| `VERTEX_INDEX_ID` | (legacy) | Vector Search 인덱스 — Firestore 전환 후 미사용, rollback 용 |
| `VERTEX_ENDPOINT_ID` | (legacy) | Vector Search 엔드포인트 — 현재 undeploy 상태 |
| `VERTEX_COLLECTION_NAME` | (legacy) | Matching Engine 컬렉션 이름 |
| `CHUNK_SIZE` | - | 의료 문서 청크 크기 (기본: 800) |
| `CHUNK_OVERLAP` | - | 청크 오버랩 (기본: 120) |
| `TOP_K` | - | 최종 검색 결과 수 (기본: 7) |
| `BM25_TOP_K` | - | BM25 후보 수 (기본: 10) |
| `HYBRID_ALPHA` | - | 벡터 검색 가중치, 0~1 (기본: 0.6) |
| `RERANK_CANDIDATES` | - | 리랭킹 전 후보 수 (기본: 50) |
| `RERANK_TOP_K` | - | 리랭킹 후 상위 결과 수 (기본: 10) |
| `MAX_CONTEXT_TOKENS` | - | LLM 컨텍스트 최대 토큰 (기본: 4000) |
| `MAX_CONVERSATION_TURNS` | - | 대화 히스토리 유지 턴 수 (기본: 5) |
| `REDIS_URL` | **O** | Redis 연결 URL (예: `redis://localhost:6379/0`). 미설정 시 앱이 기동 실패 |
| `EMBEDDING_CACHE_TTL` | - | 임베딩 Redis 캐시 TTL 초 (기본: 604800 = 7일) |

> `FIRESTORE_*` 는 Firestore DB 를 **asia-northeast3** 에 미리 생성한 뒤 기본값으로 두면 됩니다. 상세 셋업은 9.1 참고.
>
> `VERTEX_*` 항목은 Firestore 전환 후 더 이상 사용되지 않습니다. 되돌릴 필요가 있을 때만 다시 설정하세요.
>
> `REDIS_URL`은 **필수**입니다. docker-compose로 띄울 때는 `infra/docker-compose.yml`이 `REDIS_URL=redis://redis:6379/0`을 자동으로 주입하므로 `.env`에 적는 값은 무시됩니다.

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
RAGPipeline.query_stream()   (전 구간 async, 예외 시 fallback 토큰)│
  │                                                              │
  ├─ 1. 세션 관리: session_id 생성 또는 조회                        │
  │                                                              │
  ├─ 2. 대화 히스토리 로드 (Redis)                                  │
  │                                                              │
  ├─ 3. 쿼리 리라이팅 (멀티턴인 경우)                                │
  │     "그거 자세히" → "결핵 진료지침 4판 내용을 자세히 알려줘"         │
  │                                                              │
  ├─ 4. 검색 (HybridRetriever) ──────────────────────┐           │
  │     ├─ Stage 1: 하이브리드 검색 (임베딩 & BM25 병렬)│           │
  │     │   ├─ 임베딩 (Redis 캐시 hit 시 즉시 반환)     │           │
  │     │   ├─ Firestore find_nearest (async, 50개)    │           │
  │     │   ├─ BM25 (to_thread, 50개)                  │           │
  │     │   └─ RRF 퓨전 (가중 합산, 50개)              │           │
  │     ├─ Stage 2: 리랭킹 (Discovery Engine, to_thread)│           │
  │     │   └─ 시맨틱 관련도 상위 10개                 │           │
  │     ├─ Stage 3: 게시글 확장                        │           │
  │     │   └─ 같은 post_id의 추가 청크 포함 (post당 5)│           │
  │     └─ Stage 3.5 (옵션): 정렬 인텐트 감지 시        │           │
  │         └─ post_id dedupe + year DESC stable sort  │           │
  │                                                    │           │
  ├─ 5. 응답 생성 (Gemini 스트리밍, async + 지수 백오프 재시도)     │
  │     컨텍스트 + 질문 → LLM → 토큰 스트림                        │
  │     (429/5xx/네트워크 재시도, 최종 실패 시 fallback 토큰)       │
  │                                                              │
  ├─ 6. 대화 히스토리 Redis 저장 (첫 토큰 성공 시에만)              │
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
get_redis()                          → redis.asyncio.Redis 싱글턴 (REDIS_URL 미설정 시 RuntimeError)
get_pipeline()
  ├── get_retriever()
  │     ├── get_embedding_service()  → Vertex AI 임베딩 (+ Redis 캐시 주입)
  │     ├── get_vectorstore()        → Firestore VectorStore (async client, 싱글턴)
  │     ├── get_bm25_index()         → BM25 인덱스 파일 로드
  │     └── get_chunk_store()        → chunks.json 로드
  ├── get_generator()                → Gemini 클라이언트 (async + retry)
  └── get_memory()                   → ConversationMemory (Redis 필수)
```

`api/main.py:lifespan` 에서:
- 기동 시: `get_redis().ping()` 으로 Redis 연결 확인 → 실패 시 앱 기동 중단
- 워밍업: `await pipeline.query(question="테스트", session_id="warmup")` (실패는 무시)
- 종료 시: `await redis.aclose()` 로 연결 정리

---

## 5. 핵심 컴포넌트 상세

### 5.1 HybridRetriever (`src/hybrid_retriever.py`)

**3단계 + 선택적 재정렬 파이프라인 (모두 async):**

| 단계 | 메서드 | 설명 |
|------|--------|------|
| Stage 1 | `_hybrid_search()` | 임베딩 호출과 BM25 검색을 `asyncio.gather` 로 병렬 실행 → Vector Search 호출 → RRF 퓨전. `HYBRID_ALPHA`(기본 0.6)로 벡터 검색 가중치 |
| Stage 2 | `_rerank()` | Discovery Engine Ranking API 를 `asyncio.to_thread` 로 호출. 실패 시 RRF 순서 폴백 |
| Stage 3 | `_expand_by_post()` | 검색된 청크와 같은 `post_id` 청크 추가 (게시글당 최대 5개) |
| Stage 3.5 | `_prepare_for_sort()` | 쿼리에서 **정렬 인텐트** 감지 시 (`_SORT_PATTERN`) post_id 로 dedupe + year 내림차순 stable sort |

**정렬 인텐트 패턴 `_SORT_PATTERN`:**
```
정렬 | 내림차순 | 오름차순 | 날짜순 | 일자순 | 순서대로 | 순서로 | 나열 | 리스트 | 목록
```
감지되면 rerank 풀도 확장 (`recency_boost=True`) 되어 후보가 더 많이 포함됩니다.

**RRF(Reciprocal Rank Fusion) 공식:**
```
score(doc) = α × 1/(K + rank_vector) + (1-α) × 1/(K + rank_bm25)
K = 60, α = HYBRID_ALPHA
```

### 5.2 LLMGenerator (`src/generator.py`)

- **시스템 프롬프트**: 한국어 전용, 컨텍스트 기반 답변 규칙 13개. 첨부파일 다운로드 링크 형식 `[[다운로드:파일명]]`, 정렬 요청 처리 규칙(#13) 포함
- **비동기 호출**: 모든 Gemini 호출은 `client.aio.models.*` 사용 (`async def`)
- **스트리밍**: `generate_stream()` — `async for chunk in ...`, 각 chunk의 `.text` 를 yield
- **연속 생성**: `generate()` — MAX_TOKENS 로 끊기면 최대 3회 이어서 생성
- **쿼리 리라이팅**: `rewrite_query()` — 대명사 해소, 맥락 보존
- **재시도 + 백오프** (`_with_retry` 공용 헬퍼):
  - 대상: `ServerError`(5xx), `httpx.TimeoutException`, `httpx.ConnectError`, `ClientError(code=429)`
  - 기본 백오프: 초기 2초, 지수 증가 + 25% jitter, 최대 5회
  - 429(쿼터 초과)는 최소 20초로 baseline 연장 — Vertex AI 쿼터는 분 단위 회복
  - 최종 실패 시 호출자가 `_fallback_message()` 로 친절한 메시지 생성

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

- **Redis 필수** (`redis.asyncio.Redis` 주입). 생성자에 `None` 전달 시 즉시 `ValueError`
- 세션별 대화 히스토리를 `chat:session:{session_id}` 키로 저장, TTL 1800초 (30분)
- 최대 `MAX_CONVERSATION_TURNS`(기본 5) 턴 유지 (양 초과 시 오래된 메시지부터 제거)
- 모든 메서드 async (`get_history`, `add_turn`, `clear_session`). `create_session()`만 sync (UUID 생성)
- 멀티 인스턴스 배포에서도 세션이 공유됨 — Cloud Run / GKE 수평 확장 가능

### 5.5b VectorStore (`src/vectorstore.py`) — Firestore Vector Search

- **Firestore Native mode** 컬렉션에 각 청크를 document 로 저장. `embedding` 필드는 Firestore `Vector` 타입 (768차원)
- **find_nearest KNN** 쿼리로 상위 N 반환. `COSINE` 거리 → `1.0 - distance` 로 similarity score 환산
- **source_type 필터** 지원 (`FieldFilter`)
- 모든 호출이 async. `google.cloud.firestore.AsyncClient` 사용
- **과금 구조**: 쿼리당 document reads (예: top_k=10 → 10 reads ≈ $0.0006). 유휴 시 $0.
- **인덱스 2개 필수**:
  1. `embedding` 단독 vector index (차원 768, flat)
  2. `source_type + embedding` 복합 vector index (필터 검색용)
- `upsert` 는 500개 단위 배치 write. `remove` 로 ID 기반 삭제 지원.

### 5.6 EmbeddingService (`src/embeddings.py`) — Redis 캐시

- 모든 임베딩 호출이 async. `self.client.aio.models.embed_content(...)` 사용
- 쿼리 임베딩은 Redis 캐시 확인 후 API 호출: 키 `emb:{sha256(task_type:model:text)}`, TTL `EMBEDDING_CACHE_TTL` (기본 7일)
- 캐시 hit 시 Vertex AI 호출 건너뜀 → 반복 쿼리에서 지연·쿼터 모두 절감
- `embed_batch()`는 인덱싱 전용이므로 캐시하지 않음 (각 문서가 유일)
- 재시도 로직은 generator와 동일 (429 포함 5회, 20초+ baseline)

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
# 증분 인덱싱 (새로 추가/삭제된 게시글만 처리, Firestore 업서트)
make index
# 또는: docker exec infra-app-1 python scripts/index_documents.py

# 전체 재인덱싱 (문서 로드 + OCR + Firestore 업서트 + BM25 재빌드)
docker exec infra-app-1 python scripts/index_documents.py --full

# 특정 게시글 추가
docker exec infra-app-1 python scripts/index_documents.py --add 12345

# 특정 게시글 삭제
docker exec infra-app-1 python scripts/index_documents.py --delete 12345

# BM25 인덱스만 재빌드
make bm25

# chunks.json 이 이미 있는 상태에서 Firestore 만 채우기 (OCR 건너뜀, ~10분)
docker exec infra-app-1 python scripts/migrate_to_firestore.py --clear
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
  (Firestore batch write)  (chunks.json 저장)
        │
        ▼
  BM25Index.build()
  (전체 청크로 재빌드, bm25_index.pkl 저장)
```

### 6.4 저장소 구조

| 저장소 | 위치 | 형식 | 용도 |
|--------|------|------|------|
| **Firestore Vector Search** | GCP (asia-northeast3, serverless) | 문서당 embedding Vector + 메타데이터 | 시맨틱 유사도 검색 |
| BM25 인덱스 | `data/bm25_index/bm25_index.pkl` | pickle | 키워드 검색 |
| 청크 스토어 | `data/chunk_store/chunks.json` | JSON | 청크 ID → 텍스트/메타데이터 매핑 (Firestore 와 중복이지만 빠른 로컬 enrichment 용으로 유지) |

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

**오류 시 fallback 토큰:**
- 429 쿼터 초과 (최종 실패): `"현재 요청이 몰려 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."`
- 그 외 예외 / 빈 스트림: `"답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."`
- 검색 결과 0건: `"제공된 문서에서 관련 정보를 찾을 수 없습니다."`

### POST /search

리트리버 단독 호출 (LLM 호출 없이 검색 결과만). 디버깅/튜닝용.

```json
{
  "query": "행사 일정",
  "top_k": 10,
  "source_type_filter": "event"
}
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

### 9.1 docker-compose (권장)

`.env` 만 준비되면 한 번에 기동합니다. Redis·ADC·환경변수 모두 자동 처리:

```bash
# 0. (최초 1회) Firestore DB + vector index 생성
gcloud firestore databases create \
    --project=$GCP_PROJECT_ID \
    --location=asia-northeast3 \
    --type=firestore-native

# 단일 vector index (필터 없는 검색용)
gcloud firestore indexes composite create \
    --project=$GCP_PROJECT_ID --database="(default)" \
    --collection-group=medical_event_chunks --query-scope=COLLECTION \
    --field-config=vector-config='{"dimension":"768","flat":"{}"}',field-path=embedding

# 복합 vector index (source_type 필터 검색용)
gcloud firestore indexes composite create \
    --project=$GCP_PROJECT_ID --database="(default)" \
    --collection-group=medical_event_chunks --query-scope=COLLECTION \
    --field-config=order=ASCENDING,field-path=source_type \
    --field-config=vector-config='{"dimension":"768","flat":"{}"}',field-path=embedding
# 두 index 모두 READY 상태가 될 때까지 대기 (수 분~십여 분)

# 1. GCP 인증
gcloud auth application-default login

# 2. 환경변수 설정
cp .env.example .env
# .env 편집: GCP_PROJECT_ID, GCP_LOCATION 등. Firestore 관련은 기본값으로 OK.
# (REDIS_URL 은 compose 가 자동 주입하므로 값 무시됨)

# 3. 기동 (app + redis + healthcheck)
docker compose -f infra/docker-compose.yml up --build
# → http://localhost:8000
```

`infra/docker-compose.yml` 의 역할:
- `REDIS_URL=redis://redis:6379/0` env 강제 주입
- `~/.config/gcloud` 를 `/root/.config/gcloud` 로 마운트 → ADC 자동 인식
- Redis healthcheck 통과 후 앱 기동 (`depends_on.condition: service_healthy`)

### 9.2 bare-metal (로컬 Python)

```bash
# 1. 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate
make dev    # 프로덕션 + 개발 의존성 모두 설치

# 2. GCP 인증
gcloud auth application-default login

# 3. 환경변수 설정
cp .env.example .env
# .env 편집: GCP_PROJECT_ID, VERTEX_INDEX_ID, VERTEX_ENDPOINT_ID, REDIS_URL(필수)

# 4. 로컬 Redis 실행 (별도 터미널)
redis-server
# 또는 Docker 로: docker run --rm -p 6379:6379 redis:7-alpine

# 5. (최초 1회) Vector Search 인덱스 생성
make setup
# 출력된 index_id, endpoint_id를 .env에 입력

# 6. 문서 인덱싱 (async embed 를 내부에서 asyncio.run 으로 래핑)
make index

# 7. 서버 실행
fuser -k 8000/tcp 2>/dev/null; sleep 2
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
# → http://localhost:8000
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
| `test_pipeline.py` | 대화 메모리 (Redis, 세션 격리, fail-fast) | `fakeredis.aioredis` |

- `pytest.ini` 는 `asyncio_mode=auto` — async 테스트가 자동 인식됨
- `tests/conftest.py` 에 `fake_redis` 픽스처가 `fakeredis.aioredis.FakeRedis` 를 제공하므로 실제 Redis 없이 테스트 가능
- 통합 테스트(실제 Vertex AI 호출)는 여전히 미구현. 수동 검증은 `/search`, `/chat` 호출로 수행

---

## 11. 알려진 이슈 및 주의사항

### 에러 핸들링 (갖춰진 상태)

- **`query_stream()` (rag_pipeline.py)**: setup 구간과 스트림 iteration 모두 `try/except` 로 감싸져 있습니다. 예외 발생 시 `_fallback_message()` 가 429 여부를 보고 적절한 한국어 메시지를 스트림 토큰으로 yield 합니다 (`QUOTA_FALLBACK_MESSAGE` / `GENERIC_FALLBACK_MESSAGE`).
- **`_stream_response()` (chat.py)**: 파이프라인이 항상 토큰을 yield 하도록 보장되므로 라우트 계층의 try/except 는 불필요.
- **빈 스트림 (토큰 0개) 방어**: `full_answer == []` 인 경우에도 `GENERIC_FALLBACK_MESSAGE` 를 yield 해서 프론트엔드가 빈 응답으로 멈추지 않도록 함.
- **진단 로그**: `generator.generate_stream` 이 종료 시 한 줄로 `chunks/empty/first_chunk/total/finish_reason/block_reason/ctx_len/query` 를 INFO/WARNING 로 남깁니다. 빈 응답이 발생하면 `docker logs --tail 50 infra-app-1 | grep generate-stream` 로 cold-start, safety filter, quota, finish_reason 비정상 종료 등을 즉시 식별 가능. retrieve 가 빈 결과면 `[query_stream] NO documents retrieved` WARNING 도 함께 남음.

### 파일명 특수문자

- 다운로드 링크에서 괄호/공백이 포함된 파일명 처리를 위해 `[[다운로드:파일명]]` 마커 형식을 사용합니다. 기존 마크다운 링크 형식 `[텍스트](URL)`은 괄호가 포함된 파일명에서 깨짐.

### MuPDF 경고

- 일부 PDF에서 "syntax error: invalid key in dict" 경고가 발생합니다. PDF 내부 메타데이터 문제로, 텍스트 추출에는 영향 없습니다. 경고는 파일명과 함께 출력되도록 처리되어 있음.

### Vertex AI 일시 장애 / 쿼터

- 임베딩(`embeddings.py`)과 Gemini(`generator.py`) 모두 **지수 백오프 재시도 + 429 처리** 구현.
  - retryable: `ServerError`(5xx), `httpx.TimeoutException`, `httpx.ConnectError`, `ClientError(code=429)`
  - 429는 baseline 20초 (일반 2초) — 쿼터는 분 단위 회복
  - 최대 5~6회 재시도 후 최종 실패 시 상위 레이어에 raise → fallback 토큰으로 사용자 응답
- **장기 해결**: GCP 콘솔에서 Vertex AI 쿼터 증설 신청 (이번 범위 외). 운영 규모에 따라 embedding QPM / Gemini generateContent QPM / Discovery Engine rank QPS 를 함께 증설해야 함.

### 비용 구조 (2026-04-24 이후)

| 리소스 | 과금 모델 | 유휴 시 | 1000 쿼리/일 |
|--------|-----------|---------|--------------|
| Firestore Vector Search | 쿼리당 document reads | $0 | ~$0.2 |
| Gemini 2.5 Flash | input/output token 단위 | $0 | ~$0.5 |
| Vertex AI Embedding (Redis 캐시 후) | 토큰 단위 (캐시 hit 시 과금 없음) | $0 | <$0.1 |
| Discovery Engine Rank | API 호출당 | $0 | <$2 |
| **월 총합 예상** (1000 req/일) | | | **~$5** |

- **이전(Vertex AI Vector Search 2 replica 24/7) 대비 비용 99.7% 절감** ($1,600/월 → $5/월)
- 중요한 사건 기록: 2026-04-01 에 `automaticResources(min=2, max=2)` 로 Matching Engine endpoint 배포, 23일간 누적 ~$1,200 발생. 2026-04-24 undeploy 후 Firestore 로 이관.
- 일일 $50 이상 발생 중인 리소스는 월 $1,500 이상을 의미. 반드시 **GCP 결제 예산 알림** 을 걸어둘 것.

### 집계/정렬형 질문의 본질적 한계

- "모든 행사를 날짜 내림차순으로 정렬" 같은 질문은 RAG top-k 구조상 완전한 정렬을 보장하기 어려움.
- 완화: `_SORT_PATTERN` 감지 시 post_id dedupe + year 내림차순 pre-sort (Stage 3.5). LLM에는 system prompt 규칙 #13 로 "컨텍스트 순서 무시하고 모든 항목 정렬" 을 강제.
- 여전히 동일 연도 내에서 월/일 정렬은 LLM 책임. 정확도가 부족하면 `thinking_budget` 활성화(현재 0) 검토.

---

## 12. 고도화 시 참고사항

### 새 문서 카테고리 추가

1. `data/documents/{post_id}/data.json`의 `category`에 새 값 사용
2. `static/index.html`: 필터 드롭다운에 `<option>` 추가, CSS 뱃지 색상 추가, 소스 뱃지 조건 추가
3. (선택) `src/chunker.py`: 새 카테고리 전용 청킹 전략 추가. 미추가 시 `_chunk_event`(범용)로 처리됨
4. 나머지 (vectorstore, bm25, retriever, pipeline): 문자열 기반이므로 수정 불필요

### 스트리밍 에러 가시화 (이미 적용됨)

현재 `src/rag_pipeline.py.query_stream()` 에서 예외를 포착해 `type: "token"` 으로 한국어 fallback 메시지를 yield 합니다. 에러 타입을 프론트엔드에 별도로 노출하려면 다음과 같이 확장 가능:

```python
# rag_pipeline.py
except Exception as exc:
    yield {"type": "error", "code": getattr(exc, "code", None), "message": _fallback_message(exc)}
    yield {"type": "done", "session_id": session_id}
```

프론트엔드에서 `event.type === 'error'` 분기 추가 시 429/기타 구분하여 UX 차별화 가능.

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