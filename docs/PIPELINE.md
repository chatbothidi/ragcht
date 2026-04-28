# 챗봇 파이프라인 파일 매핑

> 요청이 들어와서 응답이 나가는 실제 파이프라인 순서대로 어떤 파일이 사용되는지 정리.
>
> 최종 업데이트: 2026-04-24

---

## 1. 파이프라인 파일 흐름도

```
[User]
  │ POST /chat  (SSE 스트리밍 기본)
  ▼
┌──────────────────────────────────────────────────┐
│ 1. API Entry                                     │
│    api/main.py            ← FastAPI 앱 + lifespan │
│    api/middleware.py      ← CORS, 요청 로깅      │
│    api/routes/chat.py     ← /chat 라우터         │
│    api/dependencies.py    ← 싱글턴 팩토리         │
└───────────────────┬──────────────────────────────┘
                    │ await pipeline.query_stream(...)
                    ▼
┌──────────────────────────────────────────────────┐
│ 2. Orchestrator                                  │
│    src/rag_pipeline.py    ← query_stream 전체 지휘│
│    src/models.py          ← 데이터 타입 (dataclass)│
└──┬────────────┬──────────────────────────┬───────┘
   │            │                          │
   ▼            ▼                          ▼
[Memory]    [Retrieval]                 [Generation]
┌──────┐  ┌─────────────────┐           ┌────────────────┐
│ 3.   │  │ 4. Retrieval    │           │ 5. Generation  │
│ src/ │  │ src/hybrid_     │           │ src/generator  │
│memory│  │ retriever.py    │           │ .py (async)    │
│.py   │  │ (async, 3-stage)│           │                │
│      │  │                 │           │                │
│Redis │  │ ├─ embed        │           │ Gemini async   │
│async │  │ │  src/         │           │ + retry(429)   │
│      │  │ │  embeddings.py│           │ + fallback     │
│      │  │ │  (Redis cache)│           │                │
│      │  │ ├─ vector       │           │                │
│      │  │ │  src/         │           │                │
│      │  │ │  vectorstore  │           │                │
│      │  │ │  .py(Firestore│           │                │
│      │  │ │  find_nearest)│           │                │
│      │  │ ├─ BM25         │           │                │
│      │  │ │  src/bm25_    │           │                │
│      │  │ │  index.py     │           │                │
│      │  │ ├─ enrichment   │           │                │
│      │  │ │  src/chunk_   │           │                │
│      │  │ │  store.py     │           │                │
│      │  │ └─ rerank       │           │                │
│      │  │   Discovery     │           │                │
│      │  │   Engine (SDK만)│           │                │
│      │  └─────────────────┘           └────────────────┘
└──────┘
```

---

## 2. 단계별 사용 파일 매핑

### 2.1 API 진입 (FastAPI)

| 파일 | 역할 | 호출 시점 |
|------|------|-----------|
| `api/main.py` | 앱 부트스트랩, `lifespan` 에서 Redis ping + 워밍업 | 서버 기동 시 |
| `api/middleware.py` | CORS + structlog 요청/응답 로깅 | 모든 요청 |
| `api/routes/chat.py` | `/chat` 엔드포인트, `StreamingResponse` 생성 | 매 요청 |
| `api/routes/search.py` | `/search` 엔드포인트 (retriever 단독 호출) | /search 요청 |
| `api/routes/admin.py` | `/admin/health`, `/admin/session`, `/admin/download` | 관리 요청 |
| `api/dependencies.py` | `get_redis`, `get_pipeline` 등 싱글턴 팩토리 (`@lru_cache`) | 최초 1회 |

### 2.2 오케스트레이터

| 파일 | 역할 |
|------|------|
| `src/rag_pipeline.py` | `RAGPipeline.query_stream()` — 세션 → 리라이팅 → 검색 → 생성 → 저장 전 과정 총괄. 예외 발생 시 `_fallback_message()` 로 한국어 토큰 yield |
| `src/config.py` | `Settings` (pydantic-settings), `.env` 로드. Firestore + Redis + 모델명 등 |
| `src/models.py` | `ChatRequest`, `DocumentChunk`, `RAGResponse`, `SourceCitation` 등 |

### 2.3 세션 메모리 (Redis)

| 파일 | 역할 |
|------|------|
| `src/memory.py` | `ConversationMemory.get_history/add_turn/clear_session`. `redis.asyncio` 클라이언트 사용. TTL 30분, 세션당 최대 5턴 |

**호출**: `await memory.get_history(session_id)` → ... → `await memory.add_turn(sid, q, a)` (응답 저장 시)

### 2.4 검색 (HybridRetriever, 3-stage + 정렬 인텐트)

| 파일 | 역할 | 호출 순서 |
|------|------|-----------|
| `src/hybrid_retriever.py` | 3-stage 오케스트레이터 + 정렬 인텐트 (`_SORT_PATTERN`, `_prepare_for_sort`) | 진입 |
| `src/embeddings.py` | `embed(query)` — Redis 캐시 확인 후 Vertex AI `text-multilingual-embedding-002` 호출 | Stage 1 (병렬) |
| `src/bm25_index.py` | `search(query)` — kiwipiepy 토큰화 + rank_bm25 | Stage 1 (병렬, `to_thread`) |
| `src/vectorstore.py` | `search(vec)` — Firestore `find_nearest` (async) | Stage 1 |
| `src/chunk_store.py` | `get_chunk(id)` — chunks.json 로드된 메모리 dict 에서 enrichment | Stage 1 후처리 |
| (외부) Discovery Engine SDK | `_rerank` — `semantic-ranker-default@latest` 호출 (hybrid_retriever.py 내부에서 `to_thread`) | Stage 2 |

→ 최종 결과: `list[dict]` — id, score, text, metadata 포함

### 2.5 응답 생성

| 파일 | 역할 |
|------|------|
| `src/generator.py` | `rewrite_query(query, history)` (멀티턴 시) + `generate_stream(q, context, history)` (메인 스트리밍). Gemini 2.5 Flash async, retry/backoff, 429 fallback |

**호출 순서**:
1. `rewrite_query` — 히스토리 있으면 Gemini 호출 → 쿼리 재작성
2. `build_context(documents)` — 검색 결과를 프롬프트용 텍스트로 조립
3. `generate_stream` — `client.aio.models.generate_content_stream(...)` → async for 로 토큰 yield

---

## 3. 외부 서비스 · SDK 사용 위치

| 외부 서비스 | 사용 파일 | SDK | 용도 |
|-----------|-----------|-----|------|
| Vertex AI Gemini 2.5 Flash | `src/generator.py` | `google.genai` (async `client.aio`) | 응답 생성, 쿼리 리라이팅 |
| Vertex AI Embedding | `src/embeddings.py` | `google.genai` (async) | 쿼리·문서 임베딩 |
| Firestore Vector Search | `src/vectorstore.py` | `google.cloud.firestore.AsyncClient` | KNN 검색, upsert, delete |
| Discovery Engine Rank | `src/hybrid_retriever.py` | `google.cloud.discoveryengine_v1` (sync → `asyncio.to_thread`) | 시맨틱 재랭킹 |
| Redis | `src/memory.py`, `src/embeddings.py` | `redis.asyncio` | 세션 + 임베딩 캐시 |

---

## 4. 데이터 파일 (디스크·메모리 로컬)

| 파일 | 사용처 | 용도 |
|------|-------|------|
| `data/chunk_store/chunks.json` | `src/chunk_store.py` | 16,737 청크 메타데이터 (기동 시 메모리 로드) |
| `data/bm25_index/bm25_index.pkl` | `src/bm25_index.py` | BM25 인덱스 pickle (기동 시 메모리 로드) |
| `data/documents/{post_id}/**` | `src/document_loader.py` | 원본 PDF/MD/이미지 (인덱싱 시에만) |
| `.env` | `src/config.py` | GCP 프로젝트, 모델, Firestore 이름 등 |

---

## 5. 실제 요청 1건 처리 시 파일 호출 트레이스

사용자가 `POST /chat` (stream=true, session 있음) 날렸을 때:

```
1. api/main.py:lifespan              ← 이미 실행됨 (startup)
2. api/middleware.py:log_requests    ← 요청 진입
3. api/routes/chat.py:chat           ← 라우트 매칭
4. api/dependencies.py:get_pipeline  ← @lru_cache → 기존 인스턴스 반환
5. api/routes/chat.py:_stream_response ← StreamingResponse 시작
6. src/rag_pipeline.py:query_stream  ← 오케스트레이션 시작
   ├─ 6-1. src/memory.py:get_history          ← Redis GET
   ├─ 6-2. src/generator.py:rewrite_query     ← Gemini 호출 (history 있으면)
   ├─ 6-3. src/hybrid_retriever.py:retrieve
   │       ├─ asyncio.gather:
   │       │   ├─ src/embeddings.py:embed        ← Redis 캐시 확인 → Gemini 임베딩
   │       │   └─ src/bm25_index.py:search       ← to_thread
   │       ├─ src/vectorstore.py:search          ← Firestore find_nearest
   │       ├─ src/chunk_store.py:get_chunk       ← 로컬 dict 조회
   │       ├─ src/hybrid_retriever.py:_rerank    ← Discovery Engine rank API
   │       ├─ src/hybrid_retriever.py:_expand_by_post
   │       └─ src/hybrid_retriever.py:_prepare_for_sort (정렬 인텐트 시)
   ├─ 6-4. src/generator.py:build_context       ← 컨텍스트 조립 (순수 문자열)
   ├─ 6-5. src/generator.py:generate_stream     ← Gemini 스트리밍
   │       (각 청크가 yield → api/routes/chat.py → SSE 로 클라이언트에 전달)
   └─ 6-6. src/memory.py:add_turn               ← Redis SETEX (TTL 30분)
7. 스트림 종료 (done 이벤트 yield)
8. api/middleware.py:log_requests             ← 응답 로깅
```

---

## 6. 인덱싱 파이프라인 (서빙과 분리된 배치)

실시간 요청 처리와 별개로, 문서를 Firestore 에 올릴 때의 파이프라인:

```
data/documents/{post_id}/*
  │  (data.json: id, title, category, main_file, attachments, year)
  ▼
src/document_loader.py           ← PDF/DOCX/MD/이미지 로더
  ├─ PyMuPDF (PDF 텍스트)
  ├─ Document AI OCR (스캔 PDF, 15페이지 단위 분할)
  ├─ Gemini Vision (이미지)
  └─ chardet + marked.js (MD/TXT)
  ▼
src/chunker.py                   ← 카테고리별 청킹
  ├─ medical: 섹션 헤더 → kiwipiepy 문장 분할 → 그룹핑 (~1600자)
  └─ event/기타: RecursiveCharacterTextSplitter (5000자)
  ▼
src/embeddings.py.embed_batch    ← Vertex AI 임베딩 (async, 5개 배치)
  ▼
src/vectorstore.py.upsert        ← Firestore batch write (500개 단위)
  ▼
src/chunk_store.py.save_chunks   ← data/chunk_store/chunks.json 쓰기
  ▼
src/bm25_index.py.build + save   ← data/bm25_index/bm25_index.pkl 쓰기
```

**인덱싱 실행 스크립트**:

| 스크립트 | 용도 |
|----------|------|
| `scripts/index_documents.py` | 전체(`--full`) / 증분(기본) / 개별(`--add`, `--delete`) 인덱싱 |
| `scripts/migrate_to_firestore.py` | chunks.json 직접 → Firestore (OCR 건너뛰는 빠른 경로) |
| `scripts/build_bm25_index.py` | BM25 인덱스만 재빌드 |

---

## 7. 테스트·검증 파일

| 파일 | 용도 |
|------|------|
| `tests/conftest.py` | `settings` fixture + `fake_redis` (fakeredis.aioredis) fixture |
| `tests/test_pipeline.py` | ConversationMemory (세션 격리, fail-fast) async 테스트 |
| `tests/test_hybrid_retriever.py` | RRF 퓨전 순수 로직 테스트 |
| `tests/test_chunker.py` | 청킹 로직 테스트 (의료/행사 섹션 분할) |
| `scripts/load_test.py` | 부하 테스트 (concurrency, isolation 모드) |
| `pytest.ini` | `asyncio_mode=auto` 설정 |

---

## 8. 인프라 구성 파일

| 파일 | 용도 |
|------|------|
| `infra/Dockerfile` | Python 3.12-slim + uvicorn |
| `infra/docker-compose.yml` | app + redis + healthcheck + gcloud ADC 마운트 + `PYTHONUNBUFFERED` |
| `infra/cloudbuild.yaml` | Cloud Run 배포 구성 (Google Cloud Build) |
| `requirements.txt` | 프로덕션 의존성 (google-genai, google-cloud-firestore, redis 등) |
| `requirements-dev.txt` | pytest-asyncio, fakeredis 등 개발용 |

---

## 9. 핵심 트레이드오프 요약

| 설계 결정 | 장점 | 트레이드오프 |
|-----------|------|--------------|
| `asyncio.gather` (embed + BM25 병렬) | 응답 지연 감소 | 코드 복잡도 ↑ |
| Redis 임베딩 캐시 | 반복 쿼리 Vertex 호출 $0 | 메모리 사용 ↑, TTL 관리 |
| Firestore Vector Search (Matching Engine 대체) | 서버리스, 월 $5 | cold start 첫 쿼리 100~300ms |
| BM25 로컬 pickle + chunk_store JSON | 무료 메모리 검색, 빠름 | 재인덱싱 시 파일 동기 필요 |
| Discovery Engine Rank (외부 API) | 품질 좋은 rerank | 호출 비용, 쿼터 제한 |
| Gemini 2.5 Flash `thinking_budget=0` | 빠른 응답 | 복잡한 추론 품질 ↓ |
| `@lru_cache` 싱글턴 | 초기화 1회 | 테스트 시 cache clear 필요 |
