## 2. 시스템 아키텍처

```
                                ┌─────────────────────────────┐
                                │  static/index.html          │
                                │  (fetch + ReadableStream)   │
                                └─────────────┬───────────────┘
                                              │ POST /chat (SSE)
                                              ▼
            ┌─────────────────────────────────────────────────────────┐
            │  FastAPI app (api/main.py)                              │
            │  ├ middleware: CORS + request log (api/middleware.py)   │
            │  └ routes: chat / search / admin (api/routes/*)         │
            └─────────────────────────┬───────────────────────────────┘
                                      ▼
            ┌─────────────────────────────────────────────────────────┐
            │  RAGPipeline (src/rag_pipeline.py)                      │
            │   1. Memory.get_history()                               │
            │   2. _normalize_relative_years()  ←   상대시간           │
            │   3. LLMGenerator.rewrite_query()  (history 있을 때)     │ 
            │   4. HybridRetriever.retrieve()                         │
            │   5. LLMGenerator.build_context()                       │
            │   6. LLMGenerator.generate_stream()                     │
            │   7. Memory.add_turn() + sources emit                   │
            └─────┬───────────────────────────────────┬───────────────┘
                  │                                   │
        ┌─────────▼──────────┐              ┌─────────▼─────────┐
        │ HybridRetriever    │              │ LLMGenerator      │
        │ (BM25 + Vector +   │              │ (Vertex Gemini    │
        │  RRF + Rerank)     │              │  스트리밍)        │
        └─────┬──────┬───────┘              └─────────┬─────────┘
              │      │                                │
   ┌──────────▼──┐ ┌─▼──────────────┐    ┌────────────▼─────────┐
   │ BM25Index   │ │ VectorStore    │    │ Vertex AI Gemini API │
   │ (kiwipiepy  │ │ (Firestore     │    │                      │
   │  + rank_bm25│ │  Vector Search)│    └──────────────────────┘
   │  pkl)       │ └────────────────┘
   └─────────────┘

         ┌────────────┐                  ┌─────────────────┐
         │ Redis      │ ◄────────────────│ ConversationMem │
         │ (memory +  │                  │ + EmbeddingCache│
         │  embed     │                  └─────────────────┘
         │  cache)    │
         └────────────┘
```

오프라인 인덱싱(별도 흐름):
```
data/ ─► document_loader ─► chunker ─► embeddings ─► vectorstore.upsert()
                                                  └─► chunk_store.save_chunks()
                                                  └─► bm25_index.build/save()
```

---

## 3. 인덱싱 파이프라인 (오프라인)

진입점:
- `scripts/index_documents.py` — 전체/증분 인덱싱 (Firestore + 청크 스토어 + BM25)
- `scripts/build_bm25_index.py` — BM25 인덱스만 재구축 (Firestore 의존 없음)

### 3.1 문서 로딩 (`src/document_loader.py:40` `DocumentLoader`)

| 포맷 | 처리 방식 |
|---|---|
| `.pdf` | Google **Document AI** OCR (페이지 ≤15는 온라인, 그 이상은 청크 분할 후 처리). `_load_pdf()` (L245) |
| `.jpg/.png/.webp/.gif/.bmp` | **Gemini Vision**으로 이미지 → 텍스트, 결과 캐싱. `_load_image()` (L345) |
| `.docx`, `.doc` | python-docx 단순 추출. `_load_docx()` (L403) |
| `.txt`, `.md` | 그대로 read. `_load_text()` (L414) |

`load_directory()` (L155)는 디렉터리의 `data.json` 메타파일을 기준으로 게시글(post) 단위로 메인 파일 + 첨부파일을 함께 적재. 각 청크는 다음 메타데이터를 운반함: `post_id`, `post_title`, `year`, `attachments`, `page_number`, `source_type` (`medical` | `event`).

### 3.2 청킹 전략 (`src/chunker.py:9` `DocumentChunker`)

`source_type`에 따라 다른 전략을 적용 (`chunk_document()` L29):

- **의료 문서 (`_chunk_medical()` L42)**: 진단/검사결과/처방 같은 섹션 헤더(`MEDICAL_HEADERS`)로 1차 분할 후, kiwipiepy로 문장 경계 분할 → 1600자/240자 overlap으로 그룹핑. 페이지 번호 보존.
- **행사 문서 (`_chunk_event()` L74)**: LangChain `RecursiveCharacterTextSplitter` (5000자, 300자 overlap, 단락 → 문장 → 마침표 순으로 재귀 분할).

### 3.3 임베딩 (`src/embeddings.py:32` `EmbeddingService`)

- 모델: `text-multilingual-embedding-002` (768차원, Vertex AI).
- 단일/배치 호출 모두 지원 (`embed()` L89, `embed_batch()` L107, batch_size=5).
- **Redis 캐싱**: 텍스트 SHA256 해시 키, TTL 7일(`embedding_cache_ttl`). `task_type`(`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`)도 키에 포함하여 분리.
- 재시도(`_embed_with_retry()` L49): 429 quota는 20s+ 대기, 일반 5xx는 지수 백오프.

### 3.4 인덱스 저장

- **VectorStore (`src/vectorstore.py:18`)** — Firestore Vector Search 컬렉션 `medical_event_chunks`. `upsert()` (L32)는 batch=500으로 commit. `search()` (L89)는 `find_nearest()` cosine 거리로 KNN, score = 1.0 − distance.
- **BM25Index (`src/bm25_index.py:16`)** — kiwipiepy 토큰화(명사/동사/형용사 + 불용어 14개) + `rank_bm25.BM25Okapi`. `./data/bm25_index/bm25_index.pkl`로 pickle 저장.
- **ChunkStore (`src/chunk_store.py:6`)** — 검색 결과 메타 보강용. `./data/chunk_store/chunks.json`에 chunk_id → {text, source_file, post_id, year, ...} 맵 저장.

같은 청크가 세 군데(Firestore 임베딩, BM25 pkl, ChunkStore JSON)에 동기화돼야 함 — 인덱싱 스크립트가 한 번에 셋 다 갱신.

---

## 4. 쿼리 파이프라인 (온라인)

진입점: `api/routes/chat.py:14` `chat()` → `_stream_response()` (L49) → `RAGPipeline.query_stream()` (`src/rag_pipeline.py:102`).

### 4.1 단계별

```
question
  │
  ▼ ① Memory.get_history(session_id)        ── Redis 조회 (최근 5턴)
  │
  ▼ ② _normalize_relative_years()           ── "올해"→"2026년" 결정론적 치환
  │   (rag_pipeline.py:30, 최근 추가)
  │
  ▼ ③ LLMGenerator.rewrite_query()  *if history*
  │   (generator.py:300)
  │   - 오늘 날짜 주입
  │   - 대명사 해소 ("그것", "거기서" 등)
  │   - 단, 새 주제면 이전 맥락 적용 X
  │
  ▼ ④ HybridRetriever.retrieve()
  │   (hybrid_retriever.py:50)
  │
  ▼ ⑤ LLMGenerator.build_context()          ── 출처 [1], [2]... 헤더 + 본문
  │   (generator.py:117)
  │   각 청크 헤더에 [연도: YYYY], 페이지, 첨부파일 포함
  │
  ▼ ⑥ LLMGenerator.generate_stream()
  │   (generator.py:203)
  │   - system_instruction에 오늘 날짜 + RAG 규칙 주입
  │   - history(최근 6턴) + user_message(컨텍스트 + 질문)
  │   - 토큰 yield → SSE
  │
  ▼ ⑦ Memory.add_turn() + sources/done emit
  │   (rag_pipeline.py:165)
```

### 4.2 핵심: HybridRetriever (`src/hybrid_retriever.py:15`)

`retrieve()` (L50):
1. `_hybrid_search()` (L109): VectorStore 검색 + BM25Index 검색 **병렬 실행**.
2. `_reciprocal_rank_fusion()` (L256): RRF 스코어 = `1/(K + rank + 1)`, K=60. `hybrid_alpha=0.6`로 vector 60% / BM25 40% 가중.
3. `_rerank()` (L140): 상위 `rerank_candidates=50`개를 GCP Discovery Engine Ranking API로 재순위 → `rerank_top_k=10`개로 추림.
4. `_expand_by_post()` (L210): 같은 게시물(post_id) 내 추가 청크가 있으면 컨텍스트 확장.
5. **최신성 의도 감지** 시 `_RECENCY_ALPHA=0.7`로 연도 가중치 부스트, post_id 단위 중복 제거(이벤트당 1청크만 LLM 제시).

### 4.3 핵심: LLMGenerator (`src/generator.py`)

#### `rewrite_query()` (L300)
- 입력: 사용자 질문 + 최근 4턴 히스토리.
- 출력: 독립 해석 가능한 질문.
- 프롬프트 상단에 `오늘 날짜: YYYY-MM-DD (현재 연도: YYYY년)` 주입(최근 추가).
- 규칙 5종 (대명사 해소 / 상대시간 절대연도 변환 / 시간 강제 추가 금지 / 새 주제 분리 / 출력은 질문만).
- temperature=0.1, max_output_tokens=200, thinking_budget=0.

#### `generate_stream()` (L203)
- `_with_retry()`로 stream 생성 단계 5회 재시도 (429/5xx/timeout).
- Stream 자체는 `async for chunk` 순회, `chunk.text`만 yield.
- `[generate-stream] OK | chunks=N` 로깅 vs 빈 응답 시 `EMPTY response` 경고 + `finish_reason`/`block_reason` 기록.

#### `_config()` (L109) `system_instruction`
1줄 헤더: `오늘 날짜: YYYY-MM-DD` (매 요청 동적 주입, 최근 추가)
이어서 13개 규칙: 컨텍스트 외 정보 금지 / 정보 없을 시 검색 맥락 명시 / 출처 표기 금지 / 존댓말 / 의료 정확성 / 간결 / 핵심 요약 / 청크 종합 / 숫자 범위 `~` 표기 / 첨부파일 `[[다운로드:파일명]]` 형식 / 빈 셀 `,` 금지 / 최신순 정렬 시 `[연도: YYYY]` 사용 / 정렬·내림차순·날짜순 등 명시 시 전체 정렬.

#### `build_context()` (L117)
각 청크를 다음 형태로 직렬화:
```
[1] 출처: 파일명.md (게시글: ...) [연도: 2026], 페이지 3
본문 텍스트
첨부파일: 신청서.pdf, 양식.docx
```

---

## 5. API / 서빙 레이어

### 5.1 lifespan & DI (`api/main.py`, `api/dependencies.py`)

`lifespan()` (`main.py:16`): 앱 시작 시 Redis ping → `get_pipeline()` 초기화 → **warmup query** 1회 실행 → ready. 종료 시 redis.aclose().

DI는 `@lru_cache` 싱글턴 (`api/dependencies.py`):
- `get_redis()`, `get_settings()`, `get_embedding_service()`, `get_vectorstore()`, `get_bm25_index()`, `get_chunk_store()`, `get_retriever()`, `get_generator()`, `get_memory()`, `get_pipeline()`.
- `BM25Index.load()`와 `ChunkStore.load()`는 첫 호출 시 디스크에서 로드.

### 5.2 라우트

| Method | Path | 용도 |
|---|---|---|
| POST | `/chat` | 대화 (`stream:true`면 SSE, 아니면 JSON). `api/routes/chat.py:14` |
| POST | `/search` | 검색만 (LLM 호출 없이 retriever 결과 반환). `api/routes/search.py` |
| GET | `/admin/health` | 헬스체크 |
| DELETE | `/admin/session/{id}` | 세션 메모리 비우기 |
| GET | `/admin/download/{filename}` | 데이터 디렉터리 파일 다운로드 |
| GET | `/` | `static/index.html` 서빙 |

### 5.3 SSE 스트리밍 (`api/routes/chat.py:49`)

`StreamingResponse(media_type="text/event-stream", X-Accel-Buffering=no)`로 nginx 버퍼링 비활성. 이벤트 타입 3종:
- `{"type":"token","content":"..."}` — LLM 토큰 조각
- `{"type":"sources","data":[...]}` — 검색 출처(`source_file`, `source_type`, `page_number`, `relevance_score`)
- `{"type":"done","session_id":"..."}` — 종료 신호

프런트엔드(`static/index.html:471`)는 `res.body.getReader()`로 chunk 단위 디코딩, `data: ...` 라인 파싱 후 token은 누적 표시, sources는 종료 후 렌더, done에서 session_id 갱신.

### 5.4 에러 처리 (`src/rag_pipeline.py:15`)

- 429(quota) 에러 → `QUOTA_FALLBACK_MESSAGE` ("현재 요청이 몰려…").
- 그 외 예외 → `GENERIC_FALLBACK_MESSAGE` ("답변 생성 중 오류…").
- Setup(retrieve) 실패와 stream 실패를 구분해 로깅(`logger.exception`).

---

## 6. 메모리 & 세션 (`src/memory.py:8` `ConversationMemory`)

- 저장소: Redis JSON (`chat:session:{uuid}` 키).
- 구조: `[{role: "user"|"model", content: ...}, ...]`
- 보관 정책: 최근 `max_turns=5` 턴(=10 메시지)만 유지, TTL 1800s(30분).
- API: `create_session()`, `get_history()`, `add_turn()`, `clear_session()`.

세션 ID는 클라이언트가 보관(`session_id` 응답 → 다음 요청 헤더/바디). 첫 요청 시 서버가 발급.

---

## 7. 설정 & 환경 (`src/config.py`, `.env.example`)

`Settings` (Pydantic BaseSettings, `.env` 자동 로드):

| 카테고리 | 키 | 기본값 / 비고 |
|---|---|---|
| GCP | `gcp_project_id`, `gcp_location` | `asia-northeast3` |
| 모델 | `embedding_model`, `llm_model`, `llm_location` | embed `text-multilingual-embedding-002`, llm `gemini-2.5-flash` (`.env`로 오버라이드) |
| OCR | `docai_processor` | Document AI processor 경로 |
| Vector | `firestore_database_id`, `firestore_collection_name` | `(default)`, `medical_event_chunks` |
| RAG | `chunk_size`, `chunk_overlap`, `top_k`, `bm25_top_k`, `hybrid_alpha`, `score_threshold`, `max_context_tokens` | 800/120/7/10/0.6/0.65/4000 |
| Rerank | `rerank_candidates`, `rerank_top_k` | 50/10 |
| 대화 | `max_conversation_turns` | 5 |
| Redis | `redis_url`, `embedding_cache_ttl` | None / 7d |

`get_settings()`는 `@lru_cache`로 1회 로드.

---


### 9.1 로컬 개발

```
make install     # 의존성 설치
make setup       # GCP 인증 (scripts/setup_gcp.py)
make index       # 문서 인덱싱 (전체)
make bm25        # BM25 인덱스만 재구축
make run         # uvicorn --reload (포트 8000)
```

### 9.2 Docker

```
make docker-up   # infra/docker-compose.yml — app + redis 함께 기동
make docker-down
```

`infra/Dockerfile` — `python:3.12-slim` + `pip install -r requirements.txt` + `COPY . .` + `uvicorn`.
컨테이너에 코드를 굽기 때문에 코드 변경 시 `docker compose up -d --build` 필요.
