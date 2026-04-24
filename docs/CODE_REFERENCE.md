# 코드 상세 레퍼런스

각 소스 파일의 클래스, 메서드, 파라미터, 동작 원리를 정리한 문서입니다.

---

## 목차

1. [src/config.py](#1-srcconfigpy) — 설정
2. [src/models.py](#2-srcmodelspy) — 데이터 모델
3. [src/document_loader.py](#3-srcdocument_loaderpy) — 문서 로더
4. [src/chunker.py](#4-srcchunkerpy) — 문서 청킹
5. [src/embeddings.py](#5-srcembeddingspy) — 임베딩
6. [src/vectorstore.py](#6-srcvectorstorepy) — Vector Search
7. [src/bm25_index.py](#7-srcbm25_indexpy) — BM25 인덱스
8. [src/chunk_store.py](#8-srcchunk_storepy) — 청크 스토어
9. [src/hybrid_retriever.py](#9-srchybrid_retrieverpy) — 하이브리드 검색
10. [src/generator.py](#10-srcgeneratorpy) — LLM 응답 생성
11. [src/memory.py](#11-srcmemorypy) — 대화 메모리
12. [src/rag_pipeline.py](#12-srcrag_pipelinepy) — RAG 파이프라인
13. [api/main.py](#13-apimainpy) — 앱 진입점
14. [api/middleware.py](#14-apimiddlewarepy) — 미들웨어
15. [api/dependencies.py](#15-apidependenciespy) — 의존성 주입
16. [api/routes/chat.py](#16-apirouteschatpy) — 채팅 API
17. [api/routes/search.py](#17-apiroutessearchpy) — 검색 API (리트리버 단독)
18. [api/routes/admin.py](#18-apiroutesadminpy) — 관리 API
19. [scripts/index_documents.py](#19-scriptsindex_documentspy) — 인덱싱 스크립트
20. [scripts/migrate_to_firestore.py](#20-scriptsmigrate_to_firestorepy) — Firestore 이관 스크립트
21. [infra/docker-compose.yml](#21-infradocker-composeyml) — 로컬 기동 구성

---

## 1. `src/config.py`

**역할**: `.env` 파일에서 환경변수를 읽어 타입-안전한 설정 객체를 생성합니다.

### `Settings` (BaseSettings)

pydantic-settings 기반. `.env` 파일을 자동으로 읽습니다.

| 필드 | 타입 | 기본값 | 용도 |
|------|------|--------|------|
| `gcp_project_id` | str | (필수) | GCP 프로젝트 ID |
| `gcp_location` | str | "asia-northeast3" | GCP 리전 |
| `embedding_model` | str | "text-multilingual-embedding-002" | 임베딩 모델명 |
| `llm_model` | str | "gemini-2.0-flash" | LLM 모델명 |
| `llm_location` | str \| None | None | LLM 전용 리전 (미설정 시 `gcp_location` 사용) |
| `vertex_collection_name` | str | "medical_event_docs" | Vector Search 컬렉션명 |
| `vertex_index_id` | str \| None | None | Vector Search 인덱스 리소스 ID |
| `vertex_endpoint_id` | str \| None | None | Vector Search 엔드포인트 리소스 ID |
| `chunk_size` | int | 800 | 의료 문서 청크 크기 (토큰 기준) |
| `chunk_overlap` | int | 120 | 청크 간 오버랩 (토큰 기준) |
| `top_k` | int | 7 | 최종 반환 결과 수 |
| `bm25_top_k` | int | 10 | BM25 검색 반환 수 |
| `hybrid_alpha` | float | 0.6 | RRF에서 벡터 검색 가중치 (1에 가까울수록 벡터 우선) |
| `score_threshold` | float | 0.65 | (현재 미사용) |
| `max_context_tokens` | int | 4000 | LLM에 전달할 컨텍스트 최대 토큰 |
| `rerank_candidates` | int | 50 | 리랭킹 전 후보 수 |
| `rerank_top_k` | int | 10 | 리랭킹 후 상위 결과 수 |
| `max_conversation_turns` | int | 5 | 대화 히스토리 최대 턴 수 |
| `redis_url` | str \| None | None | **필수**. Redis 주소. 미설정 시 `get_redis()` 호출 시점에 `RuntimeError`, `ConversationMemory` 생성 시 `ValueError` |
| `embedding_cache_ttl` | int | 604800 | 임베딩 Redis 캐시 TTL (초, 기본 7일) |
| `firestore_database_id` | str | "(default)" | Firestore 데이터베이스 ID |
| `firestore_collection_name` | str | "medical_event_chunks" | 벡터 저장 컬렉션 이름 |
| `vertex_*` | (legacy) | - | 이전 Matching Engine 관련. Firestore 전환 후 미사용 |

### `get_settings()`

`@lru_cache` 데코레이터로 싱글턴. 한 번만 `.env`를 파싱합니다.

---

## 2. `src/models.py`

**역할**: 프로젝트 전반에서 사용되는 데이터 구조를 정의합니다.

### Request 모델 (Pydantic)

| 모델 | 필드 | 용도 |
|------|------|------|
| `ChatRequest` | `query`, `session_id?`, `source_type_filter?`, `stream` | POST /chat 요청 |
| `SearchRequest` | `query`, `top_k`, `source_type_filter?` | POST /search 요청 |

### 내부 데이터 모델 (dataclass)

**`LoadedDocument`** — 파일에서 로드된 문서 한 건

| 필드 | 타입 | 설명 |
|------|------|------|
| `text` | str | 추출된 전체 텍스트 |
| `source_file` | str | 원본 파일명 |
| `source_type` | str | 카테고리 ("medical" / "event") |
| `pages` | list[dict] \| None | 페이지별 텍스트 (PDF OCR 시) |
| `metadata` | dict \| None | post_id, post_title, attachments 등 |

**`DocumentChunk`** — 청킹된 텍스트 조각

| 필드 | 타입 | 설명 |
|------|------|------|
| `text` | str | 청크 텍스트 |
| `source_file` | str | 원본 파일명 |
| `source_type` | str | 카테고리 |
| `chunk_index` | int | 문서 내 청크 순번 |
| `page_number` | int \| None | 추정 페이지 번호 |
| `section_title` | str \| None | 의료 문서 섹션 제목 (예: "진단") |
| `metadata` | dict \| None | 게시글 메타데이터 |

**`SourceCitation`** — 응답에 포함되는 출처 정보

| 필드 | 타입 | 설명 |
|------|------|------|
| `source_file` | str | 출처 파일명 |
| `source_type` | str | 카테고리 |
| `page_number` | int \| None | 페이지 번호 |
| `relevance_score` | float | 관련도 점수 |
| `excerpt` | str | 텍스트 미리보기 (150자) |

**`RAGResponse`** — RAG 파이프라인 최종 결과

| 필드 | 타입 | 설명 |
|------|------|------|
| `answer` | str | LLM 생성 답변 |
| `sources` | list[SourceCitation] | 출처 목록 |
| `query` | str | 원본 질문 |
| `rewritten_query` | str \| None | 리라이팅된 질문 |
| `session_id` | str | 세션 ID |

---

## 3. `src/document_loader.py`

**역할**: 다양한 포맷의 파일을 텍스트로 변환합니다.

### `DocumentLoader`

#### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `SUPPORTED_EXTENSIONS` | `.pdf, .docx, .doc, .txt, .md` | 지원 문서 확장자 |
| `IMAGE_EXTENSIONS` | `.jpg, .jpeg, .png, .gif, .webp, .bmp` | 지원 이미지 확장자 |
| `DOCAI_PROCESSOR` | 프로세서 리소스 경로 | Document AI OCR 프로세서 ID |
| `CACHE_FILENAME` | `.image_cache.json` | 게시글 폴더별 캐시 파일명 |

#### 주요 메서드

**`load_file(file_path, source_type, post_id?, post_title?, attachments?)`**
- 파일 확장자에 따라 적절한 로더를 선택하여 `LoadedDocument` 반환
- MuPDF 경고를 파일명과 함께 출력
- 게시글 메타데이터(post_id, post_title, attachments)를 문서에 추가

**`load_directory(directory)`**
- `data/documents/` 하위의 게시글 폴더를 순회
- 각 폴더의 `data.json`에서 메타데이터를 읽고 main_file + attachments를 로드
- 이미지 캐시를 폴더별로 관리 (로드 → 처리 → 저장)
- `data.json`이 없으면 레거시 구조(`medical/events` 폴더)로 폴백

**`_load_pdf(path, source_type)`**
1. 캐시 확인 → 히트 시 즉시 반환
2. 캐시 미스 → Document AI OCR 실행
3. 15페이지 이하: 전체를 한 번에 OCR (`_ocr_pdf_bytes`)
4. 15페이지 초과: 15페이지씩 분할 OCR (`_ocr_pdf_chunked`)
5. 유니코드 정규화 후 캐시 저장

**`_load_image(path, source_type)`**
1. 캐시 확인 → 히트 시 즉시 반환
2. Gemini Vision API로 이미지에서 텍스트 추출
3. 표는 마크다운 표로 변환, 일반 텍스트는 그대로 추출
4. 테이블 구분선 정규화 (`_normalize_table_separators`) 후 캐시 저장

**`_load_text(path, source_type)`**
1. `chardet`으로 인코딩 자동 감지
2. 마크다운 내 이미지 URL (`![alt](url)`) 자동 탐지
3. 각 이미지 URL에 대해 캐시 확인 → 미스 시 Gemini Vision으로 텍스트 추출
4. 추출된 텍스트를 `## 이미지에서 추출된 내용` 섹션으로 원본 텍스트에 추가

**`_load_docx(path, source_type)`**
- `python-docx`로 단락 추출, 빈 단락 제외

#### 캐싱 시스템

| 메서드 | 설명 |
|--------|------|
| `_load_image_cache(post_dir)` | `.image_cache.json` 로드 |
| `_save_image_cache()` | 변경된 캐시만 디스크에 저장 (`_cache_dirty` 플래그) |
| `_get_cached_text(key, data)` | MD5 해시 비교로 캐시 유효성 확인 |
| `_set_cache(key, data, text)` | 해시 + 추출 텍스트 + 타임스탬프 저장 |

#### 텍스트 정규화

**`_normalize_text(text)`** — 유니코드 특수문자 치환

| 원본 | 치환 | 설명 |
|------|------|------|
| `∼`, `〜`, `～` | `~` | 틸다 계열 |
| `−`, `–`, `—` | `-` | 대시 계열 |
| `\u00a0` | 공백 | 비중단 공백 |

**`_normalize_table_separators(text)`** — 마크다운 표 구분선이 300자 이상이면 `---`로 축소

**`_is_low_quality_text(text)`** — PDF 텍스트 추출 품질 판단
- 공백 비율 50% 이상 → 저품질
- 단어 평균 길이 2자 미만 → 저품질

---

## 4. `src/chunker.py`

**역할**: 로드된 문서를 검색에 적합한 크기의 청크로 분할합니다.

### `DocumentChunker`

#### 청킹 전략 분기

```
chunk_document(document)
  ├── source_type == "medical" → _chunk_medical()
  └── 그 외                     → _chunk_event()
```

#### `_chunk_medical(document)` — 의료 문서 전용

1. **섹션 분할** (`_split_by_sections`): `MEDICAL_HEADERS` 정규식으로 섹션 헤더를 찾아 분할
   ```
   대상 헤더: 진단, 소견, 처방, 검사결과, 주소, 병력, 현병력, 과거력,
             가족력, 신체검사, 치료계획, 경과, 수술소견, 퇴원요약,
             간호기록, 투약, 주의사항, 부작용, 용법, 효능, 성분
   ```
2. **문장 분할** (`_split_sentences`): kiwipiepy로 한국어 문장 단위 분리
3. **문장 그룹핑** (`_group_sentences`): 문장을 max_chars(chunk_size * 2 = 1600자) 단위로 묶고, overlap_chars(chunk_overlap * 2 = 240자) 만큼 겹침
4. 각 청크에 `[섹션제목] 텍스트` 형태로 헤더 접두사 추가

#### `_chunk_event(document)` — 범용 (행사 및 기타)

- `RecursiveCharacterTextSplitter` 사용
- 청크 크기: 5000자, 오버랩: 300자
- 구분자 우선순위: `\n\n` → `\n` → `。` → `.` → `!` → `?` → `,` → 공백

#### `_estimate_page(document, chunk_text)`

- 청크 텍스트의 앞 100자를 각 페이지 텍스트와 대조하여 페이지 번호 추정
- PDF OCR 시 생성된 `pages` 정보 활용

---

## 5. `src/embeddings.py`

**역할**: 텍스트를 768차원 벡터로 변환합니다. 모든 호출이 **async** 이며, 쿼리 임베딩은 Redis 캐시를 거칩니다.

### `EmbeddingService`

#### `__init__(settings, redis_client=None)`
- `genai.Client(vertexai=True, ...)` 로 google-genai SDK 초기화 (async는 `self.client.aio.*`)
- `redis_client` 는 쿼리 임베딩 캐시용. `None` 이면 캐시 비활성 (스크립트 용)
- `cache_ttl` 은 `settings.embedding_cache_ttl` 기본 7일

#### `_cache_key(text, task_type)`
- `sha256(f"{task_type}:{model_name}:{text}")` → `emb:{hex}` 키 반환

#### `async embed(text, task_type="RETRIEVAL_QUERY")`
- 단일 텍스트 임베딩 (검색 쿼리용)
- Redis 캐시 조회 → hit 시 즉시 반환, miss 시 API 호출 후 `setex(key, ttl, json.dumps(vec))`
- 내부적으로 `_embed_with_retry()` 호출

#### `async embed_batch(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=5)`
- 다수 텍스트 임베딩 (문서 인덱싱용)
- 5개씩 배치 처리, 캐시 미사용 (인덱싱은 유일한 텍스트이므로)

#### `async _embed_with_retry(contents, task_type, max_attempts=6, initial_delay=2.0)`
- `self.client.aio.models.embed_content(...)` 호출에 지수 백오프 재시도 적용
- 재시도 대상: `genai_errors.ServerError`, `httpx.TimeoutException`, `httpx.ConnectError`, **`genai_errors.ClientError` with `code=429`**
- 429는 baseline 20초로 연장 (일반 2초) — Vertex AI 쿼터 회복 속도 반영
- 딜레이: 지수 증가 + 25% jitter, `await asyncio.sleep(...)` 사용
- 최대 6회 실패 시 최종 예외 raise

#### `_is_retryable(exc)` (모듈 함수)
- 예외가 재시도 대상인지 판단 (429 포함)

---

## 6. `src/vectorstore.py`

**역할**: Firestore Vector Search 를 사용해 벡터를 저장·검색합니다. 모든 호출 async. 이전 Vertex AI Matching Engine 은 2026-04-24 에 undeploy 하고 이 모듈로 대체됨.

### 모듈 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `_BATCH_SIZE` | 500 | Firestore batch write 최대 크기 |

### `VectorStore`

#### 클래스 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `EMBEDDING_DIM` | 768 | text-multilingual-embedding-002 차원 수 |

#### `__init__(settings)`
- `google.cloud.firestore.AsyncClient` 생성 — `settings.gcp_project_id` + `settings.firestore_database_id`
- 컬렉션 이름은 `settings.firestore_collection_name` (기본 `medical_event_chunks`)
- **서버리스**: 별도 deploy/undeploy 없이 바로 사용 가능. 단, 운영 전에 `gcloud firestore indexes composite create` 로 벡터 인덱스를 한 번 만들어야 함.

#### `_collection()` (내부)
- `self.client.collection(self.collection_name)` 반환

#### `async upsert(chunks, embeddings)` → `list[str]`
- chunk 개수와 embedding 개수 불일치 시 `ValueError`
- 각 chunk 당 UUID 를 생성하여 Firestore document ID 로 사용
- `_BATCH_SIZE`(500) 단위로 `AsyncWriteBatch.set(...)` → `await batch.commit()`
- Document 구조:
  ```python
  {
      "text": str,
      "embedding": Vector([768 floats]),   # google.cloud.firestore_v1.vector.Vector
      "source_file": str,
      "source_type": "medical" | "event",
      "chunk_index": int,
      # Optional fields:
      "page_number": int,
      "section_title": str,
      "post_id": int, "post_title": str,
      "year": int,
      "attachments": list[str],
  }
  ```

#### `async search(query_vector, top_k=10, source_type_filter?)` → `list[dict]`
- Firestore 의 `find_nearest()` KNN 쿼리
- `DistanceMeasure.COSINE` 사용, `distance_result_field="distance"` 로 거리 반환 받음
- `source_type_filter` 있으면 `FieldFilter` 로 pre-filter
- 반환 dict 스키마:
  ```python
  {"id": "<doc_id>", "score": 1.0 - distance, **chunk_fields}
  ```
  (embedding 은 응답에서 제거해 클라이언트로 되돌려보내지 않음)

#### `async remove(ids)` → `int`
- 주어진 document ID 리스트를 `_BATCH_SIZE` 단위로 삭제 (`batch.delete`)

#### `async close()`
- AsyncClient 에 `close()` 가 있으면 await 하여 연결 정리

### 필수 Firestore 인덱스 (one-time setup)

쿼리 전에 2개 생성 필요:

1. **Vector-only** (필터 없는 검색용):
   ```
   collection_group=medical_event_chunks
   field: embedding (vector-config: dimension=768, flat)
   ```
2. **source_type + embedding** (필터 검색용):
   ```
   collection_group=medical_event_chunks
   fields:
     - source_type ASCENDING
     - embedding (vector-config: dimension=768, flat)
   ```

둘 다 `gcloud firestore indexes composite create` 로 생성. READY 상태까지 수 분 ~ 십여 분 소요.

---

## 7. `src/bm25_index.py`

**역할**: 한국어 형태소 분석 기반 BM25 키워드 검색 인덱스입니다.

### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `CONTENT_TAGS` | NNG, NNP, NNB, VV, VA, MAG | 추출 대상 형태소 태그 (명사, 동사, 형용사, 부사) |
| `STOPWORDS` | 하다, 있다, 되다, 이다, 것, 수, 등, 및, 또는, 그, 이, 저 | 제거 대상 불용어 |

### `BM25Index`

#### `tokenize(text)`
- kiwipiepy로 형태소 분석
- `CONTENT_TAGS`에 해당하는 형태소만 추출
- `STOPWORDS` 제거
- 예: "결핵 진료 지침을 알려줘" → `["결핵", "진료", "지침", "알리"]`

#### `build(chunks)`
- 전체 청크 텍스트를 토큰화
- `BM25Okapi` 모델 생성
- chunk_ids, chunk_texts, chunk_metadata 저장

#### `search(query, top_k=10, source_type_filter?)`
- 쿼리 토큰화 후 BM25 스코어 계산
- score > 0인 결과만 필터링
- `source_type_filter` 적용
- 스코어 내림차순 정렬 후 top_k 반환

#### 영속화

| 메서드 | 설명 |
|--------|------|
| `save()` | pickle로 `data/bm25_index/bm25_index.pkl`에 저장 |
| `load()` | pickle에서 로드. 파일 없으면 `False` 반환 |

---

## 8. `src/chunk_store.py`

**역할**: Vector Search의 chunk ID와 실제 텍스트/메타데이터를 매핑합니다.

Vector Search는 벡터와 ID만 저장하므로, 원본 텍스트를 가져오려면 이 매핑이 필요합니다.

### `ChunkStore`

#### 저장 형식 (`data/chunk_store/chunks.json`)

```json
{
  "chunk-uuid-1": {
    "text": "청크 텍스트...",
    "source_file": "문서.pdf",
    "source_type": "medical",
    "chunk_index": 0,
    "page_number": 1,
    "section_title": "진단",
    "post_id": 9888,
    "post_title": "게시글 제목",
    "attachments": ["첨부1.pdf"]
  }
}
```

#### 메서드

| 메서드 | 설명 |
|--------|------|
| `save_chunks(chunk_ids, chunks)` | ID-메타데이터 매핑 저장 후 디스크에 영속화 |
| `get_chunk(chunk_id)` | 단일 청크 조회 |
| `get_chunks(chunk_ids)` | 복수 청크 조회 |
| `load()` | `chunks.json`에서 메모리로 로드 |
| `get_indexed_files()` | 인덱싱된 파일명 set 반환 |
| `get_ids_by_source(source_file)` | 특정 파일의 청크 ID 목록 |
| `remove_by_source(source_file)` | 특정 파일의 청크 삭제 + 영속화 |

---

## 9. `src/hybrid_retriever.py`

**역할**: 벡터 검색과 BM25를 결합한 3단계 검색 파이프라인입니다.

### `HybridRetriever`

#### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `RRF_K` | 60 | RRF 공식의 상수 K |
| `_RECENCY_PATTERN` | `최근\|최신\|요즘\|올해\|금년\|이번 해` | 시점 인텐트 감지 (rerank 풀 확장 + year 가중) |
| `_SORT_PATTERN` | `정렬\|내림차순\|오름차순\|날짜순\|일자순\|순서대로\|순서로\|나열\|리스트\|목록` | 정렬/나열 인텐트 감지 (post_id dedupe + year DESC pre-sort) |
| `_RECENCY_ALPHA` | 0.7 | recency 감지 시 year vs 시맨틱 점수 가중치 |

#### `async retrieve(query, top_k?, source_type_filter?)`

전 구간 async. 3단계 + 선택적 재정렬을 순차 실행:

**Stage 1: `async _hybrid_search(query, source_type_filter)`**
```
1. 임베딩 호출(async)과 BM25 검색(to_thread)을 asyncio.gather 로 병렬 실행
2. Vector Search에서 상위 50개 후보 검색 (async, to_thread)
3. ChunkStore에서 각 후보의 텍스트/메타데이터 보강
4. RRF 퓨전으로 두 결과 병합 → 50개 반환
```

**RRF 공식:**
```
score(doc) = α × 1/(K + rank_vector + 1) + (1-α) × 1/(K + rank_bm25 + 1)
```
- `α` = `HYBRID_ALPHA` (기본 0.6) → 벡터 검색에 60% 가중치
- `K` = 60 (순위 차이의 영향을 완화)

**Stage 2: `async _rerank(query, candidates, top_n=10, recency_boost=False)`**
```
1. 후보를 RankingRecord로 변환 (title + content 1000자)
2. Discovery Engine Ranking API 호출 (asyncio.to_thread)
   - 모델: "semantic-ranker-default@latest"
   - recency_boost=True 면 top_n * 3 개까지 가져와 year 가중 재정렬
3. 상위 10개(또는 확장된 수) 반환
4. API 실패 시 → RRF 순서 유지 (폴백, logger.warning)
```

**Stage 3: `_expand_by_post(results, max_chunks_per_post=5)` (sync)**
```
1. 검색된 청크의 post_id 수집
2. ChunkStore에서 같은 post_id의 다른 청크 탐색
3. 게시글당 최대 5개 추가 청크 포함
4. 원래 검색 결과 뒤에 확장 청크 추가
```
- 목적: 하나의 게시글에서 여러 관련 정보를 함께 제공

**Stage 3.5: `_prepare_for_sort(docs)` — `_SORT_PATTERN` 매칭 시만 실행 (static)**
```
1. post_id별로 가장 높은 rerank_score 청크 하나만 남김 (dedupe)
2. post_id가 없는 청크는 orphan으로 보존
3. (year DESC, score DESC) 기준 stable sort
   - year None 은 -1 로 처리되어 맨 뒤
4. 정렬된 docs 반환 → LLM 컨텍스트가 이미 정렬된 상태로 구성됨
```
- 목적: "날짜순 내림차순 정렬" 류 질문에서 LLM의 정렬 부담 경감
- 보조 장치: `SYSTEM_PROMPT` 규칙 #13 이 "컨텍스트 순서 무시, 모든 항목 정렬" 을 강제

---

## 10. `src/generator.py`

**역할**: Gemini LLM을 사용하여 응답을 생성합니다.

### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `MAX_CONTINUATION_ROUNDS` | 3 | MAX_TOKENS 시 이어서 생성 최대 횟수 |

### `SYSTEM_PROMPT` (시스템 프롬프트)

LLM의 동작을 제어하는 한국어 규칙 **13개**:
1. 제공된 컨텍스트만 사용
2. 정보 없으면 검색 맥락 설명 + 미발견 안내
3. 출처 번호 `[1]` 등 본문에 미표기
4. 존댓말 사용
5. 의료 정보 정확 전달
6. 간결하고 명확한 답변
7. 긴 내용은 핵심 요약
8. 여러 문서 종합 답변
9. 숫자 범위에 `~` 기호 포함
10. 첨부파일은 `[[다운로드:파일명.확장자]]` 형식으로 링크
11. 마크다운 표 빈 셀 처리 규칙
12. "최근/최신/요즘" 질문 시 연도 내림차순 나열
13. 정렬/나열 요청 시: 컨텍스트 순서 무시, **모든 항목** 정렬, 같은 게시글 1회만, 일/월 단위까지 비교

### 모듈 함수

| 함수 | 설명 |
|------|------|
| `_is_retryable(exc)` | 재시도 대상 판정. `ServerError`, `httpx.TimeoutException`, `httpx.ConnectError`, `ClientError(code=429)` |
| `async _with_retry(coro_factory, *, label, max_attempts=5, initial_delay=2.0)` | 지수 백오프 공용 헬퍼. 429 감지 시 baseline 20초. 모든 Gemini 호출이 이를 거침 |

### `LLMGenerator`

#### `__init__(settings)`
- `genai.Client(vertexai=True)` 로 Gemini 클라이언트 생성
- `llm_location`이 설정돼 있으면 해당 리전 사용

#### `_config(temperature=0.3)`
- `temperature`: 0.3 (낮은 무작위성)
- `max_output_tokens`: 8192
- `thinking_config`: 사고 예산 0 (thinking 비활성화)
- `system_instruction`: SYSTEM_PROMPT 적용

#### `async generate(query, context, conversation_history?, temperature?)`
- 비스트리밍 응답 생성 (async)
- 대화 히스토리 최근 6턴 포함
- 프롬프트: `컨텍스트:\n{context}\n\n질문: {query}\n\n위 컨텍스트를 기반으로 답변해 주세요.`
- 모든 Gemini 호출이 `_with_retry(...)` 로 래핑됨 (429/5xx/네트워크 재시도)
- `finish_reason == "MAX_TOKENS"`이면 최대 3회 연속 생성
  - "이어서 답변해 주세요." 메시지로 계속 생성 요청
  - 이전 응답을 누적 연결

#### `async generate_stream(query, context, conversation_history?, temperature?)`
- SSE 스트리밍 응답 생성 — `AsyncGenerator[str, None]` 반환
- `self.client.aio.models.generate_content_stream(...)` 사용, 초기 호출을 `_with_retry` 로 래핑
- 각 청크의 `.text`가 None이 아닌 경우만 yield
- 연속 생성(MAX_TOKENS 처리) 없음 (스트리밍은 첫 청크 이후 재시도 불가)

#### `async rewrite_query(query, conversation_history)`
- 멀티턴 대화에서 현재 질문을 독립적으로 이해 가능하게 리라이팅
- 예: "그거 자세히" → "결핵 진료지침 4판의 치료 원칙을 자세히 알려줘"
- 대화 히스토리 최근 4턴 참조
- 규칙: 대명사 구체화, 시간/조건 미강제, 새 주제면 맥락 미적용
- `temperature=0.1` (매우 결정적)
- 결과가 5자 미만이면 원본 반환
- Gemini 호출은 `_with_retry` 로 보호됨 (429 시 장시간 대기)

#### `build_context(documents)`
- 검색된 문서 리스트를 LLM 프롬프트용 컨텍스트 문자열로 조합
- 형식: `[번호] 출처: 파일명 (게시글: 제목), 페이지 N`
- 첨부파일 목록 포함
- 문서 간 `---` 구분자

---

## 11. `src/memory.py`

**역할**: 세션별 대화 히스토리를 Redis 로 관리합니다. **Redis 필수**, in-memory fallback 은 제거되었습니다.

### `ConversationMemory`

#### `__init__(redis_client: redis.asyncio.Redis, max_turns=5, ttl=1800)`
- `redis_client` 가 `None` 이면 즉시 `ValueError`
- `ttl`: Redis 키 만료 시간 (기본 30분)

#### 메서드

| 메서드 | 설명 |
|--------|------|
| `create_session()` (sync) | UUID v4 세션 ID 생성 |
| `async get_history(session_id)` | `[{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]` 형태 반환 |
| `async add_turn(session_id, user_message, assistant_message)` | user + model 메시지 쌍 추가. `max_turns * 2` 초과 시 오래된 메시지 삭제 후 `setex(key, ttl, json)` |
| `async clear_session(session_id)` | `redis.delete(key)` |

#### Redis 키 형식
```
chat:session:{session_id}
```
값: JSON 직렬화된 히스토리 배열

---

## 12. `src/rag_pipeline.py`

**역할**: 검색 → 생성 → 메모리를 하나로 조합하는 **async 오케스트레이터**입니다. 예외 발생 시 사용자에게 친절한 fallback 토큰을 yield 합니다.

### 모듈 상수

| 상수 | 값 | 용도 |
|------|----|------|
| `QUOTA_FALLBACK_MESSAGE` | "현재 요청이 몰려 답변을 생성할 수 없습니다..." | 429 ClientError 최종 실패 시 |
| `GENERIC_FALLBACK_MESSAGE` | "답변 생성 중 오류가 발생했습니다..." | 기타 예외 / 빈 스트림 |

### 모듈 함수

- `_fallback_message(exc)` — 예외가 `ClientError(code=429)` 이면 `QUOTA_FALLBACK_MESSAGE`, 아니면 `GENERIC_FALLBACK_MESSAGE` 반환

### `RAGPipeline`

#### `async query(question, session_id?, source_type_filter?)`

비스트리밍 전체 흐름:

```
1. 세션 관리: session_id 없으면 생성
2. 대화 히스토리 로드 (Redis)
3. 히스토리 있으면 → 쿼리 리라이팅 (Gemini async)
4. HybridRetriever.retrieve(검색 쿼리) async
5. 결과 없으면 → "관련 정보를 찾을 수 없습니다." + add_turn 후 RAGResponse 반환
6. 컨텍스트 빌드 → Gemini 응답 생성 (await)
7. 대화 히스토리에 현재 턴 저장 (Redis)
8. RAGResponse(answer, sources, query, ...) 반환
```

#### `async query_stream(question, session_id?, source_type_filter?)`

스트리밍 버전. `AsyncGenerator[dict, None]`. 예외 처리 포함:

```
A. setup 블록(try/except):
   1. session_id 확보
   2. 대화 히스토리 로드 (Redis)
   3. 히스토리 있으면 → 쿼리 리라이팅
   4. HybridRetriever.retrieve
   → 예외 시 _fallback_message(exc) 토큰 + done yield 후 종료

B. 결과 없으면 → "관련 정보를 찾을 수 없습니다." 토큰 + done yield 후 종료

C. 스트리밍 생성(try/except):
   async for token in generator.generate_stream(...):
       full_answer.append(token)
       yield {"type": "token", "content": token}
   → 예외 시 stream_failed 에 저장

D. 결과 검증:
   - 스트림 실패 or 빈 full_answer → fallback 토큰 + done yield 후 종료
   - 정상 → Redis 에 add_turn, sources + done yield
```

---

## 13. `api/main.py`

**역할**: FastAPI 앱 초기화 및 서버 시작점입니다.

### `async lifespan(app)`

서버 시작 시 실행되는 asynccontextmanager:
1. `get_redis().ping()` 으로 Redis 연결 확인 (실패 시 앱 기동 중단)
2. `get_pipeline()` 호출 → 모든 컴포넌트 초기화
3. 워밍업: `await pipeline.query(question="테스트", session_id="warmup")` (실패는 무시)
4. `Pipeline ready.` 로그 후 요청 수락 시작
5. 종료 시: `await redis.aclose()` 로 연결 정리

### 라우트 등록

| 라우터 | 경로 |
|--------|------|
| `chat.router` | POST /chat |
| `search.router` | POST /search |
| `admin.router` | /admin/* |

### 정적 파일

- `/static` → `static/` 디렉토리
- `/` → `static/index.html` 반환

---

## 14. `api/middleware.py`

**역할**: CORS 설정과 요청 로깅을 담당합니다.

### `setup_middleware(app)`

**CORS 미들웨어:**
- 모든 오리진, 메서드, 헤더 허용 (`allow_origins=["*"]`)

**요청 로깅 미들웨어:**
- 모든 HTTP 요청의 메서드, 경로, 상태코드, 소요시간 기록 (structlog)
- 미처리 예외 발생 시 500 응답 + 에러 로그

---

## 15. `api/dependencies.py`

**역할**: 싱글턴 팩토리 함수들. 모든 컴포넌트를 `@lru_cache`로 한 번만 생성합니다.

| 팩토리 함수 | 생성 객체 | 초기화 작업 |
|-------------|----------|------------|
| `get_redis()` | `redis.asyncio.Redis` | `Redis.from_url(settings.redis_url, decode_responses=True)`. `redis_url` 미설정 시 `RuntimeError` |
| `get_embedding_service()` | EmbeddingService | google-genai 클라이언트 + Redis 캐시 주입 |
| `get_vectorstore()` | VectorStore | 인덱스/엔드포인트 로드 |
| `get_bm25_index()` | BM25Index | `bm25_index.pkl` 로드 |
| `get_generator()` | LLMGenerator | google-genai 클라이언트 생성 |
| `get_memory()` | ConversationMemory | `get_redis()` 주입 (fallback 없음) |
| `get_chunk_store()` | ChunkStore | `chunks.json` 로드 |
| `get_retriever()` | HybridRetriever | 위 4개 + Discovery Engine |
| `get_pipeline()` | RAGPipeline | retriever + generator + memory |

---

## 16. `api/routes/chat.py`

**역할**: 채팅 질의응답 엔드포인트입니다.

### `POST /chat`

```python
@router.post("/chat")
async def chat(request: ChatRequest, pipeline = Depends(get_pipeline)):
    ...
    result = await pipeline.query(...)
```

- `stream=True` → `StreamingResponse`로 SSE 스트림 반환
- `stream=False` → `ChatResponseModel` JSON 반환 (`await pipeline.query(...)`)

### `async _stream_response(pipeline, request)`

async generator. `async for event in pipeline.query_stream(...)` 결과를 SSE 형식으로 변환:
```
data: {"type": "token", "content": "..."}\n\n
data: {"type": "sources", "data": [...]}\n\n
data: {"type": "done", "session_id": "..."}\n\n
```

> 예외 처리는 `rag_pipeline.query_stream` 이 담당 — 예외 발생 시 fallback 토큰이 yield 되므로 라우트는 그대로 통과시키면 됨.

---

## 17. `api/routes/search.py`

**역할**: 리트리버 단독 호출 엔드포인트 (LLM 호출 없이 검색 결과만).

### `POST /search`

```python
@router.post("/search")
async def search(request: SearchRequest, retriever = Depends(get_retriever)):
    results = await retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        source_type_filter=request.source_type_filter,
    )
    return SearchResponseModel(results=results, query=request.query, total=len(results))
```

- 디버깅·튜닝 용도: 특정 쿼리가 어떤 청크를 retrieve 하는지 확인
- 반환값은 retriever 가 내보내는 dict 리스트 그대로 (rerank_score 포함)

---

## 18. `api/routes/admin.py`

**역할**: 헬스체크, 세션 관리, 파일 다운로드 엔드포인트입니다.

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/admin/health` | GET | `{"status": "ok"}` 반환 |
| `/admin/session/{session_id}` | DELETE | `await memory.clear_session(...)` — 대화 히스토리 삭제 |
| `/admin/download/{filename}` | GET | `data/documents/` 하위에서 `rglob`으로 파일 검색 후 다운로드 |

`download` 엔드포인트:
- FastAPI가 URL 디코딩을 자동 처리 (한국어/괄호/공백 포함 파일명 지원)
- `media_type="application/octet-stream"` — 브라우저가 파일로 다운로드

---

## 19. `scripts/index_documents.py`

**역할**: 문서를 로드 → 청킹 → 임베딩 → **Firestore 업로드** → BM25 재빌드하는 CLI 스크립트입니다. 전 구간 async.

### 실행 모드

```bash
python scripts/index_documents.py           # 증분 인덱싱 (기본)
python scripts/index_documents.py --full    # 전체 재인덱싱
python scripts/index_documents.py --add ID  # 특정 게시글 추가
python scripts/index_documents.py --delete ID  # 특정 게시글 삭제
```

### 구조

- `main()` 은 `asyncio.run(_main_async(args))` 로 진입. 모든 작업이 async.
- `vectorstore` 는 `FirestoreVectorStore` 인스턴스. `upsert`, `remove` 모두 await.
- 종료 시 `vectorstore.close()` 로 AsyncClient 정리.

### `async incremental_index(...)`

```
1. 디스크의 게시글 폴더 스캔 (data.json 있는 폴더의 post_id)
2. ChunkStore에서 이미 인덱싱된 post_id 조회
3. 차집합:
   new_posts = 디스크 - 인덱싱됨
   deleted_posts = 인덱싱됨 - 디스크
4. deleted_posts 의 청크 → await vectorstore.remove(ids) + ChunkStore 에서 제거
5. new_posts 각각:
   a. DocumentLoader.load_file() (try/except 로 에러 시 [SKIP])
   b. DocumentChunker.chunk_documents()
   c. await embeddings.embed_batch(texts)
   d. await vectorstore.upsert(chunks, embeddings)
   e. ChunkStore.save_chunks()
6. 변경 있으면 BM25 인덱스 재빌드 (_rebuild_bm25 helper)
```

### `async full_index(...)`

```
1. DocumentLoader.load_directory() 로 전체 문서 로드
2. DocumentChunker.chunk_documents()
3. await embeddings.embed_batch(...)
4. await _clear_firestore(vectorstore)  # 기존 collection 전체 삭제
5. await vectorstore.upsert(chunks, embeddings)
6. ChunkStore.save_chunks()
7. BM25Index.build() + save()
```

### `async _clear_firestore(vectorstore)`

- 컬렉션 전체를 순회하며 모든 document ID 수집 → `await vectorstore.remove(all_ids)`
- `--full` 모드에서만 호출됨. 증분 모드는 건드리지 않음.

---

## 20. `scripts/migrate_to_firestore.py`

**역할**: `data/chunk_store/chunks.json` 에 이미 저장된 청크를 직접 재임베딩해서 Firestore 에만 업로드하는 **빠른 경로**. `DocumentLoader`/OCR/Chunker 를 건너뛰므로 Document AI·Gemini Vision 호출이 발생하지 않아 비용이 $0.33 수준으로 제한됨.

### 사용 시나리오

- Firestore 로 최초 이관할 때 (Vertex AI Matching Engine → Firestore)
- chunks.json 이 이미 정상이고 Firestore 만 비어있을 때 빠르게 채울 때
- 테스트 용도로 `--limit 100` 같이 소량만 업로드할 때

### 실행

```bash
python scripts/migrate_to_firestore.py          # 전체 chunks.json 마이그레이션
python scripts/migrate_to_firestore.py --limit 100  # 앞 100개만 (POC)
python scripts/migrate_to_firestore.py --clear  # 기존 Firestore 컬렉션 비우고 시작
```

### 구조

```
1. chunks.json 로드
2. --clear 지정 시 기존 컬렉션 모두 삭제
3. 500개씩 페이지 단위로:
   a. EmbeddingService.embed_batch(texts)  # Vertex AI embedding
   b. VectorStore.upsert(chunks, vectors)  # Firestore batch write
   c. 진행률 로그 출력
4. 완료 메시지
```

### 주요 특징

- BM25 재빌드·chunk_store 쓰기 등 **부수 작업 없음** — 순수 Firestore 업로드 전용
- 16,737 chunks 기준 **약 10분** (embedding API 호출 대부분)
- 재임베딩 비용: ~$0.33 (1K 문자당 $0.00002)
- 중단해도 **멱등성 보장 안 됨** — 중복 ID 생성됨. 중간 실패 시 `--clear` 로 다시 시작 권장

---

## 21. `infra/docker-compose.yml`

**역할**: 로컬 개발에서 app + Redis 를 함께 기동하는 구성.

### 서비스

| 서비스 | 역할 |
|--------|------|
| `app` | FastAPI 컨테이너. `infra/Dockerfile` 로 빌드. `.env` 를 `env_file` 로 읽고, `REDIS_URL` 과 `GOOGLE_APPLICATION_CREDENTIALS` 는 compose 가 강제 주입 |
| `redis` | `redis:7-alpine`. healthcheck 로 `redis-cli ping` 을 주기적으로 수행 |

### 핵심 설정

- `environment: REDIS_URL=redis://redis:6379/0` — `.env` 의 값을 override 하여 컨테이너 간 네트워크 이름 사용
- `environment: GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json`
- `volumes: ${HOME}/.config/gcloud:/root/.config/gcloud:ro` — 호스트의 ADC 를 읽기 전용 마운트
- `volumes: ../data:/app/data` — 문서/인덱스 데이터 공유
- `depends_on: redis: condition: service_healthy` — Redis 가 ping 에 응답한 뒤 app 기동

### 기동

```bash
docker compose -f infra/docker-compose.yml up --build
```

기동 로그에 `Redis connection OK` → `Pipeline ready.` 가 보이면 정상.
