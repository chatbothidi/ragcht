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
17. [api/routes/admin.py](#17-apiroutesadminpy) — 관리 API
18. [scripts/index_documents.py](#18-scriptsindex_documentspy) — 인덱싱 스크립트

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
| `redis_url` | str \| None | None | Redis 주소 (미설정 시 인메모리) |

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

**역할**: 텍스트를 768차원 벡터로 변환합니다.

### `EmbeddingService`

#### `__init__(settings)`
- `vertexai.init(project, location)` 으로 SDK 초기화
- `TextEmbeddingModel.from_pretrained(model_name)` 으로 모델 로드
- 인메모리 캐시 딕셔너리 (`_cache`) 초기화

#### `embed(text, task_type="RETRIEVAL_QUERY")`
- 단일 텍스트 임베딩 (검색 쿼리용)
- MD5 해시 기반 인메모리 캐시 (동일 쿼리 재계산 방지)
- `_get_embeddings_with_retry()` 호출

#### `embed_batch(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=5)`
- 다수 텍스트 임베딩 (문서 인덱싱용)
- 5개씩 배치 처리
- `_get_embeddings_with_retry()` 호출

#### `_get_embeddings_with_retry(inputs, max_attempts=6, initial_delay=2.0)`
- Vertex AI API 호출에 지수 백오프 재시도 적용
- 재시도 대상 에러: `ServiceUnavailable`(503), `DeadlineExceeded`, `InternalServerError`(500), `ResourceExhausted`(429)
- 딜레이: 2s → 4s → 8s → 16s → 32s → 64s (+ 25% 지터)
- 6회 실패 시 최종 에러 발생

---

## 6. `src/vectorstore.py`

**역할**: Vertex AI Vector Search 인덱스를 생성하고 검색합니다.

### `VectorStore`

#### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `EMBEDDING_DIM` | 768 | text-multilingual-embedding-002의 차원 수 |

#### `__init__(settings)`
- Vertex AI SDK 초기화
- `vertex_index_id`와 `vertex_endpoint_id`가 설정돼 있으면 `load_existing()` 자동 호출

#### `create_collection()`
- Vector Search 인덱스 생성 (Tree-AH, COSINE, STREAM_UPDATE)
- 엔드포인트 생성 (public)
- 인덱스를 엔드포인트에 배포
- 최초 1회 `make setup`에서 호출

#### `load_existing(index_id, endpoint_id)`
- 리소스 ID로 기존 인덱스/엔드포인트 로드

#### `upsert(chunks, embeddings)`
- 청크별 UUID 생성
- DataPoint 구성: 벡터 + `source_type` 네임스페이스 + `source_file` 크라우딩 태그
- 100개씩 배치 업서트
- 반환: 생성된 chunk ID 리스트

**DataPoint 구조:**
```python
{
    "datapoint_id": "uuid",
    "feature_vector": [768 floats],
    "restricts": [{"namespace": "source_type", "allow_list": ["medical"]}],
    "crowding_tag": {"crowding_attribute": "파일명.pdf"}
}
```

#### `search(query_vector, top_k=10, source_type_filter?)`
- `find_neighbors()` 호출
- `source_type_filter`가 있으면 `Namespace` 필터 적용
- 반환: `[{"id": "...", "score": cosine_similarity}, ...]`
- score = `1.0 - cosine_distance`

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

#### `retrieve(query, top_k?, source_type_filter?)`

3단계를 순차 실행:

**Stage 1: `_hybrid_search(query, source_type_filter)`**
```
1. 쿼리 임베딩 생성 (task_type="RETRIEVAL_QUERY")
2. Vector Search에서 상위 50개 후보 검색
3. ChunkStore에서 각 후보의 텍스트/메타데이터 보강
4. BM25에서 상위 50개 후보 검색
5. RRF 퓨전으로 두 결과 병합 → 50개 반환
```

**RRF 공식:**
```
score(doc) = α × 1/(K + rank_vector + 1) + (1-α) × 1/(K + rank_bm25 + 1)
```
- `α` = `HYBRID_ALPHA` (기본 0.6) → 벡터 검색에 60% 가중치
- `K` = 60 (순위 차이의 영향을 완화)

**Stage 2: `_rerank(query, candidates, top_n=10)`**
```
1. 후보를 RankingRecord로 변환 (title + content 1000자)
2. Discovery Engine Ranking API 호출
   - 모델: "semantic-ranker-default@latest"
   - 시맨틱 관련도로 재정렬
3. 상위 10개 반환
4. API 실패 시 → RRF 순서 유지 (폴백)
```

**Stage 3: `_expand_by_post(results, max_chunks_per_post=5)`**
```
1. 검색된 청크의 post_id 수집
2. ChunkStore에서 같은 post_id의 다른 청크 탐색
3. 게시글당 최대 5개 추가 청크 포함
4. 원래 검색 결과 뒤에 확장 청크 추가
```
- 목적: 하나의 게시글에서 여러 관련 정보를 함께 제공

---

## 10. `src/generator.py`

**역할**: Gemini LLM을 사용하여 응답을 생성합니다.

### 상수

| 상수 | 값 | 설명 |
|------|---|------|
| `MAX_CONTINUATION_ROUNDS` | 3 | MAX_TOKENS 시 이어서 생성 최대 횟수 |

### `SYSTEM_PROMPT` (시스템 프롬프트)

LLM의 동작을 제어하는 한국어 규칙 11개:
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

### `LLMGenerator`

#### `__init__(settings)`
- `genai.Client(vertexai=True)` 로 Gemini 클라이언트 생성
- `llm_location`이 설정돼 있으면 해당 리전 사용

#### `_config(temperature=0.3)`
- `temperature`: 0.3 (낮은 무작위성)
- `max_output_tokens`: 8192
- `thinking_config`: 사고 예산 0 (thinking 비활성화)
- `system_instruction`: SYSTEM_PROMPT 적용

#### `generate(query, context, conversation_history?, temperature?)`
- 비스트리밍 응답 생성
- 대화 히스토리 최근 6턴 포함
- 프롬프트: `컨텍스트:\n{context}\n\n질문: {query}\n\n위 컨텍스트를 기반으로 답변해 주세요.`
- `finish_reason == "MAX_TOKENS"`이면 최대 3회 연속 생성
  - "이어서 답변해 주세요." 메시지로 계속 생성 요청
  - 이전 응답을 누적 연결

#### `generate_stream(query, context, conversation_history?, temperature?)`
- SSE 스트리밍 응답 생성
- `generate_content_stream()` 사용
- 각 청크의 `.text`가 None이 아닌 경우만 yield
- 연속 생성(MAX_TOKENS 처리) 없음

#### `rewrite_query(query, conversation_history)`
- 멀티턴 대화에서 현재 질문을 독립적으로 이해 가능하게 리라이팅
- 예: "그거 자세히" → "결핵 진료지침 4판의 치료 원칙을 자세히 알려줘"
- 대화 히스토리 최근 4턴 참조
- 규칙: 대명사 구체화, 시간/조건 미강제, 새 주제면 맥락 미적용
- `temperature=0.1` (매우 결정적)
- 결과가 5자 미만이면 원본 반환

#### `build_context(documents)`
- 검색된 문서 리스트를 LLM 프롬프트용 컨텍스트 문자열로 조합
- 형식: `[번호] 출처: 파일명 (게시글: 제목), 페이지 N`
- 첨부파일 목록 포함
- 문서 간 `---` 구분자

---

## 11. `src/memory.py`

**역할**: 세션별 대화 히스토리를 관리합니다.

### `ConversationMemory`

#### `__init__(redis_url?, max_turns=5, ttl=1800)`
- `redis_url`이 있으면 Redis 클라이언트 생성
- 없으면 인메모리 딕셔너리 사용 (`_local_store`)
- `ttl`: Redis 키 만료 시간 (기본 30분)

#### 메서드

| 메서드 | 설명 |
|--------|------|
| `create_session()` | UUID v4 세션 ID 생성 |
| `get_history(session_id)` | `[{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]` 형태로 반환 |
| `add_turn(session_id, user_message, assistant_message)` | user + model 메시지 쌍 추가. `max_turns * 2` 초과 시 오래된 메시지 삭제 |
| `clear_session(session_id)` | 세션 기록 삭제 |

#### Redis 키 형식
```
chat:session:{session_id}
```
값: JSON 직렬화된 히스토리 배열

---

## 12. `src/rag_pipeline.py`

**역할**: 검색 → 생성 → 메모리를 하나로 조합하는 오케스트레이터입니다.

### `RAGPipeline`

#### `query(question, session_id?, source_type_filter?)`

비스트리밍 전체 흐름:

```
1. 세션 관리: session_id 없으면 생성
2. 대화 히스토리 로드
3. 히스토리 있으면 → 쿼리 리라이팅
4. HybridRetriever.retrieve(검색 쿼리)
5. 결과 없으면 → "관련 정보를 찾을 수 없습니다." 반환
6. 컨텍스트 빌드 → Gemini 응답 생성
7. 대화 히스토리에 현재 턴 저장
8. RAGResponse(answer, sources, query, ...) 반환
```

#### `query_stream(question, session_id?, source_type_filter?)`

스트리밍 버전. Generator[dict] 반환:

```
1~4. query()와 동일
5. 결과 없으면 → {"type": "token", "content": "..."} + {"type": "done"} yield
6. Gemini 스트리밍 → 토큰마다 {"type": "token", "content": "..."} yield
7. 전체 답변을 히스토리에 저장
8. {"type": "sources", "data": [...]} yield
9. {"type": "done", "session_id": "..."} yield
```

---

## 13. `api/main.py`

**역할**: FastAPI 앱 초기화 및 서버 시작점입니다.

### `lifespan(app)`

서버 시작 시 실행되는 asynccontextmanager:
1. `get_pipeline()` 호출 → 모든 컴포넌트 초기화
2. 워밍업 쿼리 `"테스트"` 실행 → gRPC 연결 수립
3. `Pipeline ready.` 로그 후 요청 수락 시작

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
| `get_embedding_service()` | EmbeddingService | Vertex AI SDK + 모델 로드 |
| `get_vectorstore()` | VectorStore | 인덱스/엔드포인트 로드 |
| `get_bm25_index()` | BM25Index | `bm25_index.pkl` 로드 |
| `get_generator()` | LLMGenerator | Gemini 클라이언트 생성 |
| `get_memory()` | ConversationMemory | Redis 연결 또는 인메모리 |
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
```

- `stream=True` → `StreamingResponse`로 SSE 스트림 반환
- `stream=False` → `ChatResponseModel` JSON 반환 (현재 프론트엔드 미사용)

### `_stream_response(pipeline, request)`

async generator. `pipeline.query_stream()`의 결과를 SSE 형식으로 변환:
```
data: {"type": "token", "content": "..."}\n\n
data: {"type": "sources", "data": [...]}\n\n
data: {"type": "done", "session_id": "..."}\n\n
```

> 주의: try/except 없음. 예외 발생 시 스트림이 끊기고 프론트엔드에 에러가 표시되지 않음.

---

## 17. `api/routes/admin.py`

**역할**: 헬스체크, 세션 관리, 파일 다운로드 엔드포인트입니다.

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/admin/health` | GET | `{"status": "ok"}` 반환 |
| `/admin/session/{session_id}` | DELETE | 대화 히스토리 삭제 |
| `/admin/download/{filename}` | GET | `data/documents/` 하위에서 `rglob`으로 파일 검색 후 다운로드 |

`download` 엔드포인트:
- FastAPI가 URL 디코딩을 자동 처리 (한국어/괄호/공백 포함 파일명 지원)
- `media_type="application/octet-stream"` — 브라우저가 파일로 다운로드

---

## 18. `scripts/index_documents.py`

**역할**: 문서를 로드 → 청킹 → 임베딩 → Vector Search 업로드하는 CLI 스크립트입니다.

### 실행 모드

```bash
python scripts/index_documents.py           # 증분 인덱싱 (기본)
python scripts/index_documents.py --full    # 전체 재인덱싱
python scripts/index_documents.py --add ID  # 특정 게시글 추가
python scripts/index_documents.py --delete ID  # 특정 게시글 삭제
```

### `incremental_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)`

```
1. 디스크의 게시글 폴더 스캔 (data.json 있는 폴더의 post_id)
2. ChunkStore에서 이미 인덱싱된 post_id 조회
3. 차집합 계산:
   new_posts = 디스크 - 인덱싱됨
   deleted_posts = 인덱싱됨 - 디스크
4. deleted_posts의 청크 삭제 (Vector Search + ChunkStore)
5. new_posts 각각:
   a. DocumentLoader.load_file() (try/except로 에러 시 [SKIP])
   b. DocumentChunker.chunk_documents()
   c. EmbeddingService.embed_batch()
   d. VectorStore.upsert()
   e. ChunkStore.save_chunks()
6. 변경 있으면 BM25 인덱스 재빌드
```

### `full_index(loader, chunker, embeddings, vectorstore, chunk_store, bm25, doc_dir)`

```
1. DocumentLoader.load_directory() 로 전체 문서 로드
2. DocumentChunker.chunk_documents()
3. EmbeddingService.embed_batch()
4. VectorStore.upsert()
5. ChunkStore.save_chunks()
6. BM25Index.build() + save()
```
