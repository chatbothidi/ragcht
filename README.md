# RAG Chatbot

RAG 챗봇 시스템입니다.

## 기술 스택

- **LLM**: Vertex AI Gemini 2.5 Flash (google-genai SDK, thinking 비활성화)
- **임베딩**: text-multilingual-embedding-002 (768차원)
- **벡터 스토어**: Vertex AI Vector Search
- **리랭킹**: Discovery Engine Ranking API (semantic-ranker-default)
- **한국어 형태소**: kiwipiepy
- **검색**: 2단계 하이브리드 (Vector Search + BM25 + RRF → Rerank)
- **API**: FastAPI + SSE 스트리밍
- **대화 메모리**: Redis
- **OCR**: Google Document AI (스캔 PDF), Gemini Vision (이미지 파일/URL)

## 검색 파이프라인

```
사용자 질문
  ↓
쿼리 리라이팅 (멀티턴 시 Gemini로 대명사/맥락 해소)
  ↓
1단계: 하이브리드 검색
  - Vector Search (top_k=50)
  - BM25 한국어 형태소 (top_k=50)
  - RRF 융합 → 후보 50개
  ↓
2단계: Discovery Engine Reranking
  - 사용자 질문 기반 관련성 재정렬
  - 상위 10개 선별
  ↓
Gemini 답변 생성 (스트리밍)
```

## 사전 요구사항

- Python 3.12+
- GCP 프로젝트 (Vertex AI API, Document AI API, Discovery Engine API 활성화)
- `gcloud auth application-default login` 완료
- Docker (Redis 실행용)

## 시작하기

```bash
# 1. 가상환경 생성 및 의존성 설치
python3 -m venv .venv
source .venv/bin/activate
make install

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에서 GCP_PROJECT_ID, VERTEX_INDEX_ID, VERTEX_ENDPOINT_ID 수정

# 3. Redis 실행
make docker-up

# 4. Vector Search 셋업 (최초 1회)
make setup

# 5. 문서 인덱싱
python scripts/index_documents.py --full

서버 실행 후 `http://localhost:8000`에서 채팅 UI에 접속할 수 있습니다.

## 문서 관리

### 문서 구조

게시글 ID 기준 폴더 구조입니다.

```
data/documents/
├── [게시글 ID]/
│   ├── data.json                  # 메타데이터 (필수)
│   ├── [게시글 제목].md            # 본문
│   ├── [이미지].jpg               # 첨부 이미지 (자동 Vision 추출)
│   └── .image_cache.json          # 이미지 추출 캐시 (자동 생성)
├── 9874/
│   ├── data.json
│   ├── DCC 대회참가.md
│   ├── 참가신청서.docx
│   └── 대회 관련 자료.pdf
└── ...
```

### data.json 형식

```json
{
  "id": 12768,
  "title": "2026 결핵및 호흡기학회 춘계학술대회 개최 안내",
  "category": "event",
  "year": 2026,
  "main_file": "2026 결핵및 호흡기학회 춘계학술대회.md",
  "attachments": ["2026 결핵및 호흡기학회 춘계학술대회.jpg", "2026 결핵및 호흡기학회 춘계학술대회 행사 일정.pdf"]
}
```

| 필드 | 설명 |
|------|------|
| `id` | 게시글 ID |
| `title` | 게시글 제목 |
| `category` | `"medical"` 또는 `"event"` 등등| 
| `year` | 연도 (없으면 `null`) |
| `main_file` | 본문 파일명 |
| `attachments` | 첨부파일 목록 |

### 문서 추가 방법

```bash
source .venv/bin/activate

# 1. 폴더 생성 + data.json 작성 + 파일 배치
mkdir data/documents/12345
# data.json, 본문 파일, 첨부파일 넣기

# 2. 인덱싱
python scripts/index_documents.py

# 3. 서버 재시작
fuser -k 8000/tcp 2>/dev/null; sleep 2
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
```

### 자동 처리되는 파일 유형

| 파일 유형 | 처리 방법 |
|----------|----------|
| PDF (텍스트) | pymupdf 텍스트 추출 |
| PDF (스캔) | Document AI OCR |
| DOCX | python-docx 텍스트 추출 |
| MD/TXT | 텍스트 추출 |
| MD 안의 이미지 URL | Gemini Vision 자동 추출 (캐시) |
| 첨부 이미지 (jpg, png 등) | Gemini Vision 자동 추출 (캐시) |

### 상황별 명령어

| 상황 | 명령 |
|------|------|
| 특정 게시글 추가/재인덱싱 | `python scripts/index_documents.py --add 12345` |
| 특정 게시글 삭제 | `python scripts/index_documents.py --delete 12345` |
| 자동 감지 (추가/삭제) | `python scripts/index_documents.py` |
| 파일명/내용/폴더명 변경 | `python scripts/index_documents.py --full` |
| 전체 초기화 | `python scripts/index_documents.py --full` |

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /chat` | 챗봇 대화 (스트리밍/일괄) |
| `POST /search` | 직접 검색 |
| `GET /admin/health` | 헬스체크 |
| `GET /admin/download/{filename}` | 첨부파일 다운로드 |
| `DELETE /admin/session/{session_id}` | 세션 삭제 |

### POST /chat 예시

```json
{
  "query": "LCK 2026행사 일정을 알려줘",
  "session_id": "optional-session-id",
  "source_type_filter": "Esport",
  "stream": true
}
```

## 개발

```bash
make dev      # 개발 의존성 설치
make test     # 테스트 실행
make lint     # 린트 검사
make format   # 코드 포맷팅
```