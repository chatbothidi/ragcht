# 자체 리눅스 서버 배포 + 외부 챗봇 연동 가이드

대상: 본 RAG API를 자체 리눅스 서버(IDC, 사내 VM)에 띄우고, 다른 도메인에서 동작하는 기존 웹 챗봇(브라우저 프런트엔드)이 이 API를 직접 호출하도록 구성하려는 운영자.

전제 조건:
- 호출자는 **브라우저** — CORS, HTTPS 필수.
- API 서버는 자체 리눅스 호스트.
- LLM/임베딩/OCR은 GCP를 그대로 사용.
- 호스트와 GCP 사이 아웃바운드 HTTPS(443) 통신 가능.

---

## 1. 서버 사양 가이드

| 구분 | 최소 | 운영 권장 | 비고 |
|---|---|---|---|
| CPU | 2 vCPU | 4 vCPU | 동시 SSE 스트림이 많아질수록 증설. LLM 호출은 외부 API라 CPU 영향 적음. |
| RAM | 4 GB | 8 GB | kiwipiepy 형태소 사전 + BM25 인메모리 + ChunkStore JSON + Python 워커 |
| Disk | 20 GB SSD | 40 GB SSD | 원본 문서(`data/`), `data/bm25_index/*.pkl`, `data/chunk_store/*.json`, Redis dump, Docker layer |
| OS | Ubuntu 22.04 LTS / RHEL 9 / Debian 12 | 동일 | Linux x86_64. ARM은 일부 GCP SDK 호환 확인 필요. |
| 네트워크 인바운드 | 443 | 443 | 직접 노출 시 8000도. 22(SSH)는 관리자 IP 화이트리스트 권장. |
| 네트워크 아웃바운드 | `*.googleapis.com:443` | 동일 | Vertex AI / Firestore / Document AI / Discovery Engine |

**메모리 산정 근거**
- kiwipiepy 형태소 사전: 수백 MB (한국어 분석 기본 데이터)
- BM25 + ChunkStore: 문서 ~수천 청크 기준 수십 MB. 청크 수에 비례해 증가.
- Python 프로세스(uvicorn workers): 베이스 ~300 MB, 동시 스트림 1개당 추가 메모리 소량.
- Redis 컨테이너: ~50 MB + 세션 데이터 + 임베딩 캐시.

**지연 고려사항**
- GCP region이 `asia-northeast3`(서울)이면 한국 호스트 RTT 수십 ms.
- 다른 region이거나 해외 호스트 사용 시 LLM 첫 토큰 지연이 눈에 띄게 늘어날 수 있음.

---

## 2. 사전 준비 (1회성, GCP 측)

1. GCP 프로젝트 ID 확보, 결제 활성화.
2. 서비스 계정(SA) 생성. 다음 IAM 역할 부여:
   - `roles/aiplatform.user` — Vertex AI 호출
   - `roles/datastore.user` — Firestore 읽기/쓰기
   - `roles/documentai.apiUser` — Document AI OCR
   - `roles/discoveryengine.viewer` — 재순위 Ranking API
3. SA 키 JSON 다운로드.
4. Firestore Native DB 생성 + 벡터 인덱스 2개:
   ```bash
   gcloud firestore databases create --location=asia-northeast3 --type=firestore-native
   # 단일/복합 벡터 인덱스 생성은 docs/HANDOVER.md 9.1 명령 그대로 사용
   ```
5. Document AI processor 생성 → ID 확보.
6. (운영 중요) Vertex AI quota 모니터링 — `gemini-2.5-flash`, `text-multilingual-embedding-002` 호출량.

---

## 3. 서버 준비

```bash
# Docker / Compose plugin
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git

# 일반 사용자에 docker 권한
sudo usermod -aG docker $USER && newgrp docker

# 방화벽 (ufw 예)
sudo ufw allow 22/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

`8000`을 외부에 직접 노출하지 말 것 — 항상 nginx/Caddy 뒤에 두고 TLS 종료.

---

## 4. 배포

### 4.1 코드 + 데이터 + 키 배치

```bash
sudo mkdir -p /opt/medical-event-rag /etc/secrets
sudo chown -R $USER /opt/medical-event-rag

git clone <repo-url> /opt/medical-event-rag
cd /opt/medical-event-rag
cp .env.example .env
```

`.env` 편집 — 필수 항목:
```ini
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=asia-northeast3
DOCAI_PROCESSOR=projects/<num>/locations/<loc>/processors/<id>
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=text-multilingual-embedding-002
FIRESTORE_DATABASE_ID=(default)
FIRESTORE_COLLECTION_NAME=medical_event_chunks
REDIS_URL=redis://redis:6379/0
```

문서 데이터 동기화:
```bash
mkdir -p data
rsync -av <원본-경로>/ data/
```

SA 키 보관:
```bash
sudo install -m 600 -o root -g root sa-key.json /etc/secrets/sa-key.json
```

### 4.2 docker-compose 수정 (SA 키 마운트)

기존 `infra/docker-compose.yml`은 사용자 ADC(`~/.config/gcloud`)를 마운트. 운영 서버에선 SA 키 파일로 교체:

```yaml
services:
  app:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    ports:
      - "127.0.0.1:8000:8000"   # 외부 노출 막고 nginx만 접근
    env_file: ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - REDIS_URL=redis://redis:6379/0
      - GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/sa-key.json
    volumes:
      - ../data:/app/data
      - /etc/secrets/sa-key.json:/var/secrets/sa-key.json:ro
    depends_on:
      redis:
        condition: service_healthy
```

### 4.3 인덱싱 + 기동

```bash
cd /opt/medical-event-rag/infra

# 초기 인덱싱 (Firestore 업서트 + BM25 + ChunkStore)
docker compose run --rm app python scripts/index_documents.py
docker compose run --rm app python scripts/build_bm25_index.py

# 데몬 기동
docker compose up -d
docker compose logs -f app
```

`Pipeline ready.` 로그가 보이면 정상.

---

## 5. 리버스 프록시 + TLS (필수)

브라우저 fetch는 mixed-content 정책상 HTTPS만 허용. nginx 또는 Caddy 둘 중 하나.

### nginx 예제 (`/etc/nginx/sites-enabled/rag.conf`)

```nginx
server {
    listen 443 ssl http2;
    server_name rag.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/rag.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.your-domain.com/privkey.pem;

    # SSE를 위한 핵심 옵션
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Accel-Buffering no;
    }
}

server {
    listen 80;
    server_name rag.your-domain.com;
    return 301 https://$host$request_uri;
}
```

`proxy_buffering off`가 빠지면 SSE 토큰이 청크 단위로 모였다가 한꺼번에 떨어져 사용자 체감 지연이 큼.

### Caddy 대안 (자동 TLS)

```caddyfile
rag.your-domain.com {
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1     # SSE
    }
}
```

---

## 6. CORS 설정 (운영 시 좁히기)

현재 `api/middleware.py:13-19`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

브라우저 사양상 `allow_origins=["*"]`와 `allow_credentials=True`의 조합은 실제 쿠키/Authorization 동반 요청을 거부함. 외부 챗봇 도메인이 정해져 있으면 명시:

```python
allow_origins=[
    "https://chatbot.example.com",
    "https://staging-chatbot.example.com",
],
allow_credentials=True,
allow_methods=["GET", "POST", "DELETE"],
allow_headers=["Content-Type", "X-API-Key", "Authorization"],
```

`api/middleware.py` 한 곳만 수정 후 `docker compose up -d --build`.

---

## 7. 인증 (현재 없음 — 운영 노출 시 필수 추가)

현 API는 헤더/토큰 검사 없음 → URL만 알면 누구나 호출 가능. 운영 노출 전 한 가지 이상 적용 권고.

### 옵션 A — API key 미들웨어 (가장 간단)

`api/middleware.py`에 추가 (예시):
```python
import os
from fastapi import Request, status
from fastapi.responses import JSONResponse

EXPECTED_KEY = os.environ.get("API_KEY")

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # health, 정적 파일은 통과
    if request.url.path.startswith(("/admin/health", "/static")) or request.url.path == "/":
        return await call_next(request)
    if EXPECTED_KEY and request.headers.get("X-API-Key") != EXPECTED_KEY:
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)
```
`.env`에 `API_KEY=...` 추가, 외부 챗봇은 fetch 헤더에 `X-API-Key: <key>` 동봉.

브라우저 환경에선 키가 클라이언트 코드에 들어가므로 완전 보안은 아님 — IP 제한이나 도메인 제한과 병용.

### 옵션 B — 챗봇 백엔드 발급 JWT 검증

기존 챗봇이 사용자 인증 보유 시. `api/dependencies.py`에 `verify_jwt` 의존성 추가, 라우트에 `Depends(verify_jwt)` 부착. JWT 서명 키는 챗봇 백엔드와 공유.

### 옵션 C — 리버스 프록시에서 차단

nginx `auth_request` 또는 IP allowlist (`allow 1.2.3.4; deny all;`). 가장 단순하지만 경로별 차등 인가 어려움.

운영 처음엔 A + IP allowlist 조합으로 시작, 사용자 인증 필요해지면 B로 전환 추천.

---

## 8. 외부 챗봇 프런트엔드 연동 코드

### 8.1 엔드포인트 요약

| Method | Path | 용도 |
|---|---|---|
| POST | `/chat` | 대화. `stream:true`면 SSE, 아니면 JSON. |
| POST | `/search` | 검색만 (LLM 호출 없음). `top_k`, `source_type_filter` 지정. |
| GET  | `/admin/health` | 헬스체크. |
| DELETE | `/admin/session/{id}` | 세션 메모리 삭제. |
| GET | `/admin/download/{filename}` | 답변 안의 첨부파일 다운로드. |

### 8.2 요청 스키마 (`src/models.py:6-12`)

```ts
type ChatRequest = {
  query: string;
  session_id?: string;          // 첫 요청은 생략, 응답에서 받은 값을 다음부터 동봉
  source_type_filter?: "medical" | "event";
  stream: boolean;              // 스트리밍 여부
};
```

응답(JSON 모드) 스키마(`src/models.py:43-48`):
```ts
type ChatResponse = {
  answer: string;
  sources: Array<{
    source_file: string;
    source_type: "medical" | "event";
    page_number: number | null;
    relevance_score: number;     // 0.0 ~ 1.0
    excerpt: string;
  }>;
  query: string;
  rewritten_query?: string | null;
  session_id: string;
};
```

### 8.3 JSON 모드 (간단)

```js
const API_BASE = "https://rag.your-domain.com";

async function ask(query, sessionId) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.RAG_API_KEY,    // 옵션 A 인증 적용 시
    },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      stream: false,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();   // { answer, sources, session_id, ... }
}
```

### 8.4 SSE 모드 (스트리밍)

서버는 `data: {...}\n\n` 형식으로 세 종류 이벤트를 흘림:
- `{"type":"token","content":"..."}` — LLM 토큰 조각
- `{"type":"sources","data":[...]}` — 검색 출처
- `{"type":"done","session_id":"..."}` — 종료 + 세션 ID

`static/index.html:471-498`의 패턴을 옮겨 사용:

```js
async function askStream(query, sessionId, onToken, onSources, onDone) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.RAG_API_KEY,
    },
    body: JSON.stringify({ query, session_id: sessionId, stream: true }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();   // 미완성 라인은 다음 청크로 이월

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === "token") onToken(event.content);
      else if (event.type === "sources") onSources(event.data);
      else if (event.type === "done") onDone(event.session_id);
    }
  }
}
```

### 8.5 세션 ID 보관

- 첫 요청은 `session_id` 생략 → 서버가 발급해 응답 또는 `done` 이벤트로 전달.
- 클라이언트는 그 값을 보관(예: 로컬 변수, sessionStorage)하고 다음 호출에 동봉.
- Redis TTL 30분 — 미사용 시 자동 만료. 새 세션을 의도적으로 시작하려면 `session_id`를 비우거나 `DELETE /admin/session/{id}` 호출.

### 8.6 첨부파일 다운로드 마커 변환

답변 본문에 다음 마커가 등장할 수 있음 (`src/generator.py` SYSTEM_PROMPT 규칙 10):
```
[[다운로드:결핵진료지침(4판)_(Web용)_최종.pdf]]
```
`static/index.html:377-382`의 변환 로직과 동일하게 처리:
```js
function rewriteDownloadMarkers(text, apiBase) {
  return text.replace(/\[\[다운로드:([^\]]+?)\]\]/g, (_, filename) => {
    const trimmed = filename.trim();
    return `[${trimmed} 다운로드](${apiBase}/admin/download/${encodeURIComponent(trimmed)})`;
  });
}
```

마크다운 렌더링 후 클릭하면 `/admin/download/...`가 호출됨. 이 엔드포인트도 인증 미들웨어 통과 대상이 되도록 토큰을 query string이나 `Authorization` 헤더 등으로 전달할 수 있어야 함(브라우저 직접 클릭은 헤더 동봉이 어려움 → 옵션 A의 화이트리스트 경로 또는 서명된 URL 도입 검토).

---

## 9. 검증 체크리스트

서버에서:
```bash
# 1) 헬스체크
curl https://rag.your-domain.com/admin/health
# → {"status":"ok"}

# 2) JSON 모드
curl -X POST https://rag.your-domain.com/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <key>' \
  -d '{"query":"테스트","stream":false}'

# 3) SSE 모드 (라인이 끊기지 않고 흘러야 정상)
curl -N -X POST https://rag.your-domain.com/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <key>' \
  -d '{"query":"2026년 학술대회 알려줘","stream":true}'
```

브라우저(챗봇 도메인 콘솔)에서:
```js
fetch("https://rag.your-domain.com/chat", {
  method: "POST",
  headers: {"Content-Type":"application/json","X-API-Key":"<key>"},
  body: JSON.stringify({query:"테스트", stream:false}),
}).then(r => r.json()).then(console.log);
```
- CORS preflight(OPTIONS) → 200
- 본 요청 → 200 + JSON

---

## 10. 운영

| 항목 | 명령 / 위치 |
|---|---|
| 로그 follow | `docker compose -f infra/docker-compose.yml logs app -f` |
| 빈 응답 추적 | `docker compose ... logs app | grep "EMPTY response"` |
| 인덱스 증분 갱신 | `docker compose run --rm app python scripts/index_documents.py` |
| BM25만 재구축 | `docker compose run --rm app python scripts/build_bm25_index.py` |
| Redis 백업 | `docker run --rm -v redis_data:/data -v $PWD/backup:/backup alpine tar czf /backup/redis-$(date +%F).tgz /data` |
| 헬스체크 모니터 | 외부 모니터(UptimeRobot 등)에서 `/admin/health` 5분 간격 |
| GCP 비용 | Vertex AI / Firestore / Document AI 월 결제 알림. `docs/HANDOVER.md` 섹션 11 참고 |
| 컨테이너 재시작 | `docker compose -f infra/docker-compose.yml restart app` |
| 코드 갱신 후 | `git pull && docker compose -f infra/docker-compose.yml up -d --build` |

---

## 11. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| 브라우저에서 CORS 차단 | `api/middleware.py`의 `allow_origins`에 정확한 origin(scheme + host + port) 추가. wildcard와 credentials 동시 사용 불가. |
| 첫 SSE 토큰 도착이 느리다 | nginx `proxy_buffering off` 누락. Caddy면 `flush_interval -1`. |
| `RuntimeError: REDIS_URL is required` | `api/dependencies.py:19-23`. Redis 컨테이너 미기동 또는 env 누락. |
| Vertex AI 401/403 | SA 키 권한, processor ID, 프로젝트 일치 확인. `GOOGLE_APPLICATION_CREDENTIALS` 경로 확인. |
| Vertex AI 429 | 쿼터 초과. 사용자에겐 `QUOTA_FALLBACK_MESSAGE`(`src/rag_pipeline.py:15`)로 노출됨. GCP 쿼터 증액 신청. |
| 답변에 출처가 비어 있음 | retrieval에서 매칭 실패. `score_threshold`(`src/config.py:32`)와 `top_k`, `bm25_top_k` 점검. 인덱스 재구축. |
| 한국어 토크나이징 실패 | kiwipiepy 사전 다운로드 실패 가능. 컨테이너 빌드 시 인터넷 접근 또는 사전 캐시 확인. |
| 첨부파일 다운로드 401 | `/admin/download/*` 경로가 인증 미들웨어 통과 대상 — 가이드 7장의 화이트리스트 또는 서명 URL 처리 필요. |

---

## 12. 후속(선택)으로 적용할 변경

본 가이드는 문서 작성에 한정. 운영 시 권장하는 코드 변경은 다음 — 실제 적용은 별도 PR 진행:
1. `api/middleware.py`의 CORS `allow_origins` → 챗봇 도메인 명시.
2. `api/middleware.py`에 API key 미들웨어 추가, `.env`에 `API_KEY`.
3. `infra/docker-compose.yml` SA 키 볼륨 마운트로 변경.
4. `static/index.html`의 detached-node 렌더 버그(521-526) 정리 — 다른 도메인에서 호출 시는 직접 영향 없으나, 본 repo의 `/`에 그대로 노출되므로 정리 권장.

---

## 부록 — 참조 파일

| 영역 | 파일:라인 |
|---|---|
| CORS 설정 | `api/middleware.py:13-19` |
| 라우트 정의 | `api/routes/chat.py`, `api/routes/search.py`, `api/routes/admin.py` |
| 요청/응답 스키마 | `src/models.py:6-12`, `:43-48` |
| Redis 의존 검사 | `api/dependencies.py:19-23` |
| 다운로드 마커 변환 | `static/index.html:377-382` |
| 스트리밍 수신 패턴 | `static/index.html:471-498` |
| 서버 lifespan/warmup | `api/main.py:16-30` |
| Dockerfile | `infra/Dockerfile` |
| compose | `infra/docker-compose.yml` |
| Cloud Run 자동배포 | `infra/cloudbuild.yaml` (참고용, 본 가이드 범위 밖) |
