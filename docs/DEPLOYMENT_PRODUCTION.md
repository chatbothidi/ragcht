# medical-event-rag 운영 배포 가이드 — Rocky Linux 9 + Apache + api.lungkorea.org

대상: medical-event-rag를 Rocky Linux 9 운영 서버에 docker compose로 띄우고, 기존 Apache(httpd)에서 `api.lungkorea.org` 도메인으로 외부에 노출하는 시나리오.

전제:
- 도메인 `api.lungkorea.org`이 이미 운영 서버 IP를 가리킴 (DNS 설정 완료).
- Rocky Linux 9.x.
- Apache(httpd) 사용. nginx 아님.
- 호출자: 자체 UI(`https://api.lungkorea.org/`) 또는 기존 Laravel 챗봇(서버-투-서버 cURL 프록시).

---

## 전체 그림

```
[A] 자체 UI 경로
브라우저 ── https://api.lungkorea.org/ ──> Apache ── 127.0.0.1:8001 (RAG)

[B] Laravel 챗봇 경로 (기존 도메인은 어디든 그대로 둠)
브라우저 ── 기존 챗봇 blade UI ──> Laravel /api/chatbot/send
                                         │
                                         └── cURL ──> https://api.lungkorea.org/chat ── Apache ── RAG
```

도메인은 `api.lungkorea.org` 하나만. Laravel은 현재 위치 그대로 두고 `.env`의 `RAG_API_URL`만 가리키면 됨.

핵심 원칙:
- RAG는 **localhost에만 노출** (`127.0.0.1:8001`).
- Apache가 80/443에서 외부 트래픽 받아 RAG로 프록시.
- Rocky 9 특성상 SELinux와 firewalld 설정이 필수.

---

## Phase 1 — 서버 사전 준비

### 1.1 시스템 + Swap

```bash
sudo dnf update -y
sudo dnf install -y git curl wget vim policycoreutils-python-utils

# Swap 4GB (RAM 3.6GB라 OOM 방지)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### 1.2 Docker CE 설치

Rocky 9 기본은 Podman. docker compose 호환을 위해 Docker CE 권장:

```bash
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
docker --version && docker compose version
```

### 1.3 Firewalld

```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

8001은 외부에 절대 열지 않음. Apache가 localhost로만 접속.

### 1.4 SELinux — Rocky 9에서 자주 막히는 지점

```bash
sudo setsebool -P httpd_can_network_connect 1
getsebool httpd_can_network_connect    # → on 확인
```

이거 빠지면 Apache → docker 프록시가 503으로 막힘.

---

## Phase 2 — GCP 사전 준비 (1회성)

### 2.1 서비스 계정 + 키

Console → IAM → Service Accounts:

1. 새 SA 생성 (예: `rag-api@<project>.iam.gserviceaccount.com`)
2. 역할 부여:
   - `roles/aiplatform.user`
   - `roles/datastore.user`
   - `roles/documentai.apiUser`
   - `roles/discoveryengine.viewer`
3. JSON 키 다운로드 → 서버 전송:

```bash
sudo mkdir -p /etc/secrets
sudo install -m 600 -o root -g root sa-key.json /etc/secrets/sa-key.json
```

### 2.2 Firestore + Document AI

기존 GCP 프로젝트에 이미 셋업됐으면 스킵. 신규면 `docs/HANDOVER.md` 9.1절 명령:

```bash
gcloud firestore databases create --location=asia-northeast3 --type=firestore-native
# 단일 / 복합 벡터 인덱스 2개 생성
# Document AI processor 생성 (Console UI)
```

processor 경로 메모: `projects/<num>/locations/<loc>/processors/<id>`

---

## Phase 3 — Repo 배치 + .env

```bash
sudo mkdir -p /opt/medical-event-rag
sudo chown $USER:$USER /opt/medical-event-rag
git clone <YOUR_REPO_URL> /opt/medical-event-rag
cd /opt/medical-event-rag
cp .env.example .env
vim .env
```

`.env` 핵심 항목:
```ini
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=asia-northeast3
EMBEDDING_MODEL=text-multilingual-embedding-002
LLM_MODEL=gemini-2.5-flash
DOCAI_PROCESSOR=projects/<num>/locations/<loc>/processors/<id>

FIRESTORE_DATABASE_ID=(default)
FIRESTORE_COLLECTION_NAME=medical_event_chunks

CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=7
BM25_TOP_K=10
HYBRID_ALPHA=0.6
SCORE_THRESHOLD=0.05
MAX_CONTEXT_TOKENS=4000
MAX_CONVERSATION_TURNS=5

REDIS_URL=redis://redis:6379/0
EMBEDDING_CACHE_TTL=604800
```

---

## Phase 4 — docker-compose 운영용 패치

`infra/docker-compose.yml` 두 가지 변경:

1. **포트를 localhost로 한정**
2. **GCP 인증을 SA 키로 전환**

```yaml
services:
  app:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    ports:
      - "127.0.0.1:8001:8000"
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

  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  redis_data:
```

---

## Phase 5 — 데이터 + 인덱싱

### 옵션 A — dev 머신(WSL)에서 산출물만 복사 (권장)

dev 머신:
```bash
tar czf rag-data.tar.gz data/documents data/bm25_index data/chunk_store
scp rag-data.tar.gz user@<운영서버-IP>:/tmp/
```

운영 서버:
```bash
cd /opt/medical-event-rag
tar xzf /tmp/rag-data.tar.gz
ls data/
```

Firestore는 GCP 측이라 어디서 인덱싱하든 같은 DB. 한 번만 채우면 됨.

### 옵션 B — 운영 서버에서 직접 인덱싱

Vertex AI 비용·시간 발생. 큰 인덱스면 옵션 A 권장.

```bash
cd /opt/medical-event-rag/infra
docker compose run --rm app python scripts/index_documents.py --full
```

---

## Phase 6 — RAG 컨테이너 기동

```bash
cd /opt/medical-event-rag/infra
docker compose up -d --build
docker compose logs app -f
```

`Pipeline ready.` 보이면 정상.

```bash
curl -s http://127.0.0.1:8001/admin/health
# → {"status":"ok"}
```

이 시점에 외부 IP로 8001 직접 접속은 의도적으로 안 됨.

---

## Phase 7 — Apache 리버스 프록시

### 7.1 모듈 확인

```bash
sudo dnf install -y httpd mod_ssl
sudo systemctl enable --now httpd
httpd -M | grep -E "proxy|headers|ssl"
# proxy_module, proxy_http_module, headers_module, ssl_module 모두 보여야 함
```

### 7.2 VirtualHost 생성

`/etc/httpd/conf.d/api.lungkorea.org.conf`:

```apache
<VirtualHost *:80>
    ServerName api.lungkorea.org
    ServerAdmin admin@lungkorea.org

    ErrorLog /var/log/httpd/api.lungkorea.org-error.log
    CustomLog /var/log/httpd/api.lungkorea.org-access.log combined

    ProxyRequests Off
    ProxyPreserveHost On

    # SSE 즉시 송신 핵심 옵션
    ProxyPass        / http://127.0.0.1:8001/ flushpackets=on timeout=300 keepalive=On
    ProxyPassReverse / http://127.0.0.1:8001/

    SetEnv proxy-sendchunked 1
    RequestHeader set X-Forwarded-Proto "http"
    RequestHeader set X-Forwarded-Host "api.lungkorea.org"

    Header set Cache-Control "no-cache, no-store, must-revalidate"

    # 첨부파일명에 한글/슬래시 허용
    AllowEncodedSlashes NoDecode
</VirtualHost>
```

### 7.3 검증 + 재시작

```bash
sudo apachectl configtest    # → "Syntax OK"
sudo systemctl restart httpd
```

### 7.4 외부 접근 테스트

```bash
curl -v http://api.lungkorea.org/admin/health
# 기대: HTTP 200 + {"status":"ok"}
```

### 7.5 SSE 즉시 송신 테스트

```bash
curl -N -X POST http://api.lungkorea.org/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"천식연구회 Workshop 2022","stream":true}' \
  --max-time 30
```

`data: {...}` 라인이 토큰 단위로 흘러나오면 정상. 한꺼번에 묶여 도착하면 `flushpackets=on`이 안 먹은 것 — vhost 설정 다시 확인.

---

## Phase 8 — TLS (Let's Encrypt) — 운영 강력 권장

HTTP만으로도 동작은 하지만, Laravel이 HTTPS면 mixed-content로 fetch 차단됩니다.

```bash
sudo dnf install -y epel-release
sudo dnf install -y certbot python3-certbot-apache
sudo certbot --apache -d api.lungkorea.org \
  --email admin@lungkorea.org --agree-tos --no-eff-email --redirect
```

이 명령이:
- 인증서 발급 → `/etc/letsencrypt/live/api.lungkorea.org/`
- HTTPS vhost 자동 추가 (`/etc/httpd/conf.d/api.lungkorea.org-le-ssl.conf`)
- HTTP → HTTPS 리다이렉트 자동 설정
- 자동 갱신 cron 등록

### 중요: HTTPS vhost에도 ProxyPass 옵션 보존 확인

```bash
sudo cat /etc/httpd/conf.d/api.lungkorea.org-le-ssl.conf
```

여기서 7.2의 `ProxyPass / http://127.0.0.1:8001/ flushpackets=on ...` 라인이 그대로 있어야 합니다. 빠져있으면 추가하고 `sudo systemctl restart httpd`.

확인:
```bash
curl -v https://api.lungkorea.org/admin/health
sudo certbot renew --dry-run    # 자동 갱신 시뮬레이션
```

---

## Phase 9 — Laravel 챗봇 측 .env 갱신 (선택)

기존 Laravel 호스트 그대로 두고 `.env`만:
```env
RAG_API_URL=https://api.lungkorea.org
```

```bash
php artisan config:clear
```

`ChatController.php`는 이전 작업분 그대로 사용. 추가 코드 변경 없음.

자체 UI(`https://api.lungkorea.org/`)만 쓸 거라면 이 phase 자체 스킵.

---

## Phase 10 — 검증 체크리스트

순서대로:

1. ✅ `curl https://api.lungkorea.org/admin/health` → `{"status":"ok"}`
2. ✅ JSON 모드:
   ```bash
   curl -s -X POST https://api.lungkorea.org/chat \
     -H 'Content-Type: application/json' \
     -d '{"query":"테스트","stream":false}' --max-time 60
   ```
3. ✅ SSE 모드 (토큰 단위 흘러야 정상):
   ```bash
   curl -N -X POST https://api.lungkorea.org/chat \
     -H 'Content-Type: application/json' \
     -d '{"query":"2026년 1월 행사","stream":true}'
   ```
4. ✅ 자체 UI: 브라우저로 `https://api.lungkorea.org/` → 채팅 동작 확인
5. ✅ Laravel 연동: 기존 챗봇 메시지 입력 → 정상 답변 + localStorage `chat_thread_id`에 `thread_<uuid>` 저장
6. ✅ 첨부파일 마커 등장 시 클릭 → RAG의 `/admin/download/...`로 이동해 정상 다운로드

---

## Phase 11 — 운영

| 작업 | 명령 |
|---|---|
| 로그 follow | `cd /opt/medical-event-rag/infra && docker compose logs app -f` |
| 에러 검색 | `docker compose logs app --since 30m \| grep -E "Error\|429\|Rerank"` |
| Apache 에러 로그 | `sudo tail -f /var/log/httpd/api.lungkorea.org-error.log` |
| Apache 액세스 로그 | `sudo tail -f /var/log/httpd/api.lungkorea.org-access.log` |
| 컨테이너 재시작 | `docker compose restart app` |
| 코드 갱신 후 | `git pull && docker compose up -d --build` |
| 인덱스 증분 추가 | `docker compose run --rm app python scripts/index_documents.py --add <post_id>` |
| 메타 url 업데이트만 | `python scripts/index_documents.py --add <post_id>` (data.json 수정 후) |
| Redis 백업 | `docker run --rm -v medical-event-rag_redis_data:/data -v $PWD:/backup alpine tar czf /backup/redis-$(date +%F).tgz /data` |
| 헬스체크 모니터링 | UptimeRobot 등에서 `https://api.lungkorea.org/admin/health` 5분 간격 |
| 인증서 갱신 | 자동 (cron). 수동 강제: `sudo certbot renew --force-renewal` |

---

## 트러블슈팅 체크리스트

| 증상 | 원인 / 해결 |
|---|---|
| `http://api.lungkorea.org/admin/health` → 503 | mod_proxy 모듈 미로드, 또는 SELinux `httpd_can_network_connect` 미설정 |
| 503 + `Permission denied: AH00957` 로그 | SELinux. `sudo setsebool -P httpd_can_network_connect 1` |
| HTTPS 적용 후 SSE가 묶여서 도착 | certbot이 HTTPS vhost에 ProxyPass 옵션 누락. 7.2의 `flushpackets=on` 라인 추가 |
| `Connection refused (8001)` | docker 컨테이너 미기동. `docker compose ps` 확인 |
| Vertex AI 401/403 | `/etc/secrets/sa-key.json` 권한 또는 SA 역할 부족 |
| Vertex AI 429 | quota 초과. Console에서 증액 신청 또는 모델 다운그레이드(`gemini-2.5-flash-lite`) |
| `RuntimeError: REDIS_URL is required` | redis 컨테이너 미기동 또는 `.env` 누락 |
| 첨부파일 다운로드 시 파일명 깨짐 | vhost에 `AllowEncodedSlashes NoDecode` 추가 (위 7.2에 포함됨) |
| firewalld 차단 의심 | `sudo firewall-cmd --list-all`에서 http/https 보이는지 |
| `[Vision] 429` 인덱싱 도중 | retry는 자동 (코드에 적용됨). 너무 자주면 quota 증액 |

---

## 진행 권장 순서

한 번에 다 하지 말고 단계별로 검증하면서:

1. **세션 1 — 서버 준비**: Phase 1 ~ 4
   - 검증: `docker compose ps` + `curl http://127.0.0.1:8001/admin/health`
2. **세션 2 — 데이터 + Apache**: Phase 5 ~ 7
   - 검증: `curl http://api.lungkorea.org/admin/health` + SSE 테스트
3. **세션 3 — TLS + 통합**: Phase 8 ~ 10
   - 검증: HTTPS + Laravel 통합 테스트

각 세션에서 막히면 거기서 멈추고 알려주세요. 특히 **SELinux**, **Apache flushpackets=on**, **certbot이 ssl vhost에 옮겨놓은 ProxyPass 옵션 보존 여부** 세 가지가 가장 자주 막히는 포인트입니다.

---

## 참조 파일

| 영역 | 파일:라인 |
|---|---|
| docker-compose 원본 | `infra/docker-compose.yml` |
| Dockerfile | `infra/Dockerfile` |
| RAG API 라우트 | `api/routes/chat.py`, `api/routes/search.py`, `api/routes/admin.py` |
| Laravel 프록시 컨트롤러 | `externalChat/ChatController.php` |
| 환경변수 키 목록 | `.env.example` |
| 인덱싱 스크립트 | `scripts/index_documents.py` |
| 기존 dev 가이드 | `docs/HANDOVER.md` 섹션 9 |
