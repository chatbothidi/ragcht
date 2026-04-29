## 1. Docker 서버 기동·관리

### 1.1 첫 기동 (또는 코드 변경 후 재빌드)

cd ~/chatbot/infra
docker compose up -d --build

### 1.2 일반 기동·정지

docker compose -f ~/chatbot/infra/docker-compose.yml up -d        # 기동 (백그라운드)
docker compose -f ~/chatbot/infra/docker-compose.yml down         # 정지 + 컨테이너 제거
docker compose -f ~/chatbot/infra/docker-compose.yml stop         # 정지만 (컨테이너 유지)
docker compose -f ~/chatbot/infra/docker-compose.yml start        # 정지된 컨테이너 다시 시작

### 1.3 재시작 (메모리 캐시 새로고침)

docker compose -f ~/chatbot/infra/docker-compose.yml restart app

인덱싱 직후 BM25 / ChunkStore 메모리 새로고침에 필수.

## 2. 인덱싱 (임베딩 + Firestore + BM25 + ChunkStore)

### 2.1 전체 재인덱싱 (`--full`)

cd ~/chatbot
python scripts/index_documents.py --full


### 2.2 증분 인덱싱 (default, 새 post만)

cd ~/chatbot
python scripts/index_documents.py

### 2.3 특정 post 추가/재인덱싱 (`--add`)

cd ~/chatbot
python scripts/index_documents.py --add 12345 

`12345` 자리에 post 디렉터리명(=`data/documents/12345/`의 숫자 ID).

이미 인덱싱된 post를 `--add`하면 기존 청크 삭제 후 재처리 (data.json 수정 시 사용).

### 2.4 특정 post 삭제 (`--delete`)

cd ~/chatbot
python scripts/index_documents.py --delete 12345 

해당 post의 청크를 Firestore + ChunkStore + BM25에서 모두 제거.

---

## 3. 로그 확인(실시간)

docker compose -f ~/chatbot/infra/docker-compose.yml logs app -f

---

## 4. 배포 (git + requirements.lock)

### 4.1 코드 변경만 — dev → 운영

dev:

git add <changed_files>
git commit -m "..."
git push

운영 서버:

cd ~/chatbot
git pull
docker compose -f infra/docker-compose.yml up -d --build

### 4.2 의존성 추가 / 업데이트

dev:

# (1) requirements.txt에 추가 (사람이 읽는 문서)
echo "newpackage>=1.0" >> ~/chatbot/requirements.txt

# (2) 컨테이너 재빌드
cd ~/chatbot/infra
docker compose build --no-cache app
docker compose up -d

# (3) 동작 확인
docker compose exec app python -c "import newpackage"

# (4) 새 lock 추출 (반드시 컨테이너 안에서 freeze)
docker compose exec app pip freeze > ../requirements.lock

# (5) git commit + push
cd ..
git add requirements.txt requirements.lock
git commit -m "Add newpackage"
git push

운영 서버 — 4.1 흐름 그대로 (git pull + build).

### 4.3 lock 파일 초기 생성 (첫 셋업 시 1회)

cd ~/chatbot/infra
docker compose exec app pip freeze > ../requirements.lock
cd ..
git add requirements.lock
git commit -m "Pin Python deps to requirements.lock"
git push

Dockerfile은 `COPY requirements.lock` + `pip install -r requirements.lock`로 빌드. requirements.txt는 사람용 참고로만 유지.

### 4.4 배포 확인

운영 서버에서 빌드된 컨테이너의 의존성이 lock과 일치하는지:

docker compose -f ~/chatbot/infra/docker-compose.yml exec app pip freeze | diff - ~/chatbot/requirements.lock

차이 0이면 dev와 운영 환경 완전 일치.
