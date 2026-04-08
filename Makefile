.PHONY: install dev setup index run test lint clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

setup:
	python scripts/setup_gcp.py

index:
	python scripts/index_documents.py

bm25:
	python scripts/build_bm25_index.py

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

format:
	ruff format src/ api/ tests/

docker-up:
	cd infra && docker compose up -d

docker-down:
	cd infra && docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
