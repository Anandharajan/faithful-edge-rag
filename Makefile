.PHONY: install lint typecheck test check services-up services-down api

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

check: lint typecheck test

services-up:
	docker compose up -d postgres qdrant minio redis prometheus grafana

services-down:
	docker compose down

api:
	uvicorn faithful_edge_rag.api.main:app --reload

