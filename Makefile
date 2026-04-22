.PHONY: help venv install dev run lint fmt docker-build docker-run docker-push frontend-dev frontend-build clean

IMAGE    ?= ghcr.io/muriki1234/ai-football-assistant
TAG      ?= latest
PORT     ?= 8000

help:
	@echo "Common targets:"
	@echo "  make install         Install server deps in server/.venv"
	@echo "  make dev             Run server with autoreload on :$(PORT)"
	@echo "  make run             Run server in production mode on :$(PORT)"
	@echo "  make docker-build    Build Docker image ($(IMAGE):$(TAG))"
	@echo "  make docker-run      Run image with /workspace bind-mounted"
	@echo "  make docker-push     Push image to ghcr.io"
	@echo "  make frontend-dev    Start the Vite dev server"
	@echo "  make frontend-build  Build frontend for prod"
	@echo "  make clean           Remove caches and build artefacts"

venv:
	python3 -m venv server/.venv

install: venv
	./server/.venv/bin/pip install --upgrade pip
	./server/.venv/bin/pip install -r server/requirements.txt

dev:
	./server/.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port $(PORT) --reload

run:
	./server/.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port $(PORT) --workers 1 --proxy-headers

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -it --gpus all \
		-p $(PORT):8000 \
		--env-file server/.env \
		-v $(PWD)/.dev-workspace:/workspace \
		$(IMAGE):$(TAG)

docker-push:
	docker push $(IMAGE):$(TAG)

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf server/outputs/* server/uploads/* frontend/dist
