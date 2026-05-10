# ══════════════════════════════════════════════════════════
# Clary — Ask First Health Intelligence
# Makefile
# ══════════════════════════════════════════════════════════

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Config ─────────────────────────────────────────────────
PYTHON        := python3
PIP           := $(PYTHON) -m pip
APP_DIR       := app
STREAMLIT_APP := $(APP_DIR)/main.py
TEST_DIR      := .
DOCKER_COMPOSE := docker compose

# Colours
RESET  := \033[0m
BOLD   := \033[1m
BLUE   := \033[34m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m

# ══════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════

.PHONY: install
install: ## Install all Python dependencies
	@echo -e "$(BLUE)Installing dependencies…$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo -e "$(GREEN)✓ Dependencies installed$(RESET)"

.PHONY: install-dev
install-dev: install ## Install with dev extras (black, mypy, ruff)
	$(PIP) install black ruff mypy types-redis pytest pytest-asyncio pytest-cov
	@echo -e "$(GREEN)✓ Dev dependencies installed$(RESET)"

.PHONY: env
env: ## Copy .env.example to .env (does not overwrite existing)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo -e "$(GREEN)✓ .env created from .env.example$(RESET)"; \
		echo -e "$(YELLOW)⚠  Fill in OPENAI_API_KEY and other secrets$(RESET)"; \
	else \
		echo -e "$(YELLOW).env already exists — skipping$(RESET)"; \
	fi

.PHONY: data-dir
data-dir: ## Create local data directories
	mkdir -p data/chroma data/neo4j data/redis eval/reports
	@echo -e "$(GREEN)✓ Data directories created$(RESET)"

# ══════════════════════════════════════════════════════════
# DOCKER INFRASTRUCTURE
# ══════════════════════════════════════════════════════════

.PHONY: docker-up
docker-up: ## Start all infrastructure services (Neo4j, ChromaDB, Redis, PostgreSQL)
	@echo -e "$(BLUE)Starting infrastructure…$(RESET)"
	$(DOCKER_COMPOSE) -f infra/docker-compose.yml up -d
	@echo -e "$(GREEN)✓ Services started$(RESET)"
	@echo "  Neo4j:    http://localhost:7474"
	@echo "  ChromaDB: http://localhost:8000"
	@echo "  Redis:    redis://localhost:6379"

.PHONY: docker-down
docker-down: ## Stop all infrastructure services
	@echo -e "$(YELLOW)Stopping infrastructure…$(RESET)"
	$(DOCKER_COMPOSE) -f infra/docker-compose.yml down
	@echo -e "$(GREEN)✓ Services stopped$(RESET)"

.PHONY: docker-logs
docker-logs: ## Tail logs from all infrastructure services
	$(DOCKER_COMPOSE) -f infra/docker-compose.yml logs -f

.PHONY: docker-reset
docker-reset: docker-down ## Wipe all infrastructure data and restart fresh
	$(DOCKER_COMPOSE) -f infra/docker-compose.yml down -v
	rm -rf data/chroma data/neo4j
	$(MAKE) data-dir docker-up

# ══════════════════════════════════════════════════════════
# APPLICATION
# ══════════════════════════════════════════════════════════

.PHONY: streamlit
streamlit: ## Launch the Streamlit app (frontend)
	@echo -e "$(BLUE)Starting Streamlit on http://localhost:8501$(RESET)"
	streamlit run $(STREAMLIT_APP) \
		--server.port 8501 \
		--server.headless false \
		--browser.gatherUsageStats false

.PHONY: run
run: docker-up streamlit ## Start infrastructure + Streamlit in one command

.PHONY: run-worker
run-worker: ## Start the Celery async pattern analysis worker
	@echo -e "$(BLUE)Starting Celery worker…$(RESET)"
	celery -A agents.worker worker \
		--loglevel=info \
		--concurrency=4 \
		--queues=pattern_analysis

.PHONY: run-flower
run-flower: ## Start Celery Flower monitoring dashboard
	celery -A agents.worker flower --port=5555

# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

.PHONY: ingest
ingest: ## Load the synthetic dataset into all stores (ChromaDB + Neo4j)
	@echo -e "$(BLUE)Ingesting askfirst_synthetic_dataset…$(RESET)"
	$(PYTHON) -m eval.runner --ingest data/askfirst_synthetic_dataset.json
	@echo -e "$(GREEN)✓ Dataset ingested$(RESET)"

.PHONY: seed-semantic
seed-semantic: ## Seed the medical knowledge base in ChromaDB
	@echo -e "$(BLUE)Seeding semantic store…$(RESET)"
	$(PYTHON) -c "from memory.semantic_store import SemanticStore; import asyncio; asyncio.run(SemanticStore().seed(force=True))"
	@echo -e "$(GREEN)✓ Semantic store seeded$(RESET)"

# ══════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════

.PHONY: test
test: ## Run all unit tests (no infrastructure required)
	@echo -e "$(BLUE)Running unit tests…$(RESET)"
	$(PYTHON) -m pytest -m "not integration and not slow" \
		--cov=agents --cov=temporal --cov=memory --cov=schemas \
		--cov-report=term-missing \
		-q

.PHONY: test-integration
test-integration: ## Run integration tests (requires live Neo4j + Redis + ChromaDB)
	@echo -e "$(YELLOW)Running integration tests (requires infrastructure)…$(RESET)"
	$(PYTHON) -m pytest -m "integration" -v

.PHONY: test-all
test-all: test test-integration ## Run all tests including integration

.PHONY: test-agents
test-agents: ## Run only agent tests
	$(PYTHON) -m pytest agents/tests/ -v

.PHONY: test-temporal
test-temporal: ## Run only temporal reasoning tests
	$(PYTHON) -m pytest temporal/tests/ -v

# ══════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════

.PHONY: eval
eval: ## Run the full evaluation harness against 8 golden patterns
	@echo -e "$(BLUE)Running evaluation harness…$(RESET)"
	$(PYTHON) -m eval.runner \
		--dataset data/askfirst_synthetic_dataset.json \
		--output eval/reports/latest.json
	@echo -e "$(GREEN)✓ Evaluation complete — see eval/reports/latest.json$(RESET)"

.PHONY: eval-report
eval-report: ## Print the latest evaluation report
	@cat eval/reports/latest.json | $(PYTHON) -m json.tool

# ══════════════════════════════════════════════════════════
# CODE QUALITY
# ══════════════════════════════════════════════════════════

.PHONY: format
format: ## Auto-format all Python files with Black
	@echo -e "$(BLUE)Formatting with Black…$(RESET)"
	black .
	@echo -e "$(GREEN)✓ Formatting done$(RESET)"

.PHONY: lint
lint: ## Lint with Ruff (fast) + check formatting
	@echo -e "$(BLUE)Linting with Ruff…$(RESET)"
	ruff check . --fix
	black --check .
	@echo -e "$(GREEN)✓ Lint passed$(RESET)"

.PHONY: typecheck
typecheck: ## Static type checking with mypy
	@echo -e "$(BLUE)Running mypy…$(RESET)"
	mypy agents/ memory/ schemas/ temporal/ graph/ --ignore-missing-imports
	@echo -e "$(GREEN)✓ Type check passed$(RESET)"

.PHONY: check
check: lint typecheck test ## Run full quality gate (lint + types + tests)
	@echo -e "$(GREEN)$(BOLD)✓ All checks passed$(RESET)"

# ══════════════════════════════════════════════════════════
# CLEAN
# ══════════════════════════════════════════════════════════

.PHONY: clean
clean: ## Remove build artifacts, caches, and temp files
	@echo -e "$(YELLOW)Cleaning…$(RESET)"
	find . -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"        -delete 2>/dev/null || true
	find . -type f -name "*.pyo"        -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
	@echo -e "$(GREEN)✓ Clean done$(RESET)"

.PHONY: clean-data
clean-data: ## Remove all local data (ChromaDB, Neo4j exports) — DESTRUCTIVE
	@echo -e "$(RED)⚠  This will delete all local vector store and graph data!$(RESET)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -rf data/chroma data/neo4j
	$(MAKE) data-dir
	@echo -e "$(GREEN)✓ Data directories reset$(RESET)"

# ══════════════════════════════════════════════════════════
# DEPLOYMENT
# ══════════════════════════════════════════════════════════

.PHONY: docker-build
docker-build: ## Build Docker images for app and worker
	docker build -f infra/Dockerfile.app    -t clary-app:latest .
	docker build -f infra/Dockerfile.worker -t clary-worker:latest .
	@echo -e "$(GREEN)✓ Docker images built$(RESET)"

.PHONY: docker-push
docker-push: ## Push images to container registry (set REGISTRY env var)
	docker tag clary-app:latest    $(REGISTRY)/clary-app:latest
	docker tag clary-worker:latest $(REGISTRY)/clary-worker:latest
	docker push $(REGISTRY)/clary-app:latest
	docker push $(REGISTRY)/clary-worker:latest

# ══════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo -e "$(BOLD)Clary — Ask First Health Intelligence$(RESET)"
	@echo -e "$(BLUE)Usage: make [target]$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""