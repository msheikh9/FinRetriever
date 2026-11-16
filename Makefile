PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
# Fully pinned environment for `make install-locked` (docs/environment.md)
LOCK   ?= $(if $(filter Darwin,$(shell uname -s)),requirements/lock-macos-arm64.txt,requirements/lock-linux-x86_64-cpu.txt)

CORPUS_CONFIG    ?= configs/data/corpus_v1.yaml
TRAIN_CONFIG     ?= configs/train/biencoder_bge_base_smoke.yaml
QUERIES_CONFIG   ?= configs/train/data_prep_v1.yaml
# Subcommand for `make queries`: prepare (offline), generate --split train|validation [--confirm-api-use],
# build-pairs, check
QUERIES_ARGS     ?= prepare
BENCHMARK_CONFIG ?= configs/eval/benchmark.yaml
# Extra flags for `make download`, e.g. DOWNLOAD_ARGS=--dry-run
DOWNLOAD_ARGS    ?=
# Extra flags for `make corpus`, e.g. CORPUS_ARGS=--strict
CORPUS_ARGS      ?=
# Subcommand and flags for `make annotate`, e.g. ANNOTATE_ARGS=status
ANNOTATE_ARGS    ?= --help

.PHONY: help install install-dev install-locked test lint format \
        download corpus queries annotate baselines train-biencoder benchmark

help:
	@echo "FinRetriever Makefile targets:"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install runtime + dev dependencies and the package in editable mode"
	@echo "  install-locked  Install the fully pinned tested environment (LOCK=requirements/lock-...)"
	@echo "  test            Run the pytest suite"
	@echo "  lint            Run ruff over the repo"
	@echo "  format          Run black and ruff --fix"
	@echo "  download        Download raw SEC 10-K filings into data/raw (needs SEC_USER_AGENT)"
	@echo "  corpus          Build the passage corpus + manifest from data/raw (offline)"
	@echo "  queries         Training-data preparation and synthetic pairs (QUERIES_ARGS=\"prepare\", ...)"
	@echo "  annotate        Gold-set annotation tool (ANNOTATE_ARGS=\"status\", ...)"
	@echo "  baselines       Run BM25 and zero-shot dense baselines"
	@echo "  train-biencoder Fine-tune the bi-encoder"
	@echo "  benchmark       Run the final retrieval benchmark"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps

install-locked:
	$(PIP) install -r $(LOCK)
	$(PIP) install -e . --no-deps

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

download:
	$(PYTHON) scripts/01_download_filings.py --config $(CORPUS_CONFIG) $(DOWNLOAD_ARGS)

corpus:
	$(PYTHON) scripts/01_build_corpus.py --config $(CORPUS_CONFIG) $(CORPUS_ARGS)

queries:
	$(PYTHON) scripts/02_generate_queries.py $(QUERIES_ARGS) --config $(QUERIES_CONFIG)

annotate:
	$(PYTHON) scripts/03_annotate_gold_set.py $(ANNOTATE_ARGS)

baselines:
	$(PYTHON) scripts/04_run_baselines.py --config $(BENCHMARK_CONFIG)

train-biencoder:
	$(PYTHON) scripts/05_train_biencoder.py --config $(TRAIN_CONFIG) $(TRAIN_ARGS)

benchmark:
	$(PYTHON) scripts/06_run_benchmark.py --config $(BENCHMARK_CONFIG)
