SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

.PHONY: env snomed eval hit ont clean docker-build-eval docker all default docker

env:
	@echo "[ENV] Bootstrapping environment dependencies..."
	./scripts/bootstrap.sh

snomed:
	@echo "[SNOMED] Downloading SNOMED CT (ensure NHS_API_KEY is set in .env)"
	./scripts/download_snomed.sh

eval:
	@echo "[EVAL] Building evaluation dataset..."
	./scripts/process_snomed.sh
	./scripts/download_and_process_mirage.sh
	./scripts/process_diff_and_sample.sh
	./scripts/verify-annotation.sh
	./scripts/build_eval_data.sh

hit:
	@echo "[HIT] Building HiT dataset (running pipeline)..."
	./scripts/build_hit_data.sh

ont:
	@echo "[ONT] Building OnT dataset (running pipeline)..."
	./scripts/build_ont_data.sh

clean:
	@echo "[CLEAN] Removing generated data..."
	./scripts/clean.sh

docker-build-eval: env snomed eval
	@echo "[DOCKER-BUILD-EVAL] Environment setup & finished building evaluation data..."

docker: env
	@echo "[DOCKER] Finished environment setup"

all: env snomed eval hit ont
	@echo "[ALL] Finished full pipeline!"

default: all
