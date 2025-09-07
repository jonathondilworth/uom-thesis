SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

.PHONY: init env snomed process-snomed process-mirage sample build-eval eval hit ont clean docker-build-eval docker all default docker

init:
	@echo "[INIT] Initialising project enviornment variables and .env"
	./scripts/init.sh

env:
	@echo "[ENV] Bootstrapping environment dependencies..."
	./scripts/bootstrap.sh

snomed:
	@echo "[SNOMED] Downloading SNOMED CT (ensure NHS_API_KEY is set in .env)"
	./scripts/download_snomed.sh

process-snomed:
	@echo "[PROCESS-SNOMED] Processing SNOMED CT..."
	./scripts/process_snomed.sh

process-mirage:
	@echo "[PROCESS-MIRAGE] Processing MIRAGE..."
	./scripts/download_and_process_mirage.sh

sample:
	@echo "[SAMPLE] Sampling processed datasets (SNOMED CT and MIRAGE)..."
	./scripts/process_diff_and_sample.sh
	./verify-annotaton.sh

build-eval:
	@echo "[BUILD-EVAL] Building evaluation data..."
	./scripts/build_eval_data.sh

eval:
	@echo "[EVAL] Building evaluation dataset from scracth..."
	./scripts/process_snomed.sh
	./scripts/download_and_process_mirage.sh
	./scripts/process_diff_and_sample.sh
	./scripts/verify-annotation.sh
	./scripts/build_eval_data.sh

hit-data:
	@echo "[HIT] Building HiT dataset (running pipeline)..."
	./scripts/build_hit_data.sh

ont-data:
	@echo "[ONT] Building OnT dataset (running pipeline)..."
	./scripts/build_ont_data.sh

hit-train:
	@echo "[HIT] Starting HiT training..."
	./scripts/train_hit.sh

ont-train:
	@echo "[ONT] Starting OnT training..."
	./scripts/train_ont.sh

clean:
	@echo "[CLEAN] Removing generated data..."
	./scripts/clean.sh

docker-build:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t uom-thesis-dev .

docker-run-dry:
	docker run --rm -it --env-file ./.env -v $$PWD:/work -w /work uom-thesis-dev

docker-make:
	docker run --rm -it --env-file ./.env -v $$PWD:/work -w /work uom-thesis-dev make

all: init env snomed eval hit-data ont-data
	@echo "[ALL] Finished running data prep pipeline!"

default: all
