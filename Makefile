SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

IMAGE ?= uom-thesis:gpu

.PHONY: init env snomed process-snomed process-mirage sample build-eval eval hit-train hit-data ont-train ont-data clean docker-build-eval docker all default docker models embeddings single-target multi-target no-rag sbert-rag hit-rag ont-rag docker-build docker-init docker-test docker-snomed docker-mirage docker-sample tests docker-embeddings docker-models docker-single-target docker-multi-target docker-axioms docker-no-rag docker-hit-rag docker-ont-rag docker-sbert-rag

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
	./scripts/verify-annotaton.sh

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

models:
	@echo "[MODELS] Fetching both SNOMED-tuned and pretrained encoders ..."
	./scripts/download_models.sh

embeddings:
	@echo "[EMBEDDINGS] Generating embeddings ... "
	./scripts/produce_embeddings.sh

single-target:
	@echo "[SINGLE-TARGET] Running single target experiments ... "
	./scripts/single_target.sh

multi-target:
	@echo "[MULTI-TARGET] Running multiple target experiments ... "
	./scripts/multitarget.sh

verb-axioms:
	@echo "[VERB-AXIOMS] Verbalising axioms for RAG ... "
	./scripts/verbalise_axioms.sh

no-rag:
	@echo "[NO-RAG] Running no rag experiment ... "
	./scripts/no-rag.sh

sbert-rag:
	@echo "[SBERT-RAG] Running SBERT rag experiment ... "
	./scripts/sbert-rag.sh

hit-rag:
	@echo "[HiT-RAG] Running HiT rag experiment ... "
	./scripts/hit-rag.sh

ont-rag:
	@echo "[OnT-RAG] Running OnT rag experiment ... "
	./scripts/ont-rag.sh

clean:
	@echo "[CLEAN] Removing generated data..."
	./scripts/clean.sh

tests:
	@echo "[TESTS] Running tests..."
	pytest

docker-build:
	docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 --build-arg GPU=1 --build-arg ENV_NAME=knowledge-retrieval-env -t uom-thesis:gpu .

docker-test:
	docker run --rm -it --gpus all -v "$PWD":/work --env-file ./.env uom-thesis:gpu bash -lc 'python -c "import torch; print(\"CUDA:\", torch.cuda.is_available())";'

docker-init:	
	docker run --rm -it -v $$PWD:/work -w /work $(IMAGE) make init

docker-snomed:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make snomed
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make process-snomed

docker-mirage:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make process-mirage

docker-sample:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make sample

docker-eval:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make build-eval

docker-models:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make models

docker-embeddings:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make embeddings

docker-single-target:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make single-target

docker-multi-target:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make multi-target

docker-axioms:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make verb-axioms

docker-no-rag:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make no-rag

docker-sbert-rag:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make sbert-rag

docker-hit-rag:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make hit-rag

docker-ont-rag:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make ont-rag

docker-make:
	docker run --rm -it --gpus all --env-file ./.env -v $$PWD:/work -w /work $(IMAGE) make docker-all

docker-all: init snomed process-snomed process-mirage sample models embeddings single-target multi-target verb-axioms no-rag sbert-rag hit-rag ont-rag
	@echo "[ALL] Finished running entire pipeline! ... for docker"

all: init env snomed process-snomed process-mirage sample models embeddings single-target multi-target verb-axioms no-rag sbert-rag hit-rag ont-rag
	@echo "[ALL] Finished running entire pipeline!"

default: all
