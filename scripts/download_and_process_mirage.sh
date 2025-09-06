#!/usr/bin/env bash
set -euo pipefail

BENCH_LOC="./data/benchmark.json"

# load environment variables (export -> load .env -> re-import)

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"

echo "[Step 1/4] Downloading MIRAGE benchmark data as 'benchmark.json' (saving to ./data) ... "

wget -qO "$BENCH_LOC" https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/refs/heads/main/benchmark.json

echo "DONE."

echo "[Step 2/4] Extracting questions from './data/benchmark.json ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/load_benchmark_extract_questions.py \
    --input "./data/benchmark.json" \
    --output-dir "./data"

echo "Extracted questions for benchmark."

echo "[Step 3/4] Preparing SpaCy dependencies for running NER on './data/benchmark-questions.json ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python -m spacy download en_core_web_sm

conda run -n "$AUTO_ENV_NAME" --no-capture-output pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz --no-deps

echo "Success!"

echo "[Step 4/4] Running NER on './data/benchmark-questions.json to build ./data/benchmark-questions-with-entities.json ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/load_questions_and_extract_entities.py \
    --input ./data/benchmark-questions.json \
    --output-dir ./data \
    --extract-biomed \
    --ner scispacy

echo "DONE!"
