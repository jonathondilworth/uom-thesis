#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

echo "Loading annotated data and fetching ancestors ... this WILL take a while!"

echo "STARTING ..."

echo "[Step 1/3] Loading IRIs from annotations ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python extract_iris_from_annotations.py \
  --input ./data/manually-annotated.json

echo "IRIs written to ./data/dataset_iri_list.json."

echo "[Step 2/3] Preparing data_out with ancestors for each annotated IRI  ... "

INPUT_JSON="./data/dataset_iri_list.json"

mkdir -p ./data_out

for iri in $(jq -r '.[]' "$INPUT_JSON"); do
    
    iri_id=$(basename "$iri")

    echo "Processing $iri_id ..."

    python snomed_ancestors.py \
        --input ./data/snomed_inferred_hasse.ttl \
        --iri "$iri" \
        --json-out "./data_out/${iri_id}.json" \
        --enforce-direct \
        --include-anonymous \
        --include-equivalents
done

echo "[Step 3/3] Merging ancestors into final evaluation dataset  ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python merge_ancestors.py \
  --input-json ./data/manually-annotated.json \
  --merge-dir ./data_out \
  --output-dir ./data

echo "COMPLETED! Check './data' dir for 'evaluation_dataset.json'"