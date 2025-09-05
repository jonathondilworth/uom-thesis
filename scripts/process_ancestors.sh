#!/bin/bash

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