#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"

echo "Project root is: $project_root"

# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

echo ""

echo "[Step 1/2] Building HiT dataset (mixed) for training HiT models as used within the paper ... "

echo "" | conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/load_taxonomy.py \
  --ontology ./data/snomedct-international.owl \
  --training-data ./data/hit_dataset

echo "Extracted SNOMED CT entity lexicon."

echo "[Step 2/2] Preprocessing SNOMED CT entity lexicon(s) ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/preprocess_entity_lexicon.py \
  --input ./data/hit_dataset/multi/entity_lexicon.json \
  --output ./data/hit_dataset/multi/local_entity_lexicon.json \
  --strip-parens \
  --to-lower-case

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/preprocess_entity_lexicon.py \
  --input ./data/hit_dataset/mixed/entity_lexicon.json \
  --output ./data/hit_dataset/mixed/local_entity_lexicon.json \
  --strip-parens \
  --to-lower-case

echo "...DONE!"
