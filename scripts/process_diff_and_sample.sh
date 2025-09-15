#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"


# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi


echo ""

echo "[Step 1/4] Loading SNOMED CT pre-processed entity lexicon and extracting the label set ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/entity_lexicon_to_label_mappings.py

echo "Extracted SNOMED CT label set from SNOMED CT entity lexicon."

echo "[Step 2/4] Loading MIRAGE entities and constructing the OOV entity list (no duplicates) ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/process_MIRAGE_get_entity_list.py \
  --input ./data/benchmark-questions-entities.jsonl \
  --output ./data

echo "Extracted MIRAGE entity set from MIRAGE benchmark-questions-entities.jsonl."

echo "[Step 3/4] Computing lexically disjoint entity set/list (strings-diff.json) between SNOMED and MIRAGE ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/compute_normalised_inter_and_diff.py \
  --input ./data/label_verbalisations.json \
  --compare ./data/MIRAGE-entity-list.json \
  --output-diff ./data

echo "Disjoint lexical mentions written to 'strings-diff.json' in './data'."

echo "[Step 4/4] Sampling N=50 from set(SNOMED CT labels).diff(set(MIRAGE labels)) ... "

echo "[SAMPLING PROCEDURE] Sampling is set to: $SAMPLING_PROCEDURE"

echo "[SAMPLING PROCEDURE] Sampling N=$SAMPLE_N OOV strings"

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/sample.py \
  --input ./data/strings-diff.json \
  --procedure $SAMPLING_PROCEDURE \
  --n $SAMPLE_N

echo "Sampling procedure... COMPLETE!"

if grep -q '^SAMPLING_PROCEDURE=random$' .env 2>/dev/null; then
  echo "[Warning] SAMPLING_PROCEDURE is set to 'random' in .env; YOU MUST MANUALLY ANNOTATE THE TARGETS!!"
  echo "[Warning] THIS WILL BREAK THE AUTOMATED BUILD SCRIPT..!"
  echo "[Warning] You should only be seeing this warning if you know what you're doing!"
fi

echo ""

echo "...DONE!"
