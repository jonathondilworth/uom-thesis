#!/usr/bin/env bash
set -euo pipefail

# WARNING: this script should only run when NHS_API_KEY is not set in .env

set -a
. .env
set +a

# for this backup script:
DATA_DIR="./data"
EXPECTED_SNOMED_LOC="${DATA_DIR}/snomedct-international.owl"
TMP_DIR="$(mktemp -d)"

# grab the ALT snomed URL
ALT_URL="${ALT_SNOMED_ONTOLOGY_URL:-}"

# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$ALT_URL" ]]; then
  echo "[ERROR] ALT_SNOMED_ONTOLOGY_URL is not set. Please check your .env"
  exit 1
fi

# in case data does not exist yet (it likely doesn't when this script is ran)
mkdir -p "$DATA_DIR"

echo "Downloading SNOMED CT fallback ontology from:"
echo "       $ALT_URL"

curl -L "$ALT_URL" -o "${TMP_DIR}/ontologies.zip"

echo "Unzipping archive..."

unzip -q "${TMP_DIR}/ontologies.zip" -d "$TMP_DIR"

# move snomed.owl to the expected locationn
SNOMED_PATH="$(find "$TMP_DIR" -type f -name 'snomed.owl' | head -n1)"
if [[ -z "$SNOMED_PATH" ]]; then
  echo "[ERROR] Could not find snomed.owl in the downloaded archive."
  exit 1
fi

echo "Moving snomed.owl to ${EXPECTED_SNOMED_LOC}"
mv "$SNOMED_PATH" "$EXPECTED_SNOMED_LOC"

echo "Cleaning up temp files..."
rm -rf "$TMP_DIR"

echo "[DONE] SNOMED CT fallback ontology available at ${EXPECTED_SNOMED_LOC}"