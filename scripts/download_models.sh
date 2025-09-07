#!/usr/bin/env bash
set -euo pipefail

# Load .env if present (provides SNOMED_MODELS_URL, ONT_PRETRAINED_MODEL_URL)
if [[ -f .env ]]; then
  set -a
  . .env
  set +a
fi

MODELS_DIR="./models"
SNOMED_URL="${SNOMED_MODELS_URL:-https://drive.google.com/file/d/1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT/view}"
ONT_URL="${ONT_PRETRAINED_MODEL_URL:-https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view}"

mkdir -p "$MODELS_DIR"

echo "Downloading SNOMED-tuned encoders ... "

./scripts/download_gdrive_zip.sh "$SNOMED_URL" "$MODELS_DIR"

echo "Downloading pretrained OnT encoders ... "

./scripts/download_gdrive_zip.sh "$ONT_URL" "$MODELS_DIR"

echo "...Done! Models unpacked to $MODELS_DIR"