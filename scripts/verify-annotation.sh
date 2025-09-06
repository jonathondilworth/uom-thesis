#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

DATA_DIR="./data"
READY="${DATA_DIR}/ready-for-annotation.json"
MANUAL="${DATA_DIR}/manually-annotated.json"
MANUAL_ALT="${DATA_DIR}/manually-annotated-2.json"

# sanity check
echo "Annotation procedure is ${SAMPLING_PROCEDURE:-<unset>}"
echo ""

if [[ "${SAMPLING_PROCEDURE:-}" == "deterministic" ]]; then
  echo "[Step 1/1] Verifying deterministic annotation procedure ... "
  if [[ -f "$MANUAL" ]]; then
    echo "[VERIFICATION PROCEDURE] Manually Annotated Exists... DONE!"
    exit 0
  else
    echo "[ERROR] The sampling procedure is set to deterministic, but no file exists at './data/manually-annotated.json'!"
    exit 1
  fi

elif [[ "${SAMPLING_PROCEDURE:-}" == "random" ]]; then
  echo "[Step 1/2] Checking for '${READY}' ... "
  if [[ ! -f "$READY" ]]; then
    echo "[WARING] SAMPLING_PROCEDURE=random, but '${READY}' does not exist."
    exit 1
  fi

  echo "[Step 2/2] Renaming '${READY}' into '${MANUAL}' (handling collisions) ... "

  if [[ -f "$MANUAL" ]]; then
    if [[ -f "$MANUAL_ALT" ]]; then
      echo "[WARNING] '${MANUAL}' and '${MANUAL_ALT}' already exist. Refusing to overwrite."
      echo "[WARNING] Move or remove one of them, then rerun."
      exit 1
    fi
    echo "[WARNING] '${MANUAL}' already exists; renaming new file to '${MANUAL_ALT}'."
    mv -f -- "$READY" "$MANUAL_ALT"
  else
    mv -f -- "$READY" "$MANUAL"
  fi

  echo "[VERIFICATION PROCEDURE] Renamed \`ready-for-annotation.json\` to \`manually-annotated.json\`"
  echo "DONE!"
  exit 0

else
  echo "[WARN] SAMPLING_PROCEDURE must be 'deterministic' or 'random' (got: '${SAMPLING_PROCEDURE:-<unset>}')."
  exit 1
fi