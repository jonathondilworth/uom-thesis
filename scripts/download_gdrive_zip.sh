#!/usr/bin/env bash
set -euo pipefail

# usage: dl_gdrive_zip.sh "<GDRIVE_URL_OR_ID>" "<DEST_DIR>"

# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

# ensure neccesary arguments are provided
if [[ $# -lt 2 ]]; then
  echo "[ERROR] Usage: $0 <gdrive-url-or-id> <dest-dir>"
  exit 1
fi

# args
SRC="$1"
DEST_DIR="$2"

# check that gdown has been installed (TODO: ensure this is set-up during boostrap)
if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown (local) ... "
  conda run -n "$AUTO_ENV_NAME" --no-capture-output python -m pip install --quiet --no-input gdown || conda run -n "$AUTO_ENV_NAME" --no-capture-output pip install --quiet gdown
fi

mkdir -p "$DEST_DIR"

TMPDIR="$(mktemp -d)"
ZIP_PATH="${TMPDIR}/payload.zip"

echo "Downloading from Google Drive ... "

if [[ "$SRC" =~ ^https?:// ]]; then
  conda run -n "$AUTO_ENV_NAME" --no-capture-output gdown "$SRC" -O "$ZIP_PATH"
else
  conda run -n "$AUTO_ENV_NAME" --no-capture-output gdown  --id "$SRC" -O "$ZIP_PATH"
fi

echo "Unzipping into $DEST_DIR"

unzip -o -q "$ZIP_PATH" -d "$DEST_DIR"

echo "Cleaning up temp files"

rm -rf "$TMPDIR"

echo "Unpacked zip to $DEST_DIR ... DONE!"