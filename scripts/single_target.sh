#!/usr/bin/env bash
set -euo pipefail

# load environment variables (export -> load .env -> re-import)

set -a
. .env
set +a

# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/single_target.py

echo "DONE!"
