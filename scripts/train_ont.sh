#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"

echo "Project root is: $project_root"

export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

echo ""

echo "[Step 1/1] Starting OnT trainer (requires OnT in lib) ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./lib/OnT/train_ont.py \
  -c ./lib/OnT/config.yaml

echo "Training DONE!"


