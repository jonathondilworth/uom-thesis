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

echo "[Step 1/1] Starting HiT trainer (requires HierarchyTransformers) ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python -m ./lib/hierarchy_transformers/scripts/train_hit.py \
  -c ./lib/hierarchy_transformers/scripts/config.yaml

echo "Training DONE!"

