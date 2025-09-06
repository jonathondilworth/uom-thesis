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

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./lib/hierarchy_transformers/scripts/training/training_hit.py \
  -c ./lib/hierarchy_transformers/scripts/training/hit/config_hit.yaml

echo "Training DONE!"

