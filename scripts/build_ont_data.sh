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

echo "[Step 1/1] Building OnT dataset for training OnT models as used within the paper ... "

echo "" | conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./lib/OnT/normalize/modELNormalize.py \
  --input ./data/snomedct-international.owl \
  --output ./data/ont_dataset

echo "Built OnT dataset."

echo "...DONE!"