#!/usr/bin/env bash
set -euo pipefail

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"

echo "Project root is: $project_root"


# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi


export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

echo ""

echo "[Step 1/1] Building OnT dataset for training OnT models as used within the paper ... "

#echo "" | conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./lib/OnT/normalization/modELNormalize.py \
#  --input ./data/snomedct-international.owl \
#  --output ./data/ont_dataset

echo "[Step 2/2] Handling role_inverse.json edge case (creating a single obj in role_inverse.json file)..."

touch ./data/ont_dataset/OnT/role_inverse.json

echo "{}" >> ./data/ont_dataset/OnT/role_inverse.json

echo "Built OnT dataset."

echo "...DONE!"
