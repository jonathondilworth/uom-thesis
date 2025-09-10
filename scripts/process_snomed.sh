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

# ADDED TO SUPPORT DOCKER:

# Resolve a ROBOT runner
ROBOT_LOCAL="./robot/bin/robot"
if [[ -x "$ROBOT_LOCAL" ]]; then
  ROBOT="$ROBOT_LOCAL"
elif command -v robot >/dev/null 2>&1; then
  ROBOT="$(command -v robot)"
elif [[ -n "${ROBOT_JAR:-}" && -f "$ROBOT_JAR" ]]; then
  ROBOT=(java -jar "$ROBOT_JAR")
else
  echo "[ERROR] ROBOT not found: tried ./robot/bin/robot, 'robot' on PATH, and \$ROBOT_JAR." >&2
  exit 1
fi
# END OF DOCKER SUPPORT

echo "[Step 1/4] Extracting SNOMED CT entity lexicon (requires HierarchyTransformers) ... "

echo "" | conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/load_taxonomy.py

echo "Extracted SNOMED CT entity lexicon."

echo "[Step 2/4] Preprocessing SNOMED CT entity lexicon ... "

conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/preprocess_entity_lexicon.py \
  --input ./data/entity_lexicon.json \
  --output ./data/preprocessed_entity_lexicon.json \
  --strip-parens \
  --to-lower-case

echo "PRE-PROCESSING DONE!"

echo "[Step 3/4] Converting SNOMED OWL to TTL ... "

alias robot="./robot/bin/robot"

# for debugging
VERY_VERBOSE=1

# in some cases, we will need to use .ttl (due to an issue with the conversion from RF2 -> OWL)

"${ROBOT[@]}" convert \
  --input ./data/snomedct-international.owl \
  --output ./data/snomedct-international.ttl

echo "CONVERSION FINITO!."

echo "[Step 4/4] Reasoning to produce inferred view via ELK (requires ROBOT, see bootstrap.sh) ... "

"${ROBOT[@]}" reason \
  --reasoner ELK \
  --input ./data/snomedct-international.owl \
  --equivalent-classes-allowed all \
  --remove-redundant-subclass-axioms true \
  --output ./data/snomed_inferred_hasse.ttl

echo "REASONING DONE!"

echo ""

echo "Finishing processing SNOMED CT"
