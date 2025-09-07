#!/usr/bin/env bash
set -euo pipefail

# TODO: 
#  - improve build script to ensure future versions (or small changes to the NHS API) can be dealt with
#  - ensure appropriate error handling (script should fail gracefully)
#  - errors and graceful failure should propagate to the build pipeline

# load environment variables (export -> load .env -> re-import)

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"

# grab the ALT snomed URL (in case we need to obtain the public SNOMED 2023 OWL file)
ALT_URL="${ALT_SNOMED_ONTOLOGY_URL:-}"

# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi

echo ""

echo "Preparing project data files (this may take a moment or two!) ..."

echo "Note, you do not have set NHS_API_KEY in .env an older, publicly available SNOMED version will be downloaded ... "
echo ""
echo "requires dependencies... see: environment.yml, requirements.txt, openjdk-17-jdk"

echo "STARTING ..."

# IF ALT_URL is set, then we can SKIP:

if [[ -n "$ALT_URL" ]]; then
  echo "[INFO] ALT_SNOMED_ONTOLOGY_URL is set. Skipping NHS TRUD download pipeline."
  echo "[INFO] Falling back to Zenodo copy from 2023."
  ./scripts/download_snomed_fallback.sh
  echo "COMPLETED! Check './data' dir for snomedct-international.owl"
  exit 0
fi

echo "DOWNLOADING: snomed-owl-toolkit-5.3.0-executable.jar ..."

wget -q https://github.com/IHTSDO/snomed-owl-toolkit/releases/download/5.3.0/snomed-owl-toolkit-5.3.0-executable.jar

echo "OKAY!"

echo "DOWNLOADING: SNOMED CT RF2 (latest release) ..."

# relies on NHS_API_KEY=... in '.env'
snomed_version="$(conda run -n "$AUTO_ENV_NAME" --no-capture-output python ./scripts/get_snomed.py | tail -n 1)"
echo "OKAY!"

rf2_zip="./data/SnomedCT_InternationalRF2_PRODUCTION_${snomed_version}T120000Z.zip"

if [[ ! -f "$rf2_zip" ]]; then
  echo "RF2 archive not found: $rf2_zip" >&2
  exit 1
fi

echo "Converting: SNOMED RF2 archive to .owl ..."

java -Xms4g --add-opens java.base/java.lang=ALL-UNNAMED \
  -jar snomed-owl-toolkit-5.3.0-executable.jar \
  -rf2-to-owl \
  -rf2-snapshot-archives "$rf2_zip" \
  -iri "http://snomed.info/sct/900000000000207008" \
  -version "$snomed_version"

echo "OKAY!"

snomed_owl_file="ontology-${snomed_version}.owl"

echo "Moving: './${snomed_owl_file}' to './data/snomedct-international.owl' ..."
mv  "./${snomed_owl_file}" ./data/snomedct-international.owl
echo "OKAY!"

echo "COMPLETED! Check './data' dir for snomedct-international.owl"
