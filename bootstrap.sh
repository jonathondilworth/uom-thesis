#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
ENV_FILE="${ENV_FILE:-environment.yml}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
INSTALLER="${INSTALLER:-/tmp/miniconda.sh}"
UPDATE_MINICONDA="${UPDATE_MINICONDA:-false}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

# --
# bootstrap script
# --

# run as root (TODO: consider removing this)
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then 
    SUDO="sudo"; 
else 
    SUDO=""; 
fi

# removing the installer from /tmp/ upon cleanup
cleanup() { [[ -f "$INSTALLER" ]] && rm -f "$INSTALLER" || true; }
trap cleanup EXIT

# sanity check
echo "OS Version: $(lsb_release -sd 2>/dev/null || echo unknown)"
echo "CONDA_DIR=$CONDA_DIR | ENV_FILE=$ENV_FILE | REQ_FILE=$REQ_FILE"

# update apt-get & certs
if command -v apt-get >/dev/null 2>&1; then
  echo "[Step 0] Installing prerequisites (curl, ca-certificates) ... "
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -yqq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq curl ca-certificates
fi

# --
# set-up project dependencies
# --

echo "[Step 1/7] Downloading Miniconda ... "

if [[ ! -d "$CONDA_DIR" ]]; then
  INSTALLER=/tmp/miniconda.sh
  wget -qO "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x "$INSTALLER"
  # sanity check
  head -n1 "$INSTALLER"
else
  echo "Miniconda already present at $CONDA_DIR — skipping download ... "
fi

echo "[Step 2/7] Installing Miniconda to $CONDA_DIR ... "

if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  echo "[Miniconda] fresh install to $CONDA_DIR ... "
  /bin/bash "$INSTALLER" -b -p "$CONDA_DIR"
else
  if [[ "$UPDATE_MINICONDA" == "true" ]]; then
    echo "[Miniconda] updating in place (-u) ... "
    rm -rf "$CONDA_DIR"/pkgs/*.conda "$CONDA_DIR"/pkgs/*.tar.bz2 || true
    conda clean --all -y || true 2>/dev/null
    /bin/bash "$INSTALLER" -b -u -p "$CONDA_DIR"
  else
    echo "[Miniconda] present — skipping update ... "
  fi
fi

# load conda for this shell & prepare for subsequent cmds
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

echo "[Step 3] Syncing conda environment from $ENV_FILE..."

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found in $(pwd)" >&2
  exit 1
fi

# extract env name
ENV_NAME="${ENV_NAME:-$(awk -F': *' '/^[[:space:]]*name:[[:space:]]*/ {print $2; exit}' "$ENV_FILE" || true)}"
if [[ -n "${ENV_NAME:-}" ]]; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  # if no name in environment.yml let conda pick/create
  conda env update -f "$ENV_FILE" --prune || true
  ENV_NAME="$(conda env list --json | python - <<'PY'
import json,sys,os
data=json.load(sys.stdin)
envs=[p for p in data.get('envs',[]) if os.path.basename(p)!='base']
print(os.path.basename(sorted(envs,key=lambda p: os.path.getmtime(p))[-1]) if envs else '')
PY
)"
fi

# success?
if [[ -z "${ENV_NAME:-}" ]]; then
  echo "ERROR: Could not determine conda environment name." >&2
  exit 1 # nope!
fi
echo "Using env: $ENV_NAME" # yep!

echo "[Step 4] Verifying environment ... "

conda run -n "$ENV_NAME" --no-capture-output python -V >/dev/null

echo "[Step 5] Installing pip requirements ... "

if [[ -f "$REQ_FILE" ]]; then
  conda run -n "$ENV_NAME" --no-capture-output python -m pip install --upgrade pip
  conda run -n "$ENV_NAME" --no-capture-output python -m pip install -r "$REQ_FILE"
else
  echo "No $REQ_FILE — skipping."
fi

echo "[Step 6] Installing OpenJDK 17 ... "

if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq openjdk-17-jdk
else
  echo "apt-get not found; install OpenJDK 17 manually." >&2
fi

echo "[Step 7] Installing Maven ... "

if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq maven
else
  echo "apt-get not found; install Maven manually." >&2
fi

echo "[Step 8] Install final python packages ... "

conda run -n "$ENV_NAME" --no-capture-output python -m pip install phonemizer
conda run -n "$ENV_NAME" --no-capture-output python -m pip install flash_attn
conda run -n "$ENV_NAME" --no-capture-output python -m pip install sentencepiece
conda run -n "$ENV_NAME" --no-capture-output python -m pip install pysbd
conda run -n "$ENV_NAME" --no-capture-output python -m pip install geoopt
conda run -n "$ENV_NAME" --no-capture-output python -m pip install deeponto

# now we should already have all the dependencies we need for packages:
conda run -n "$ENV_NAME" --no-capture-output python -m pip install scispacy --no-deps
conda run -n "$ENV_NAME" --no-capture-output python -m pip install hierarchy_transformers --no-deps

echo "[Step 9] Install ROBOT for CLI-based reasoning ... "

git clone git@github.com:ontodev/robot.git
cd robot
mvn clean package
cd ..

# -- IMPORTANT: write the $ENV_NAME to the .env file (enables the remaining build scripts to run)

echo "AUTO_ENV_NAME=$ENV_NAME" >> .env

# alternatively, overwrite if it already exists (for now, we'll leave as is)

# if grep -q '^ENV_NAME=' .env 2>/dev/null; then
#   sed -i "s/^ENV_NAME=.*/ENV_NAME=$ENV_NAME/" .env
# else
#   echo "ENV_NAME=$ENV_NAME" >> .env
# fi

# -- FINITO.

echo "DONE!"

echo ""

echo "For interactive use, run:"

echo ""

echo "  source \"$CONDA_DIR/etc/profile.d/conda.sh\" && conda activate \"$ENV_NAME\""

echo ""

echo "Versions:"
conda run -n "$ENV_NAME" --no-capture-output python -V
java -version
mvn -v

echo ""

echo "finished ... "