#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
ENV_FILE="${ENV_FILE:-environment.yml}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

echo "CONDA_DIR: ${CONDA_DIR}"
echo "ENV_FILE: ${ENV_FILE}"
echo "REQ_FILE: ${REQ_FILE}"
echo "MINICONDA_URL: ${MINICONDA_URL}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[Step 1/7] Downloading Miniconda (if needed) ... "

if [[ ! -d "$CONDA_DIR" ]]; then
  tmp_installer="$(mktemp)"
  curl -fsSL "$MINICONDA_URL" -o "$tmp_installer"
  echo "[Step 2/7] Installing Miniconda to $CONDA_DIR ... "
  bash "$tmp_installer" -b -p "$CONDA_DIR"
  rm -f "$tmp_installer"
else
  echo "Miniconda already present at $CONDA_DIR — skipping install."
fi

# init conda

source "$CONDA_DIR/etc/profile.d/conda.sh"

conda config --set always_yes yes --set changeps1 no

echo "[Step 3/7] Creating conda environment from $ENV_FILE (if missing) ... "

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found in $(pwd)" >&2
  exit 1
fi

# get the ENV name from environment.yml

ENV_NAME="${ENV_NAME:-$(awk -F': *' '/^name:/ {print $2; exit}' "$ENV_FILE" || true)}"

if [[ -z "${ENV_NAME:-}" ]]; then
  echo "No 'name:' found in $ENV_FILE. conda will create with the name inside the file (if present)."
  conda env create -f "$ENV_FILE" || echo "Environment may already exist, continuing ... "
else
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists — skipping create."
  else
    conda env create -f "$ENV_FILE"
  fi
fi

# if no ENV_NAME then use most recent env

if [[ -z "${ENV_NAME:-}" ]]; then
  ENV_NAME="$(conda env list | awk 'NF && $1 !~ /#|base/ {print $1}' | tail -n1)"
fi
if [[ -z "${ENV_NAME:-}" ]]; then
  echo "ERROR: Could not determine conda environment name." >&2
  exit 1
fi

echo "[Step 4/7] Activating conda env '$ENV_NAME' ... "

conda activate "$ENV_NAME"

echo "[Step 5/7] Installing Python requirements from $REQ_FILE (if present) ... "
if [[ -f "$REQ_FILE" ]]; then
  python -m pip --version >/dev/null 2>&1 || conda install pip
  python -m pip install --upgrade pip
  python -m pip install -r "$REQ_FILE"
else
  echo "No $REQ_FILE found — skipping pip requirements."
fi

echo "[Step 6/7] Installing OpenJDK 17 (apt) ... "

$SUDO apt-get update -y
$SUDO apt-get install -y openjdk-17-jdk

echo "[Step 7/7] Installing Maven (apt) ... "

$SUDO apt-get install -y maven

echo "DONE! ... Active conda env: $ENV_NAME"

echo "Versioning:"

python -V
java -version
mvn -v