#!/usr/bin/env bash
set -Eeuo pipefail

# -------- Config --------
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
ENV_FILE="${ENV_FILE:-environment.yml}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
INSTALLER="${INSTALLER:-/tmp/miniconda.sh}"
UPDATE_MINICONDA="${UPDATE_MINICONDA:-false}"

# Sudo helper
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then SUDO="sudo"; else SUDO=""; fi

cleanup() { [[ -f "$INSTALLER" ]] && rm -f "$INSTALLER" || true; }
trap cleanup EXIT

echo "Ubuntu: $(lsb_release -sd 2>/dev/null || echo unknown)"
echo "CONDA_DIR=$CONDA_DIR | ENV_FILE=$ENV_FILE | REQ_FILE=$REQ_FILE"

# -------- Step 0: ensure base tools on Ubuntu 24.04 --------
if command -v apt-get >/dev/null 2>&1; then
  echo "[Step 0] Installing prerequisites (curl, ca-certificates)..."
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -yqq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq curl ca-certificates
fi

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
  INSTALLER=/tmp/miniconda.sh
  wget -qO "$INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x "$INSTALLER"
  echo "[Step 2/7] Installing Miniconda to $CONDA_DIR ... "
  head -n1 "$INSTALLER"
  chmod +x "$INSTALLER"
  /bin/bash "$INSTALLER" -b -p "${CONDA_DIR:-$HOME/miniconda}"
else
  echo "Miniconda already present at $CONDA_DIR — skipping install."
fi

# -------- Steps 1–2: Miniconda install/update --------

if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  echo "[Miniconda] fresh install to $CONDA_DIR"
  /bin/bash "$INSTALLER" -b -p "$CONDA_DIR"
else
  if [[ "$UPDATE_MINICONDA" == "true" ]]; then
    echo "[Miniconda] updating in place (-u) ..."
    # clear cache first to avoid InvalidArchiveError
    rm -rf "$CONDA_DIR"/pkgs/*.conda "$CONDA_DIR"/pkgs/*.tar.bz2 || true
    conda clean --all -y || true 2>/dev/null
    /bin/bash "$INSTALLER" -b -u -p "$CONDA_DIR"
  else
    echo "[Miniconda] present — skipping update."
  fi
fi

# Load conda for this shell (no need to run `conda init` in scripts)
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda config --set auto_activate_base false

# (Optional) Accept ToS if supported (older conda may not have this command)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# -------- Step 3: Create/Update env from environment.yml --------
echo "[Step 3] Syncing conda environment from $ENV_FILE..."
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found in $(pwd)" >&2
  exit 1
fi

# Extract env name (simple & robust)
ENV_NAME="${ENV_NAME:-$(awk -F': *' '/^[[:space:]]*name:[[:space:]]*/ {print $2; exit}' "$ENV_FILE" || true)}"
if [[ -n "${ENV_NAME:-}" ]]; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  # If no name in YAML, let conda pick/create, then infer
  conda env update -f "$ENV_FILE" --prune || true
  ENV_NAME="$(conda env list --json | python - <<'PY'
import json,sys,os
data=json.load(sys.stdin)
envs=[p for p in data.get('envs',[]) if os.path.basename(p)!='base']
print(os.path.basename(sorted(envs,key=lambda p: os.path.getmtime(p))[-1]) if envs else '')
PY
)"
fi
if [[ -z "${ENV_NAME:-}" ]]; then
  echo "ERROR: Could not determine conda environment name." >&2
  exit 1
fi
echo "Using env: $ENV_NAME"

# -------- Step 4: verify env works (non-interactive) --------
echo "[Step 4] Verifying environment..."
conda run -n "$ENV_NAME" --no-capture-output python -V >/dev/null

# -------- Step 5: pip requirements (optional) --------
echo "[Step 5] Installing pip requirements (if $REQ_FILE exists)..."
if [[ -f "$REQ_FILE" ]]; then
  conda run -n "$ENV_NAME" --no-capture-output python -m pip install --upgrade pip
  conda run -n "$ENV_NAME" --no-capture-output python -m pip install -r "$REQ_FILE"
else
  echo "No $REQ_FILE — skipping."
fi

# -------- Step 6: OpenJDK 17 --------
echo "[Step 6] Installing OpenJDK 17 (apt)..."
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq openjdk-17-jdk
else
  echo "apt-get not found; install OpenJDK 17 manually." >&2
fi

# -------- Step 7: Maven --------
echo "[Step 7] Installing Maven (apt)..."
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq maven
else
  echo "apt-get not found; install Maven manually." >&2
fi


# -------- Step 8: Final python packages --------
echo "[Step 8] Install final python packages"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install phonemizer
conda run -n "$ENV_NAME" --no-capture-output python -m pip install flash_attn
conda run -n "$ENV_NAME" --no-capture-output python -m pip install sentencepiece
conda run -n "$ENV_NAME" --no-capture-output python -m pip install pysbd
conda run -n "$ENV_NAME" --no-capture-output python -m pip install geoopt
conda run -n "$ENV_NAME" --no-capture-output python -m pip install deeponto
conda run -n "$ENV_NAME" --no-capture-output python -m pip install scispacy --no-deps
conda run -n "$ENV_NAME" --no-capture-output python -m pip install hierarchy_transformers --no-deps

echo "✅ Done."
echo "For interactive use, run:"
echo "  source \"$CONDA_DIR/etc/profile.d/conda.sh\" && conda activate \"$ENV_NAME\""
echo
echo "Versions:"
conda run -n "$ENV_NAME" --no-capture-output python -V
java -version
mvn -v
