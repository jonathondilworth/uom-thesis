#!/usr/bin/env bash
set -Eeuo pipefail

echo "[INFO] Initialising project!"

echo "[WARNING] DO NOT call from anywhere besides the root dir; or else everything will break! You have been warned!"

# WARNING: do not call from anywhere besides the root dir
# (else everything will break! You have been warned!)
project_root="$PWD"

echo "[INFO] Exposing lib to pythonpath"

# exposes forked version(s) of libraries (e.g. hierarchy_transformers)
# forks are neccesary due to support for local training 
# (additional pre-processing steps required as HiT doesn't build datasets
#  for training locally into the same format the trainer expects; i.e. the
#  entity lexicon produced by hierarchy_transformers.datasets.construct
#  is not compatible with  hierarchy_transformers.datasets.load_zenodo_dataset
#  & also the `training_hit.py` script has to be modified to support this).
export PYTHONPATH="$project_root/lib${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] Checking .env exists (if not, writing it)"

# if .env doesn't already exist, create it
touch .env

echo "[INFO] Saving sampling procedural data inside .env (if not already set)"

# for later scripts, we require setting the SAMPLING_PROCEDURE
# default to deterministic, but if set to random, do not overwrite
if ! grep -q '^SAMPLING_PROCEDURE=' .env 2>/dev/null; then
  echo "SAMPLING_PROCEDURE=deterministic" >> .env
fi

# same for the number of samples (default=50)
if ! grep -q '^SAMPLE_N=' .env 2>/dev/null; then
  echo "SAMPLE_N=50" >> .env
fi

echo "[INFO] Saving project root inside .env (if not already set)"

# also, this amount of orchestration requires the root dir to be stored as an ENV variable
if ! grep -q '^project_root=' .env 2>/dev/null; then
  echo "project_root=$project_root" >> .env
fi

# if the user has not provided any preference for wandb.ai (weights and biases) -- simply disable by default:
# set WANDB_MODE=['online', 'offline', 'disabled', 'shared']
if ! grep -q '^WANDB_MODE=' .env 2>/dev/null; then
  echo "WANDB_MODE=disabled" >> .env
fi

echo "[INFO] Initialisation finished."