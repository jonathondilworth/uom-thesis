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
  echo "Setting project_root (.env) to: $project_root"
  echo "project_root=$project_root" >> .env
fi

# if the user has not provided any preference for wandb.ai (weights and biases) -- simply disable by default:
# set WANDB_MODE=['online', 'offline', 'disabled', 'shared']
if ! grep -q '^WANDB_MODE=' .env 2>/dev/null; then
  echo "Setting WANDB_MODE (.env) to: disabled"
  echo "WANDB_MODE=disabled" >> .env
fi

# we also need to pull down models for end-to-end reproducability:
# pretrained encoders (ANATOMY, GALEN, GO):
if ! grep -q '^ONT_PRETRAINED_MODEL_URL=' .env 2>/dev/null; then
  echo "Setting ONT_PRETRAINED_MODEL_URL (.env) to: https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view?usp=drive_link"
  echo "ONT_PRETRAINED_MODEL_URL=https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view?usp=drive_link" >> .env
fi

# # # IMPORTANT : TODO # # #

# SNOMED TUNED ENCODERS:
if ! grep -q '^SNOMED_MODELS_URL=' .env 2>/dev/null; then
  echo "Setting SNOMED_MODELS_URL (.env) to: TODO"
  echo "SNOMED_MODELS_URL=TODO" >> .env
fi

# # # IMPORTANT : TODO # # #

# since most people running this build script are unlikely to have a copy of SNOMED / a SNOMED license
# a copy is hosted (I presume, under a valid license) @ https://zenodo.org/doi/10.5281/zenodo.10511042
# downloadable (as a .zip) @ https://zenodo.org/records/14036213/files/ontologies.zip?download=1

if ! grep -q '^NHS_API_KEY=' .env 2>/dev/null; then
  echo "Setting ALT_SNOMED_ONTOLOGY_URL (.env) to: https://zenodo.org/records/14036213/files/ontologies.zip?download=1"
  echo "ALT_SNOMED_ONTOLOGY_URL=https://zenodo.org/records/14036213/files/ontologies.zip?download=1" >> .env
fi

echo "[INFO] Initialisation finished."
