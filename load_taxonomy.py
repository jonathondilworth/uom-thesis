#!/usr/bin/env python3
"""
load_taxonomy.py
----------------

TODO
"""

# imports

from __future__ import annotations
from typing import Literal, overload

from deeponto.onto import Ontology
from deeponto.onto.taxonomy import OntologyTaxonomy
from deeponto.utils.file_utils import create_path
from HierarchyTransformers.src.hierarchy_transformers.datasets.construct import HierarchyDatasetConstructor

from pathlib import Path

# utils

@overload
def relative_to_absolute(path: Path | str, return_str: Literal[True]) -> str: ...

@overload
def relative_to_absolute(path: Path | str, return_str: Literal[False]) -> Path: ...

def relative_to_absolute(path: str | Path, return_str: bool=True) -> str | Path:
    """
    I find it outrageous how much boilerplate is required for linter compliance
    (...for such a simple function)
    """
    abs_path: Path = Path(path).expanduser().resolve()
    return str(abs_path) if return_str else abs_path


# flags # TODO: accept via argparse \w defaults
CONSTRUCT_HIT_TRAINING_DATASET_FLAG = False
CONSTRUCT_AND_SAVE_ENTITY_LEXICON_FLAG = True
CONSTRUCT_AND_SAVE_TRIPPLE_STORE = False # TODO: implement

# data dirs # TODO: accept via argparse \w defaults
DATA_DIR = "./data"
#HIT_DATA_DIR = "./data/hit_dataset"
HIT_DATA_DIR = "./data/hit_dataset_minified"

# file names # TODO: accept via argparse \w defaults
#ONTOLOGY_FILE_NAME = "snomedct-international.owl"
#SRC_ENTITY_LEXICON_FILE_NAME = "entity_lexicon.json"
#TARGET_ENTITY_LEXICON_FILE_NAME = "snomed_entity_lexicon.json"

ONTOLOGY_FILE_NAME = "./data/snomed_inferred_direct_aug.ttl"
SRC_ENTITY_LEXICON_FILE_NAME = "entity_lexicon.json"
TARGET_ENTITY_LEXICON_FILE_NAME = "inferred_snomed_entity_lexicon.json"


# produce fully qualified (absolute) paths for data processing
ONTOLOGY_ABS_PATH = relative_to_absolute(
    f"{DATA_DIR}/{ONTOLOGY_FILE_NAME}", 
    return_str=True
)
SRC_ENTITY_LEXICON_ABS_PATH = relative_to_absolute(
    f"{DATA_DIR}/{SRC_ENTITY_LEXICON_FILE_NAME}",
    return_str=False # !important: see .save_entity_lexicon
)
TARGET_ENTITY_LEXICON_ABS_PATH = relative_to_absolute(
    f"{DATA_DIR}/{TARGET_ENTITY_LEXICON_FILE_NAME}",
    return_str=False # !important: see .save_entity_lexicon
)
HIT_DATASET_ABS_PATH = relative_to_absolute(
    f"{HIT_DATA_DIR}", 
    return_str=True
)
DATA_DIR_ABS_PATH = relative_to_absolute(
    f"{DATA_DIR}",
    return_str=True
)
# ^ I know, it's not pythonic. TODO: fix

# script (main) # TODO: implement script properly

snomed_ontology = Ontology(ONTOLOGY_ABS_PATH, reasoner_type="elk")
snomed_taxonomy = OntologyTaxonomy(snomed_ontology, reasoner_type="elk")
hit_data_constructor = HierarchyDatasetConstructor(snomed_taxonomy)


if CONSTRUCT_HIT_TRAINING_DATASET_FLAG:
    """
    TODO: add comment
    """
    #hit_data_constructor.construct("./data/hit_dataset")
    hit_data_constructor.construct("./data/hit_dataset_minified")

if CONSTRUCT_AND_SAVE_ENTITY_LEXICON_FLAG:
    """
    See: deeponto.utils.file_utils.create_path, @params: parents: bool, exist_ok: bool
    TODO: add comment
    """
    create_path(DATA_DIR)
    hit_data_constructor.save_entity_lexicon(DATA_DIR)
    assert SRC_ENTITY_LEXICON_ABS_PATH.exists()
    SRC_ENTITY_LEXICON_ABS_PATH.rename(TARGET_ENTITY_LEXICON_ABS_PATH)
    if (TARGET_ENTITY_LEXICON_ABS_PATH.exists()):
        print("Success: SNOMED Entity Lexicon created!") # TODO: replace with logger


# TODO: Implement
if CONSTRUCT_AND_SAVE_TRIPPLE_STORE:
    pass

# TODO: file: restructure & refactor
