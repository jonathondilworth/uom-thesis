from __future__ import annotations
from pathlib import Path
import argparse

from deeponto.onto import Ontology
from deeponto.onto.taxonomy import OntologyTaxonomy
from deeponto.utils.file_utils import create_path

from HierarchyTransformers.src.hierarchy_transformers.datasets.construct import HierarchyDatasetConstructor

# utils:

def get_full_path(rel_path: str) -> str:
    """Expand and resolve a relative path to an absolute path string"""
    return str(Path(rel_path).expanduser().resolve())

# script:

def main() -> None:
    
    parser = argparse.ArgumentParser(
        description="Construct HiT training dataset and/or save SNOMED entity lexicon"
    )
    
    # use: ./data/hit_dataset to compile training data to expected location
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to write HiT training data, default: None (doesn't compile training data)",
    )

    parser.add_argument(
        "--entity-lexicon",
        type=str,
        default="./data",
        help="Dir to write entity lexicon JSON file, default: ./data (entity_lexicon.json)",
    )

    parser.add_argument(
        "--ontology",
        type=str,
        default="./data/snomedct-international.owl",
        help="Path to the ontology OWL file, default: ./data/snomedct-international.owl",
    )

    parser.add_argument(
        "--infer",
        action="store_true",
        help="Use reasoning (elk) instead of structural taxonomy (bool)",
    )

    args = parser.parse_args()
    
    reasoner = "elk" if args.infer else "struct"

    snomed_ontology = Ontology(get_full_path(args.ontology), reasoner_type=reasoner)
    snomed_taxonomy = OntologyTaxonomy(snomed_ontology, reasoner_type=reasoner)

    hit_data_constructor = HierarchyDatasetConstructor(snomed_taxonomy)

    if args.training_data:
        hit_data_constructor.construct(get_full_path(args.training_data))

    if args.entity_lexicon:
        create_path(get_full_path(args.entity_lexicon))
        hit_data_constructor.save_entity_lexicon(get_full_path(args.entity_lexicon))


if __name__ == "__main__":
    main()
