import argparse
import json
import re

from pathlib import Path

# utils

_regex_parens = re.compile(r"\s*\([^)]*\)") # for parentheses removal (prevent leakage)

def strip_parens(s: str) -> str:
    return _regex_parens.sub("", s)

# get CLI args

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Preprocessing for SNOMED entity lexicon"
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="path to snomed entity lexicon (JSON file)"
    )

    parser.add_argument(
        "--output", "-o", required=True,
        help="path to output preprocessed lexicon (cannot overwrite file at input path)"
    )

    # norm(\mathcal{L})
    parser.add_argument(
        "--strip-parens", "-s", action="store_true",
        help="strips the branch names in parentheses from the output"
    )

    # norm(\mathcal{L})
    parser.add_argument(
        "--to-lower-case", "-l", action="store_true",
        help="convert all labels to lowercase"
    )

    return parser.parse_args()


def preprocess(data: dict, to_lower: bool, remove_parens: bool) -> dict:
    """
    convert every 'label' in the SNOMED entity lexicon to 'name' (optionally prepro)
    :data SNOMED entity lexicon (dict)
    :to_lower normalise to lower (bool)
    :remove_parens normalise to prevent leakage (bool)
    :return preprocessed entity lexicon dictionary (dict)
    """
    for entry in data.values():
        labels = entry.get("label")
        if isinstance(labels, list):
            # renames 'rdfs:label' to 'name' (per training script requirements)
            if remove_parens and to_lower:
                entry["name"] = strip_parens(
                    entry["label"][0].lower()
                )
            elif remove_parens and not to_lower:
                entry["name"] = strip_parens(
                    entry["label"][0]
                )
            else:
                entry["name"] = entry["label"][0]
            entry.pop("label")
        if isinstance(labels, string):
            # the instance in which label = "owl:Thing"
            entry["name"] = "owl:Thing"
    return data

# script: main

def main():
    args = parse_args()
    
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if str(input_path) == str(output_path):
        raise ValueError(f"input and output paths must be distinct: {input_path}")

    with open(input_path, "r") as json_file_contents:
        data = json.load(json_file_contents)

    cleaned = preprocess(data, args.to_lower_case, args.strip_parens)

    with open(output_path, "w") as output:
        json.dump(cleaned, output, indent=4, separators=(",", ": "), sort_keys=False)

    print(f"Successfully wrote cleaned JSON to {output_path}") # TODO: switch to logger


if __name__ == "__main__":
    main()
