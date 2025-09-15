import argparse
import json
import random
import sys
from pathlib import Path
from copy import deepcopy

TEMPLATE = {
    "internal_id": "",
    "source_dataset": "REQUIRES ANNOTATION",
    "question_id": "REQUIRES ANNOTATION",
    "question": "REQUIRES ANNOTATION",
    "entity_mention": {
        "entity_literal": "",
        "start_position": 0,
        "end_position": 0,
        "entity_type": "REQUIRES_ANNOTATION"
    },
    "target_entity": {
        "iri": "REQUIRES ANNOTATION",
        "rdfs:label": "REQUIRES ANNOTATION",
        "skos:prefLabel": "REQUIRES ANNOTATION",
        "skos:altLabels": [],
        "depth": 0
    }
}

def main():
    parser = argparse.ArgumentParser(
        description="Entity sampling procedure"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to JSON file containing a list of strings"
    )
    parser.add_argument(
        "--procedure", 
        required=True, 
        choices=["deterministic", "random"],
        help="Sampling procedure to use."
    )
    parser.add_argument(
        "-n", "--number", 
        type=int, 
        default=50,
        help="Number of entities to sample."
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    try:
        with input_path.open("r", encoding="utf-8") as f:
            entity_list = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load input JSON, {e}")
        sys.exit(1)

    if not isinstance(entity_list, list):
        print("[ERROR] Input JSON is not a list of strings")
        sys.exit(1)

    if args.procedure == "deterministic":
        manual_path = data_dir / "manually-annotated.json"
        if manual_path.exists():
            print("[SAMPLING PROCEDURE] Manually annotated data is present at './data/manually-annotated.json'")
        else:
            print("[ERROR] The sampling procedure is set to deterministic, but no file exists at './data/manually-annotated.json'!")
        sys.exit(0)

    # else, procedure is set to random:
    
    n = args.number
    candidates = [s for s in entity_list if isinstance(s, str) and len(s) > 5]

    if len(candidates) < n:
        print(f"[ERROR] Not enough candidates (needed {n}, got {len(candidates)})")
        sys.exit(1)

    sampled = random.sample(candidates, n)

    out_list = []
    for idx, ent in enumerate(sampled):
        obj = deepcopy(TEMPLATE)
        obj["internal_id"] = f"mc-{idx:04d}" # up to 1000 smaples
        obj["entity_mention"]["entity_literal"] = ent
        out_list.append(obj)

    out_path = data_dir / "ready-for-annotation.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("...finished.")


if __name__ == "__main__":
    main()