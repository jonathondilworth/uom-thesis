import argparse
import json
from pathlib import Path

def main():

    parser = argparse.ArgumentParser(
        description="Collect unique 'entities' from MIRAGE's JSONL file and write them to MIRAGE-entity-list.json"
    )

    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input MIRAGE JSONL file"
    )

    parser.add_argument(
        "--output", 
        required=True, 
        help="Directory to write output JSON file to"
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "MIRAGE-entity-list.json"

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    entities = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ents = obj.get("entities", [])
        for e in ents:
            entities.add(str(e).strip())

    result = sorted(entities) # implicit conversion to []

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(result)} unique entities to {output_path}")

if __name__ == "__main__":
    main()