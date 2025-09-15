import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


IRI_RE = re.compile(r"(\d+)$")


REQUIRED_KEYS = (
    "parent_entities",
    "ancestors",
    "anonymous_axioms",
    "equivalent_classes",
)


def extract_id_from_iri(iri: str):
    m = IRI_RE.search(iri)
    if not m:
        raise ValueError(f"Could not extract numeric id from IRI: {iri}")
    return m.group(1)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_item_with_graph(item: Dict[str, Any], graph_obj: Dict[str, Any]) -> None:
    for key in REQUIRED_KEYS:
        item[key] = graph_obj.get(key, [])


def main():

    parser = argparse.ArgumentParser(
        description="Merge required keys from ancestor parse into manually annotated data"
    )

    parser.add_argument(
        "--input-json",
        "-i",
        help="Path to manually-annotated.json (list of objects).",
    )

    parser.add_argument(
        "--merge-dir",
        "-m",
        help="Directory containing <SNOMED_ID>.json files.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directory to write evaluation_dataset.json.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    merge_dir = Path(args.merge_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "evaluation_dataset.json"

    items = load_json(input_path)
    if not isinstance(items, list):
        raise TypeError(f"Expecting a JSON list at {input_path}")

    merged = []
    missing = 0
    for idx, item in enumerate(items):
        try:
            iri = item["target_entity"]["iri"]
            snomed_id = extract_id_from_iri(iri)
        except Exception as e:
            raise RuntimeError(f"Item {idx} missing/invalid target_entity.iri") from e

        graph_file = merge_dir / f"{snomed_id}.json"
        if graph_file.is_file():
            graph_obj = load_json(graph_file)
            if not isinstance(graph_obj, dict):
                raise TypeError(f"Expecting a dict at {graph_file}")
            merge_item_with_graph(item, graph_obj)
        else:
            missing += 1
            for k in REQUIRED_KEYS:
                item.setdefault(k, [])

        merged.append(item)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(merged)} items to {out_path}")
    
    if missing:
        print(f"Note: {missing} item(s) had no matching graph file in {merge_dir}")


if __name__ == "__main__":
    main()