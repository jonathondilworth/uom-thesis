import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract IRIs from manually-annotated.json")
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to manually-annotated.json"
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    iris = [item["target_entity"]["iri"] for item in data]

    Path("./data").mkdir(parents=True, exist_ok=True)

    with open("./data/dataset_iri_list.json", "w", encoding="utf-8") as f:
        json.dump(iris, f, ensure_ascii=False, indent=2)

    print("IRIs written to ./data/dataset_iri_list.json")


if __name__ == "__main__":
    main()