import argparse
import json
import os
import sys

def load_set_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading {path}, see: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        print(f"Error, {path} must contain list of strings (JSON) .. ", file=sys.stderr)
        sys.exit(1)

    return {s.lower() for s in data}

def main():
    parser = argparse.ArgumentParser(
        description="Compute lexical overlap between input file and compare file, write the diff to disk"
    )
    parser.add_argument(
        "--input", "-i",
        dest="file_one",
        required=True,
        help="Path to the first JSON file (list of strings)"
    )
    parser.add_argument(
        "--compare", "-c",
        dest="file_two",
        required=True,
        help="Path to the second JSON file (list of strings)"
    )
    parser.add_argument(
        "--output-diff", "-od",
        dest="output_diff",
        help="Directory in which to write 'strings-diff.json' containing items in the input file not in the comparison file"
    )
    parser.add_argument(
        "--output-inter", "-oi",
        dest="output_inter",
        help="Directory in which to write 'strings-inter.json' containing items in both the input file and the comparison file"
    )
    args = parser.parse_args()

    set1 = load_set_from_json(args.file_one)
    set2 = load_set_from_json(args.file_two)

    if not set1:
        print("coverage: 0% (no strings in first file)")
        if args.output_diff:
            os.makedirs(args.output_diff, exist_ok=True)
            output_path = os.path.join(args.output_diff, "strings-diff.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            print(f"Wrote diff to {output_path}")
        return

    intersection = set1 & set2
    coverage_pct = (len(intersection) / len(set1)) * 100
    formatted = f"{coverage_pct:.2f}".rstrip('0').rstrip('.')
    print(f"coverage: {formatted}%")

    if args.output_diff:
        diff = sorted(set1 - set2)
        try:
            os.makedirs(args.output_diff, exist_ok=True)
            output_path = os.path.join(args.output_diff, "strings-diff.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(diff, f, indent=2)
            print(f"Wrote diff to {output_path}")
        except OSError as e:
            print(f"Error writing diff file, see: {e}", file=sys.stderr)
            sys.exit(1)

    if args.output_inter:
        inter = sorted(intersection)
        try:
            os.makedirs(args.output_inter, exist_ok=True)
            output_path = os.path.join(args.output_inter, "strings-inter.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(inter, f, indent=2)
            print(f"Wrote diff to {output_path}")
        except OSError as e:
            print(f"Error writing inter file, see: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()