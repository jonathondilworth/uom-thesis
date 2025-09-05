from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Any, Dict, List


def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract questions from a MIRAGE benchmark file."
    )
    
    parser.add_argument(
        "--dataset",
        "--benchmark",
        "--input",
        "-i",
        dest="input_path",
        required=True,
        help="Path to (MIRAGE) benchmark.json",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        required=True,
        help="Directory in which the questions.json will be written",
    )

    return parser.parse_args()


def extract_questions(raw: Dict[str, Any]):
    """
    :raw parsed JSON object containing benchmark.sjon 
    """
    questions: List[Dict[str, Any]] = []

    for dataset_name, dataset_blob in raw.items():
        if not isinstance(dataset_blob, dict):
            continue

        for qid, qrec in dataset_blob.items():
            item: Dict[str, Any] = {
                "source_dataset": dataset_name,
                "question_id": str(qid),
                "question": qrec.get("question", "").strip(),
            }
            metadata = {
                k: v
                for k, v in qrec.items()
                if k not in {"question", "options", "answer"}
            }
            if metadata:
                item["metadata"] = metadata

            questions.append(item)

    return questions


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with in_path.open("r", encoding="utf-8") as fp:
            raw_json = json.load(fp)
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] Cannot open input file: {exc}")

    questions = extract_questions(raw_json)
    out_path = out_dir / "benchmark-questions.json"
    payload = questions

    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"Wrote {len(questions):,} questions to {out_path}")


if __name__ == "__main__":
    main()