"""
Annotate benchmark-questions.json with NER entity mentions

run script with:

python load_questions_and_extract_entities.py \
    --input ./data/benchmark-questions.json \
    --output-dir ./data \
    --extract-biomed \
    --ner scispacy
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


try:
    import spacy
except ImportError as e:
    sys.exit("spaCy is required for entity extraction")

_SCISPACY_AVAILABLE = False

try:
    import scispacy
    from spacy.util import get_package_path
except ImportError:
    pass
else:
    _SCISPACY_AVAILABLE = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach entity lists to every question in a MIRAGE question file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to benchmark-questions.json",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where the benchmark-questions-with-entities.json / jsonl files will be written",
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--extract-head",
        action="store_true",
        help="Use lightweight rule-based HEAD entity extraction (default).",
    )

    mode.add_argument(
        "--extract-biomed",
        action="store_true",
        help="Run biomedical NER with SciSpaCy",
    )

    parser.add_argument(
        "--ner",
        default="scispacy",
        choices={"scispacy", "biobert", "quickumls", "bern2"},
        help="Which NER backend to use when --extract-biomed is set (default: scispacy)",
    )

    return parser.parse_args()


def load_questions(in_path: Path):
    """
    :in_path (Path) to benchmark-questions.json
    :returns dict with 'questions' key -> ...
    """
    with in_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, list):
        return {"questions": payload, "_passthrough": {}}

    if "questions" not in payload:
        raise ValueError("Input JSON object is missing the mandatory 'questions' key.")

    passthrough = {k: v for k, v in payload.items() if k != "questions"}
    return {"questions": payload["questions"], "_passthrough": passthrough}


# for extracting HEAD entities
_HEAD_NLP = spacy.load("en_core_web_sm")


def extract_head_entities(question: str):
    """
    :question question string to process for extraction of HEAD entities
    """
    doc = _HEAD_NLP(question)
    results = []
    for sent in doc.sents:
        span = None
        subjects = [tok for tok in sent if tok.dep_ in {"nsubj", "nsubjpass"}]
        if subjects:
            subj = subjects[0]
            span = doc[subj.left_edge.i : subj.right_edge.i + 1]
        else:
            for chunk in sent.noun_chunks:
                span = chunk
                break
        if span:
            results.append(
                {
                    "entity_literal": span.text,
                    "start_position": span.start_char,
                    "end_position": span.end_char,
                    "entity_type": "HEAD",
                }
            )
    return results


def load_scispacy_model(model_name: str = "en_ner_bionlp13cg_md"):
    """
    SciSpaCy models are ordinary spaCy packages; tries to download & install model if not found
    :model_name name of the scispacy model to load/download+load
    """
    try:
        return spacy.load(model_name, disable=["parser", "textcat"])
    except OSError:
        print(f"[SciSpaCy] Downloading model {model_name}", file=sys.stderr)
        from spacy.cli import download # type: ignore
        download(model_name)
        return spacy.load(model_name, disable=["parser", "textcat"])


def extract_scispacy_entities(question: str, nlp) -> List[Dict[str, Any]]:
    """
    TODO: doc comment
    """
    doc = nlp(question)
    return [
        {
            "entity_literal": ent.text,
            "start_position": ent.start_char,
            "end_position": ent.end_char,
            "entity_type": ent.label_,
        }
        for ent in doc.ents
    ]


def annotate_questions(qs: Sequence[Dict[str, Any]], *, use_biomed: bool, ner_backend: str) -> None:
    """
    TODO: doc comment
    """
    if use_biomed:
        if ner_backend != "scispacy":
            warnings.warn(
                f"[WARN] NER backend '{ner_backend}' not yet implemented. "
                "Falling back to --extract-head behaviour.",
                RuntimeWarning,
            )
            use_biomed = False

    if use_biomed:
        if not _SCISPACY_AVAILABLE:
            sys.exit("SciSpaCy requested but the library is not installed")
        model_name = "en_ner_bionlp13cg_md"
        biomed_nlp = load_scispacy_model(model_name)
    else:
        biomed_nlp = None

    for q in qs:
        question_text: str = q.get("question", "")
        if use_biomed:
            annotations = extract_scispacy_entities(question_text, biomed_nlp)
        else:
            annotations = extract_head_entities(question_text)
        q["entities"] = annotations



def main() -> None:
    args = parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    data_blob = load_questions(in_path)
    questions = data_blob["questions"]

    use_biomed = bool(args.extract_biomed)
    annotate_questions(
        questions,
        use_biomed=use_biomed,
        ner_backend=args.ner.lower()
    )

    base_name = in_path.stem

    out_json = out_dir / f"{base_name}-entities.json"
    out_jsonl = out_dir / f"{base_name}-entities.jsonl"

    payload = questions

    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    with out_jsonl.open("w", encoding="utf-8") as fp:
        for q in questions:
            fp.write(
                json.dumps(
                    {"question": q["question"], "entities": [e["entity_literal"] for e in q["entities"]]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(questions):,} annotated questions to {out_json}")
    print(f"JSONL file saved to {out_jsonl}")


if __name__ == "__main__":
    main()
