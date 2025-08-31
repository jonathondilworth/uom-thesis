import torch
from pathlib import Path
from data_utils import (
  load_json,
  merge_entity_mentions,
)
from llm_utils import (
  MistralLLM,
  BaseEntitySelector,
  SimilarityEntitySelector,
  ApproximateNearestNeighbourEntitySelector,
  SubsumptionEntitySelector,
  permute_mcqa_options,
  chat_prompt_template_no_rag,
  chat_prompt_template_with_axioms
)
from retrievers import (
  BaseRetriever,
  BaseModelRetriever,
  SBERTRetriever,
  HiTRetriever,
  OnTRetriever
)
from math_functools import (
  batch_cosine_similarity,
  batch_poincare_dist_with_adaptive_curv_k,
  entity_subsumption,
  concept_subsumption
)
from copy import copy, deepcopy
from tqdm import tqdm
from datetime import datetime, timezone
import numpy as np
import random, re, csv, json, time, pytz


snomed_concept_information: dict = load_json(Path("./data/snomed_axioms.json"))

COMMON_STOPWORDS = {
  "the","a","an","and","or","of","to","in","on","for","with","without","from",
  "by","at","as","is","are","was","were","be","been","being","that","which",
  "this","these","those","it","its","their","there","then","than","such"
}

SNOMED_CONCEPTS_NEAR_TOP = {
  "clinical finding","finding","disease","disorder","procedure","event", "body structure", "organism",
  "substance", "situation with explicit context","qualifier value","morphologic abnormality"
}

def lexical_repr_without_stopwords(s: str) -> set[str]:
  """set difference between the input string and 'common stopwords'"""
  return set(re.findall(r"[a-z]+", s.lower())) - COMMON_STOPWORDS

def lexical_overlap_score(input: str, reference_set: set[str]) -> int:
  """set (lexical) intersection between input_string and the reference vocab (typically, the question)"""
  return len(lexical_repr_without_stopwords(input) & reference_set)

def is_broad_concept(rdfs_label: str) -> bool:
  """basic lexical operation to check whether an rdfs_label is a high level SNOMED CT concept"""
  return (rdfs_label.split(" (", 1)[0]).lower() in SNOMED_CONCEPTS_NEAR_TOP

def select_axioms_for_prompt(axioms: list[str], question_text: str, *, min_overlap: int = 1) -> list[str]:
  """Return axioms that exceed the lexical overlap of `min_overlap` with the `question_text`"""
  clean_question_text = lexical_repr_without_stopwords(question_text)
  retained_axioms = []
  for this_axiom in axioms:
    lex_score = lexical_overlap_score(this_axiom, clean_question_text)
    if lex_score >= min_overlap:
      retained_axioms.append(this_axiom)
  return retained_axioms

def get_axiom_verbalisations(iris: list[str], *, max_verbalisations: int = 3, question_text: str, min_overlap: int = 1) -> list[str]:
  returnable_verbalisations: list[str] = []
  for iri in iris: # skip any 'owl:Thing'(s)
    if not iri or iri == "owl:Thing": # Though these shouldn't be in IRI
      continue
    snomed_concept = snomed_concept_information[iri]
    if not snomed_concept:
      continue
    label = snomed_concept['label'] # rdfs_label
    if is_broad_concept(label):
      continue
    # this is pretty horrible, TODO: make this a little less, difficult to read ...
    verbalisations = snomed_concept.get("verbalization") or snomed_concept.get("verbalisations") or {}
    subclass_axioms = (verbalisations.get("subclass_of") or [])[:max_verbalisations]
    equiv_axioms = (verbalisations.get("equivalent_to") or [])[:max_verbalisations]
    returnable_verbalisations.extend([f"{label} {s}" for s in subclass_axioms])
    returnable_verbalisations.extend([f"{label} {e}" for e in equiv_axioms])
  return select_axioms_for_prompt(
    returnable_verbalisations, 
    question_text, 
    min_overlap=min_overlap
  )

'''
get_axiom_verbalisations: previously causing issues (overwriting the axioms array), patch provided above ^

@DeprecationWarning
def get_axiom_verbalisations(iris: list[str], *, max_verbalisations: int = 3, question_text: str, min_overlap: int = 1) -> list[str]:
  """fetch axiom verbalisations from a file that has been pre-generated (takes too long at run-time)"""
  returnable_verbalisations = []
  for iri in iris: # skip any 'owl:Thing'(s)
    if iri == "owl:Thing": # Though these shouldn't be in IRI
      continue
    snomed_concept = snomed_concept_information[iri]
    label = snomed_concept['label'] # rdfs_label
    if is_broad_concept(label):
      continue
    verbalisations = snomed_concept['verbalization']
    subclass_axioms = verbalisations['subclass_of'][:max_verbalisations]
    equiv_axioms = verbalisations['equivalent_to'][:max_verbalisations]
    # this is pretty horrible, TODO: make this a little less, difficult to read ...
    complete_subclass_verbalisations = [f"{label} {verb_str}" for verb_str in subclass_axioms] if subclass_axioms is not [] else []
    complete_equiv_verbalisations = [f"{label} {verb_str}" for verb_str in equiv_axioms] if equiv_axioms is not [] else []
    returnable_verbalisations = [*complete_subclass_verbalisations, *complete_equiv_verbalisations]
  return select_axioms_for_prompt(
    returnable_verbalisations,
    question_text,
    min_overlap=min_overlap
  )
'''

def question_seed(dataset: str, q: dict, idx: int) -> int:
    qid = q.get("id", idx)
    return (hash((dataset, str(qid))) & 0xFFFFFFFF)

def count_predictions(incorrect_questions):
  selected_options = set()
  predictions = []
  for response in incorrect_questions:
    selected_options.add(response['predicition'])
    predictions.append(response['predicition'])
  counts = {}
  for selected_option in selected_options:
    counts[selected_option] = 0
  for prediction in predictions:
    counts[prediction] += 1
  return counts

def count_actual_answers(incorrect_questions):
  answer_set = set()
  answers = []
  for response in incorrect_questions:
    answer_set.add(response['question']['answer'])
    answers.append(response['question']['answer'])
  counts = {}
  for possible_answer in answer_set:
    counts[possible_answer] = 0
  for answer in answers:
    counts[answer] += 1
  return counts


class QATestHarness:
    
  _use_rag: bool
  _shuffle_question_options: bool
  _permute_question_options: bool
  _retrieval_k: int
  _append_k: int
  _top_k: int
  _benchmark_data: dict
  _biomedical_entity_mentions: dict
  _head_entity_mentions: dict

  _allowable_datasets = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']

  _retriever: BaseModelRetriever
  _entity_selector: BaseEntitySelector

  _correct_questions: list[dict]
  _incorrect_questions: list[dict]

  _shuffle_seed: int

  _llm: MistralLLM

  def __init__(self, benchmark_data_fp: Path, biomedical_mentions_fp: Path, head_mentions_fp: Path, *, log_dir: Path | str = "./runs"):
    self._benchmark_data = load_json(benchmark_data_fp)
    self._biomedical_entity_mentions = load_json(biomedical_mentions_fp)
    self._head_entity_mentions = load_json(head_mentions_fp)
    self._correct_questions = []
    self._incorrect_questions = []
    self._log_dir = Path(log_dir)
    self._log_dir.mkdir(parents=True, exist_ok=True)
    self._pred_dir = self._log_dir / "predictions"
    self._pred_dir.mkdir(parents=True, exist_ok=True)
    self._metrics_csv = self._log_dir / "qa_runs.csv"
  
  @classmethod
  def set_random_seed(cls, seed_value: int = 42):
    """apply a random seed value to python/np/torch random utils"""
    random.seed(seed_value) # python
    np.random.seed(seed_value) # numpy
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed_value) # single GPU, fails silently
      torch.cuda.manual_seed_all(seed_value) # multi GPU, fails silently
  
  def set_use_rag(self, use_rag: bool):
    """set to true if tests should run with prompt enrichment (fetching axiom verbalisations)"""
    self._use_rag = use_rag
    return self
  
  def register_retriever(self, retriever: BaseModelRetriever):
    """register a retriever to inject axiom verbalisations into the prompt/context during question answering (required `if use_rag == true`)"""
    if not self._use_rag:
      raise ValueError("You can only register a retriever when `use_rag` is set to True.")
    # else:
    self._retriever = retriever
    return self

  def register_entity_selector(self, selector: BaseEntitySelector):
    """register an entity selector cls to employ a selection criteria during entity and axiom retrieval (required `if use_rag == true`)"""
    if not self._use_rag:
      raise ValueError("You can only register an entity selector when `use_rag` is set to True.")
    # else:
    self._entity_selector = selector
    return self
  
  def register_llm(self, llm: MistralLLM):
    """register a specific LLM for when this harness is run"""
    self._llm = llm
    return self
  
  def set_retrieval_k(self, k: int):
    """the k_threhold for the retriever (cut-off applied during retrieval of each mention)"""
    self._retrieval_k = k
    return self
  
  def set_append_k(self, k: int):
    """the number of entities to select during retrieval (for each entity mention)"""
    self._append_k = k
    return self
  
  def set_top_k(self, k: int):
    """the number of entities to select from the entire pool of retrieved entities (ranked by score)"""
    self._top_k = k
    return self
  
  def set_shuffle_question_options(self, shuffle: bool):
    """set to true if the order of the (option, answer) should be randomised during question presentation"""
    self._shuffle_question_options = shuffle
    return self

  def set_permute_question_options(self, permute: bool):
    """set to true if the arrangement of option -> answer mappings should undergo permutation prior to question presentation"""
    self._permute_question_options = permute
    return self
  
  def _ensure_metrics_header(self):
    """writes a csv header to the metrics log"""
    if not self._metrics_csv.exists():
      with self._metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
          "timestamp","run_id","dataset","model","use_rag","retriever","rank_metric","shuffle_options",
          "permute_options","retrieval_k","append_k","top_k", "correct","incorrect","total","accuracy"
        ])

  def _append_metrics_row(self, row: dict):
    """accepts a row of metrics and writes experimental data to the log file"""
    self._ensure_metrics_header()
    with self._metrics_csv.open("a", newline="") as f:
      w = csv.writer(f)
      w.writerow([
        row["timestamp"], row["run_id"], row["dataset"], row["model"], row["use_rag"],
        row["retriever"], row["rank_metric"],row["shuffle_options"], row["permute_options"],
        row.get("retrieval_k",""), row.get("append_k",""), row.get("top_k",""),
        row["correct"], row["incorrect"], row["total"], row["accuracy"]
      ])

  def _write_predictions_jsonl(self, dataset: str, run_id: str, records: list[dict]):
    """saves predictions to jsonl for inspection/debugging"""
    out = self._pred_dir / f"{dataset}_{run_id}.jsonl"
    with out.open("w") as f:
      for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
  
  def run(self, dataset_name: str):
    """run the test harness across a specific dataset from the provided benchmark"""
    if dataset_name not in self._allowable_datasets:
      raise ValueError(f"You're trying to run the test harness agaisnt: {dataset_name}, which is not registered as an allowable dataset.")
    
    run_id = f"{dataset_name}-{int(time.time())}"
    ts = datetime.now(tz=pytz.timezone('Europe/London')).isoformat(timespec="seconds") + "Z"
    
    print(f"Using RAG: {self._use_rag}")
    questions_with_entity_mentions = merge_entity_mentions(
      self._benchmark_data, self._biomedical_entity_mentions, self._head_entity_mentions, allowable_datasets=[dataset_name]
    )
    
    # ensure questions are cleared
    self._correct_questions = []
    self._incorrect_questions = []

    print(f"Processing {dataset_name} ...")
   
    for question_index, question in enumerate(tqdm(questions_with_entity_mentions)):

      # options for shuffling the arrangement of the answers
      question['shuffle'] = self._shuffle_question_options
      question['seed'] = question_seed(dataset_name, question, question_index)
      
      question['axioms'] = []

      if self._permute_question_options: # remaps the option -> answer mapping (implemented after observed LM bias towards selecting `A`)
        permutated_options, new_answer_key = permute_mcqa_options(question['options'], question['answer'])
        question['original_options'] = copy(question['options'])
        question['original_answer'] = copy(question['answer'])
        del question['options']
        del question['answer']
        question['options'] = permutated_options
        question['answer'] = new_answer_key

      if self._use_rag:
        self._entity_selector.encode_and_rank_candidates(question['entities'], ret_k=self._retrieval_k, append_k_per_entity=self._append_k) # type: ignore
        top_candidate = self._entity_selector._all_mention_results[0] if len(self._entity_selector._all_mention_results) > 0 else []
        if top_candidate:
          rank, iri, score, label = top_candidate
          # filters out some noise \w lexical gating (verb_str \cap question_str) >= min_overlap
          # required for adding large numbers of verbalisations into the prompt context
          axioms = get_axiom_verbalisations(
            [iri],
            max_verbalisations=5,
            question_text=question['question'],
            min_overlap=2
          )
          question['axioms'] = axioms if axioms else []
        # end: if
      # end: if

      response_letter = self._llm.generate_mcqa_letter(
        "mirage_mcqa_axiom_rag_chat" if self._use_rag else "mirage_mcqa_no_rag_chat",
        question,
        max_new_tokens=2
      )

      question['prediction'] = response_letter

      if response_letter == question['answer']:
        self._correct_questions.append(question)
      else:
        self._incorrect_questions.append(question)
    
    total = len(questions_with_entity_mentions)
    correct = len(self._correct_questions)
    incorrect = len(self._incorrect_questions)
    acc = round((correct / total) * 100, 2)

    print(f"Total correct: {correct}")
    print(f"Total incorrect: {incorrect}")
    print(f"Accuracy: {acc}% \n")
    
    metrics = {
      "timestamp": ts,
      "run_id": run_id,
      "dataset": dataset_name,
      "model": getattr(self._llm, "_hf_id", "No _hf_id (custom)"),
      "use_rag": bool(self._use_rag),
      "retriever": False if not bool(self._use_rag) else type(self._retriever),
      "rank_metric": False if not bool(self._use_rag) else self._retriever._score_fn.__qualname__,
      "shuffle_options": bool(self._shuffle_question_options),
      "permute_options": bool(self._permute_question_options),
      "retrieval_k": getattr(self, "_retrieval_k", None) if self._use_rag else None,
      "append_k": getattr(self, "_append_k", None) if self._use_rag else None,
      "top_k": getattr(self, "_top_k", None) if self._use_rag else None,
      "correct": correct,
      "incorrect": incorrect,
      "total": total,
      "accuracy": acc,
    }

    self._append_metrics_row(metrics)

    per_question = []
    for question in questions_with_entity_mentions:
      per_question.append({
        "dataset": dataset_name,
        "question": question.get("question"),
        "options": question.get("options"),
        "gold": question.get("answer"),
        "prediction": question.get("prediction"),
        "correct": question.get("prediction") == question.get("answer"),
        "axioms": question.get("axioms", [])
      })
    self._write_predictions_jsonl(dataset_name, run_id, per_question)

    return metrics

  def run_multiple(self, datasets: list[str]):
    """run tests for a subset of the datasets associated with the provided benchmark"""
    for dataset in datasets:
      self.run(dataset)
      