from abc import abstractmethod
from collections import OrderedDict, defaultdict
from pathlib import Path
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,           # type: ignore
    LogitsProcessorList         # type: ignore
)
from typing import override
from typing import Any, Callable
from tqdm import tqdm
from logits_processor_zoo.transformers import (
  CiteFromPromptLogitsProcessor,
  MultipleChoiceLogitsProcessor,
)
from thesis.utils.retrievers import BaseRetriever, QueryResult
from copy import copy, deepcopy
import torch
import random
import re

# Usage Links/Documentation:
# https://blogs.novita.ai/low-cpu-and-memory-usage-optimization-tips/


DEPENDENCIES_PATTERN = re.compile(
    r"(all of the above|none of the above|neither|both|option\s+[A-F]|answers?\s+[A-F])",
    re.IGNORECASE
)

def foldr_xs_to_csv(xs: list[str]) -> str:
  """recursive foldr (right fold) for creating a single csv row"""
  if len(xs) == 1:
    return xs[0]
  return str(f"{xs[0]},{foldr_xs_to_csv(xs[1:])}")


def format_options(options: dict[str, str], **kwargs) -> str:
  """produces a list of options for inclusion within prompts"""
  return "\n".join(f"{k}. {opt}" for k, opt in options.items())


def shuffle_dict(d: dict, *, seed: int | None = None) -> dict:
  """TODO: add doc-comment"""
  rng = random.Random(seed) if seed is not None else random
  keys = list(d.keys())
  rng.shuffle(keys)
  return {k: d[k] for k in keys}


def shuffle_options(options: dict[str, str], *, seed: int | None = None) -> dict[str, str]:
  """TODO: add doc-comment"""
  options_copy = copy(options)
  shuffled_options = shuffle_dict(options_copy, seed=seed)
  return shuffled_options


def format_and_shuffle_options(options: dict[str, str], *, seed: int | None = None) -> str:
  """TODO: add doc-comment"""
  randomised_options = shuffle_options(options=options, seed=seed)
  return format_options(randomised_options)


def opt_letters(options: dict[str, str]) -> str:
  """produces a comma seperated list of a dicts keys, e.g. A,B,C,D"""
  return foldr_xs_to_csv(list(options.keys()))


def prompt_template_no_rag(question: str, options: dict[str, str], *, shuffle: bool = False, seed: int | None = None, **kwargs) -> str:
  """produce a simple biomedical question answering (MC) template, discards additional kwargs (e.g. answer)"""
  options_formatting_fn = format_and_shuffle_options if (shuffle and not (any(DEPENDENCIES_PATTERN.search(txt or "") for txt in options.values()))) else format_options
  return (
    "You are a careful medical expert. Your output will be used only in a research context. "
    "Return only the letter of the best answer. \n\n"
    f"Here is the question: \n {question} \n\n"
    f"Here are the choices: \n{options_formatting_fn(options, seed=seed)} \n\n"
    "Answer (letter only): "
  )


def prompt_template_with_axioms(question: str, options: dict[str, str], axioms: list[str], *, shuffle: bool = False, seed: int | None = None, **kwargs) -> str:
  """produce a biomedical MCQA prompt for RAG, discards additional kwargs (e.g. answer)"""
  if not axioms:
    return prompt_template_no_rag(question=question, options=options, shuffle=shuffle, seed=seed, **kwargs)
  axiomatic_context = "\n".join(axiom for axiom in axioms)
  options_formatting_fn = format_and_shuffle_options if (shuffle and not (any(DEPENDENCIES_PATTERN.search(txt or "") for txt in options.values()))) else format_options
  return (
    "You are a careful medical expert. The context can help, use it only if it directly supports the answer. "
    "Return only the letter of the best answer. \n\n"
    f"Helpful context: \n {axiomatic_context} \n\n"
    f"Here is the question: \n {question} \n\n"
    f"Here are the choices: \n {options_formatting_fn(options, seed=seed)} \n\n"
    "Answer (letter only): "
  )


def chat_prompt_template_no_rag(question: str, options: dict[str, str], *, shuffle: bool = False, seed: int | None = None, **kwargs):
  """TODO"""
  user_content = prompt_template_no_rag(question, options, shuffle=shuffle, seed=seed)
  return [
    {"role": "user", "content": user_content}
  ]


def chat_prompt_template_with_axioms(question: str, options: dict[str, str], axioms: list[str], *, shuffle: bool = False, seed: int | None = None, **kwargs):
  """TODO"""
  user_content = prompt_template_with_axioms(
    question=question,
    options=options,
    axioms=axioms,
    shuffle=shuffle,
    seed=seed
  )
  return [
    {"role": "user", "content": user_content}
  ]


def permute_mcqa_options(options: dict[str, str], correct_letter: str, seed: int | None = None):
  """Returns (new_options, new_correct_letter). Skips shuffling if meta-options are detected."""
  # Do not shuffle if a question relies on the order in which the options are presented
  if any(DEPENDENCIES_PATTERN.search(txt or "") for txt in options.values()):
    return options, correct_letter
  letters = list(options.keys())
  items = list(options.items())
  # shuffle
  rng = random.Random(seed) if seed is not None else random
  rng.shuffle(items)
  # reassign
  new_options = {}
  for i in range(len(items)):
    new_options[letters[i]] = items[i][1]
  # remap the answer
  correct_text = options[correct_letter]
  new_correct = next(L for L, txt in new_options.items() if txt == correct_text)
  return new_options, new_correct


def is_chat_template(obj) -> bool:
  """TODO"""
  if (obj and isinstance(obj, list)) and (obj[0] and isinstance(obj[0], dict)):
    return ("role" in obj[0] and "content" in obj[0])
  # else:
  return False


def normalise_mention_string(s: str) -> str:
  """TODO"""
  return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


##########
# LLM Wrappers
##########

class MistralLLM:

  _hf_id: str
  _model: Any
  _tokenizer: Any
  _callable_prompt_templates: dict[str, Callable]
  _PERMISSIBLE_OPTIONS = re.compile(r"\b([A-Z])\b")

  def __init__(self, hf_identifier: str, **kwargs):
    self._hf_id = hf_identifier
    self._model = None
    self._tokenizer = None
    self._callable_prompt_templates = {}
    
  def load_model(self, **kwargs):
    self._model = AutoModelForCausalLM.from_pretrained(self._hf_id, **kwargs)
    self._model.eval()
    return self

  def load_tokenizer(self, **kwargs):
    self._tokenizer = AutoTokenizer.from_pretrained(self._hf_id, **kwargs)
    if self._tokenizer.pad_token_id is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
    if hasattr(self._tokenizer, "padding_side"):
        self._tokenizer.padding_side = "left"
    return self

  def register_generation_config(self, **kwargs):
    self._model.generation_config = GenerationConfig(**kwargs)
    return self

  def register_prompt_template_fn(self, callback_key: str, fn: Callable):
    self._callable_prompt_templates[callback_key] = fn
    return self

  @torch.inference_mode()
  def generate(self, prompt: str, **kwargs):
    inputs = self._tokenizer(
      prompt,
      return_tensors="pt", 
    ).to(self._model.device)
    # generate output
    out = self._model.generate(
      **inputs,
      **kwargs
    ) # decode & return
    return self._tokenizer.decode(out[0], skip_special_tokens=True)

  def generate_inject_template(self, template_key: str, template_args: dict, **kwargs):
    template_fn = self._callable_prompt_templates[template_key]
    prompt = template_fn(**template_args)
    return self.generate(prompt, **kwargs)
  
  def produce_template_prompt(self, template_key: str, template_args: dict, **kwargs):
    template_fn = self._callable_prompt_templates[template_key]
    prompt = template_fn(**template_args)
    return prompt
  
  def generate_constrain_logits(self, prompt: str, logits_processor_list: list | None = None, max_tokens: int = 1000, **kwargs):
    if logits_processor_list is None:
      logits_processor_list = []
    return self.generate(
      prompt,
      max_new_tokens = max_tokens,
      min_new_tokens = 1,
      logits_processor = LogitsProcessorList(logits_processor_list),
      **kwargs
    )
  
  # TODO: clean this up a little bit, fn should accept the delimiter
  def generate_inject_template_and_constrain_logits_for_mcqa(self, template_key: str, template_args: dict, **kwargs):
    mclp = MultipleChoiceLogitsProcessor(
      tokenizer=self._tokenizer,
      choices=list(template_args["options"].keys()),
      delimiter="."
    )
    template_fn = self._callable_prompt_templates[template_key]
    prompt = template_fn(**template_args)
    return self.generate_constrain_logits(prompt, [mclp], max_tokens=1, **kwargs)

  # TODO: fix this, simply grabbing the last character from the string *may* result in inaccuracies (works fine for now!)
  def generate_single_letter_for_mcqa(self, template_key: str, template_args: dict, **kwargs):
    response = self.generate_inject_template_and_constrain_logits_for_mcqa(template_key, template_args, **kwargs)
    return str(response)[len(str(response)) - 1:]

  # UPDATED: slightly more sensible interface than ^

  def input_from_prompt_or_chat(self, prompt_or_messages):
    if is_chat_template(prompt_or_messages) and hasattr(self._tokenizer, "apply_chat_template"):
      return self._tokenizer.apply_chat_template(
        prompt_or_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
      ).to(self._model.device)
    # else, (regular non-chat 'prompt_or_message'):
    print("Failed to apply chat template ...")
    return self._tokenizer(prompt_or_messages, return_tensors="pt").to(self._model.device)

  def render_chat_template_with_inputs(self, template_key: str, template_args: dict):
    template_fn = self._callable_prompt_templates[template_key]
    prompt_or_messages = template_fn(**template_args)
    return self.input_from_prompt_or_chat(prompt_or_messages)

  @torch.inference_mode()
  def generate_mcqa_letter(self, template_key: str, template_args: dict, *, max_new_tokens: int = 1, mclp_delimiter = ".", **kwargs) -> str:
    choices = list(template_args["options"].keys())
    mclp = MultipleChoiceLogitsProcessor(tokenizer=self._tokenizer, choices=choices, delimiter=mclp_delimiter)
    inputs = self.render_chat_template_with_inputs(template_key, template_args)
    out = self._model.generate(
      **inputs,
      do_sample=False,
      min_new_tokens=1,
      max_new_tokens=max_new_tokens,
      logits_processor=LogitsProcessorList([mclp]),
      return_dict_in_generate=True,
    )
    prompt_len = inputs["attention_mask"].sum(dim=1).tolist()[0]
    new_ids = out.sequences[0, prompt_len:]
    new_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
    matched_choice = self._PERMISSIBLE_OPTIONS.search(new_text)
    if not matched_choice or matched_choice.group(1) not in choices:
      raise ValueError(f"Could not parse MCQA letter from '{new_text}' with choices '{choices}'")
    return matched_choice.group(1)
  


##########
# EntitySelectors
##########

class BaseEntitySelector:

  _STOPWORDS = frozenset([
    "patient","pt","pts","person","people","individual","individuals",
    "male","female","man","woman","boy","girl","child","children","kid",
    "infant","newborn","neonate","adult","elderly","senior","parent","parents",
    "he","she","they","them","him","her","his","hers","their","theirs",
    "you","your","yours","we","our","ours","i","me","my","mine","one",
    "someone","anyone","everyone","nobody","somebody","this","that","these","those",
    "presented","presents","presenting","complains","complained","reports","reported",
    "history","h/o","c/o","since","for","with","had","found","noted","developed",
    "exhibits","demonstrates","shows","reveals","diagnosed","diagnosis","examination","exam",
    "on","during","before","after","prior",
    "which","what","when","where","why","how","whose","whom",
    "true","false","correct","incorrect","appropriate","best","most","least",
    "except","not","all","following","choose","select","mark","option","options",
    "both","none","either","neither","above","below",
    "feature","features","sign","signs","symptom","symptoms","finding","findings",
    "test","tests","result","results","value","values","level","levels","rate","ratio",
    "management","treatment","therapy","mechanism","complication","evaluation","investigation","investigations",
    "method","methods","technique","techniques","approach","approaches","procedure","procedures",
    "cause","causes","type","types","class","classes","category","categories","group","groups",
    "age","aged","old","year","years","yr","yrs","month","months","mo","mos",
    "week","weeks","day","days","hour","hours","hr","hrs","minute","minutes","min","mins",
    "always","never","usually","commonly","rarely","frequently","sometimes","generally","typically",
    "mainly","mostly","predominantly","severe","mild","moderate","acute","chronic","subacute","persistent","recurrent",
    "according","guidelines","classification","defined","definition","called","known","named","term","terminology",
    "left","right","bilateral","unilateral","anterior","posterior","medial","lateral","superior","inferior",
    "proximal","distal","upper","lower","central","peripheral",
    "hospital","clinic","ward","opd","er","icu","casualty",
    "region","area","part","portion","site","surface","margin","border","apex","base",
    "volume","pressure","temperature","saturation","score","grade","stage","index",
  ])
  _all_mention_results: list[QueryResult]
  _retriever: BaseRetriever

  def __init__(self, retriever: BaseRetriever):
    self._retriever = retriever

  @abstractmethod
  def encode_and_rank_candidates(self, entities, **kwargs): ...

  # we need this to chop the candidate list down
  # for instance, if we fetch 5 candidate mentions per entity mention
  # and there are 3 entity mentions for a given question, the candiate list: 5 x 3 = 15
  # but, we likely want to select the best 3 to 5 of those 15 (whilst ensuring no duplicates):
  def get_top_candidates(self, top_k=3):
    existing_iris_in_final_candidates = set()
    final_candidates = []
    for result in self._all_mention_results:
      iri = result[1]
      if iri not in existing_iris_in_final_candidates:
        existing_iris_in_final_candidates.add(iri)
        final_candidates.append(result)
    return final_candidates[:top_k]



class SubsumptionEntitySelector(BaseEntitySelector):
    """Responsible for retrieving entities using the associated retriever (HiTRetriever or OnTRetriever).
         Scores by entity/concept subsumption, scores <= 0; higher = better."""
    @override
    def encode_and_rank_candidates(self, entities, *, ret_k=1, append_k_per_entity=1, reverse_score_order=True, lambda_weight=0.4, **kwargs):
      if append_k_per_entity > ret_k:
        raise ValueError("You cannot append more items (per mention) to the result set than you fetch from the retriever.")
      self._all_mention_results = []
      for mention in entities:
        term = normalise_mention_string(mention.get('entity_literal', ''))
        if not term or term in self._STOPWORDS:
          continue
        # else:
        results = self._retriever.retrieve(
          mention['entity_literal'], 
          top_k=ret_k,
          reverse_candidate_scores=reverse_score_order,
          weight=lambda_weight,
          model=self._retriever._model # type: ignore
        )
        if not results:
          continue
        for retrieval_result_index in range(min(append_k_per_entity, len(results))):
          self._all_mention_results.append(results[retrieval_result_index])
      # sort retrieval results by score (higher = better .. but the results are ordered in desc, 'closer to zero is better'; score <= 0)
      score_value_loc_at_idx = 2 # hardcoded value ...
      self._all_mention_results.sort(key=lambda result: result[score_value_loc_at_idx], reverse=True)



class ApproximateNearestNeighbourEntitySelector(BaseEntitySelector):
    """Responsible for retrieving entities using the associated retriever (HiTRetriever or OnTRetriever). 
       Scores by distance, minimised distance (lower) = better."""
    @override
    def encode_and_rank_candidates(self, entities, *, ret_k=1, append_k_per_entity=1, reverse_score_order=False, **kwargs):
      if append_k_per_entity > ret_k:
        raise ValueError("You cannot append more items to the result set than you fetch from the retriever.")
      self._all_mention_results = []
      for mention in entities:
        term = normalise_mention_string(mention.get('entity_literal', ''))
        if not term or term in self._STOPWORDS:
          continue
        # else:
        results = self._retriever.retrieve(
          mention['entity_literal'], 
          top_k=ret_k,
          reverse_candidate_scores=reverse_score_order,
          model=self._retriever._model # type: ignore
        )
        if not results:
          continue
        for retrieval_result_index in range(min(append_k_per_entity, len(results))):
          self._all_mention_results.append(results[retrieval_result_index])
      # sort retrieval results by score (lower = better .. in this case; and dist >= 0)
      score_value_loc_at_idx = 2 # hardcoded value ...
      self._all_mention_results.sort(key=lambda result: result[score_value_loc_at_idx], reverse=reverse_score_order)



class SimilarityEntitySelector(BaseEntitySelector):
  """Responsible for retrieving entities using the associated retriever (SBERTRetriever, or any future classes that score by similarity). 
    Scores by distance, maximised similarity (higher) = better."""
  @override
  def encode_and_rank_candidates(self, entities, *, ret_k=1, append_k_per_entity=1, reverse_score_order=True, **kwargs):
    if append_k_per_entity > ret_k:
      raise ValueError("You cannot append more items to the result set than you fetch from the retriever.")
    self._all_mention_results = []
    for mention in entities:
      term = normalise_mention_string(mention.get('entity_literal', ''))
      if not term or term in self._STOPWORDS:
        continue
      # else:
      results = self._retriever.retrieve(
        mention['entity_literal'], 
        top_k=ret_k,
        reverse_candidate_scores=reverse_score_order
      )
      if not results:
        continue
      for retrieval_result_index in range(min(append_k_per_entity, len(results))):
        self._all_mention_results.append(results[retrieval_result_index])
    score_value_loc_at_idx = 2 # hardcoded value ...
    self._all_mention_results.sort(key=lambda result: result[score_value_loc_at_idx], reverse=reverse_score_order)

