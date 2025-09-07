from thesis.utils.retrievers import (
  BaseRetriever,
  SBERTRetriever,
  HiTRetriever,
  OnTRetriever
)
from thesis.utils.gpu_retrievers import (
    GPUHiTRetriever,
    GPUOnTRetriever
)
from pathlib import Path

### GEN

import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    import flash_attn  # type: ignore
    ATTN_IMPL = "flash_attention_2"
except Exception:
    ATTN_IMPL = "sdpa"

# Use bf16 on CUDA, else float32
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

### END GEN

from thesis.utils.math_functools import (
  batch_cosine_similarity,
  batch_poincare_dist_with_adaptive_curv_k,
  entity_subsumption,
  concept_subsumption
)
from thesis.utils.llm_utils import (
  MistralLLM,
  BaseEntitySelector,
  SimilarityEntitySelector,
  ApproximateNearestNeighbourEntitySelector,
  SubsumptionEntitySelector,
  chat_prompt_template_no_rag,
  chat_prompt_template_with_axioms
)
from thesis.utils.harness_utils import (
  QATestHarness
)

# ------------------------------------
# LLM options:
# ------------------------------------
# "mistralai/Mistral-7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.3"
# "BioMistral/BioMistral-7B"
# ------------------------------------

LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 42

# instanciate a retriever
sbert_ret = SBERTRetriever(
  embeddings_fp=Path(f"./embeddings/sbert-plm-embeddings.npy"),
  meta_map_fp=Path("./embeddings/entity_mappings.json"),
  verbalisations_fp=Path("./embeddings/verbalisations.json"),
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_cosine_similarity
)

# and an entity selector
sbert_entity_selector = SimilarityEntitySelector(sbert_ret)

# and an LLM
mistral_llm = MistralLLM(LLM_MODEL_ID)

mistral_llm.load_tokenizer(use_fast=True).load_model(
    device_map="auto",
    # torch_dtype="auto",
    # low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
    attn_implementation=ATTN_IMPL,
    low_cpu_mem_usage=True
).register_generation_config(
    do_sample=False,
    num_beams=1,
    pad_token_id=mistral_llm._tokenizer.pad_token_id,
    eos_token_id=mistral_llm._tokenizer.eos_token_id
)
mistral_llm.register_prompt_template_fn("mirage_mcqa_no_rag_chat", chat_prompt_template_no_rag)
mistral_llm.register_prompt_template_fn("mirage_mcqa_axiom_rag_chat", chat_prompt_template_with_axioms)

# ideally, we would load from config (TODO: load cfgNode \w yacs or hydra)
tests = QATestHarness(
  Path("./data/benchmark.json"),
  Path("./data/benchmark-questions-entities"),
  Path("./data/benchmark-questions-entities")
).set_shuffle_question_options(True).set_permute_question_options(
  True
).set_retrieval_k(100).set_append_k(10).set_top_k(1).set_use_rag(True).register_retriever(
  sbert_ret
).register_entity_selector(
  sbert_entity_selector
).register_llm(
  mistral_llm
)

# 5 runs:

QATestHarness.set_random_seed(SEED)

tests.set_use_rag(False)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(False)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(False)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(False)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(False)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])