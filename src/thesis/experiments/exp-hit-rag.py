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
import torch

##########
# GLOBALS
##########

# ------------------------------------
# LLM options:
# ------------------------------------
# "mistralai/Mistral-7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.3"
# "BioMistral/BioMistral-7B"
# ------------------------------------

LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 42
USE_GPU_RETRIEVER = True


##################################################
# BOOTSTRAP: encoder, retriever & entity selector
##################################################

common_map = Path("./embeddings/entity_mappings.json") # *entity mappings
common_verbalisations = Path("./embeddings/verbalisations.json") # rdfs:label(s) & verbs
embeddings_dir = "./embeddings" # dir for embeddings

if not USE_GPU_RETRIEVER:

    # fine-tuned embedding model, for embedding entity mentions
    retriever_model_fp = hit_SNOMED25_model_path = Path('./models/snomed_models/HiT-mixed-SNOMED-25/final')

    # accepts an entity mention &
    # 1. produces an embedding
    # 2. measures the `score_fn` agaisnt existing embeddings
    # 3. returns a ranked list of entities as tuples: (rank, iri, score, verbalisation)
    hit_retriever = HiTRetriever(
        embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
        meta_map_fp=common_map,
        verbalisations_fp=common_verbalisations,
        model_fp=retriever_model_fp,
        score_fn=batch_poincare_dist_with_adaptive_curv_k
    )

else:

    retriever_model_fp = hit_SNOMED25_model_path = Path('./models/snomed_models/HiT-mixed-SNOMED-25/final')

    hit_retriever = GPUHiTRetriever(
        embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
        meta_map_fp=common_map,
        verbalisations_fp=common_verbalisations,
        resident=True,
        torch_device="cuda:0",
        torch_dtype=torch.float32
    )

    hit_retriever.register_local_model(retriever_model_fp)

# provides a ranking for a pool of entities (drawn from multiple multiple mentions for the same question)
entity_selector = ApproximateNearestNeighbourEntitySelector(hit_retriever)


##################
#  BOOTSTRAP: LLM
##################

# initialises a LLM & exposes methods for 
# RAG \w axiom verbalisation-based prompt enrichment
mistral_llm = MistralLLM(LLM_MODEL_ID)

mistral_llm.load_tokenizer(use_fast=True).load_model(
    device_map="auto",
    torch_dtype="auto",
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
  Path("./data/MIRAGE/benchmark.json"), 
  Path("./data/MIRAGE/benchmark-questions-entities"), 
  Path("./data/MIRAGE/benchmark-questions-entities")
).set_shuffle_question_options(True).set_permute_question_options(
  True
).set_retrieval_k(100).set_append_k(10).set_top_k(1).set_use_rag(True).register_retriever(
  hit_retriever # type: ignore
).register_entity_selector(
  entity_selector # type: ignore
).register_llm(
  mistral_llm # type: ignore
)

# 5 runs:

QATestHarness.set_random_seed(SEED)

tests.set_use_rag(True)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(True)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(True)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(True)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])

tests.set_use_rag(True)
tests.run_multiple(['pubmedqa', 'bioasq', 'mmlu', 'medqa', 'medmcqa'])