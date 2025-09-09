from __future__ import annotations

from pathlib import Path
from pydantic import validate_call
from argparse import ArgumentParser, Namespace

from datetime import datetime
from typing import Union, Callable

import numpy as np
import json
import pickle
import logging

from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from thesis.utils.data_utils import (
    load_json,
    load_concepts_to_list,
    naive_tokenise,
    parallel_tokenise
)

from thesis.utils.llm_utils import (
    ApproximateNearestNeighbourEntitySelector,
    MistralLLM, 
    chat_prompt_template_no_rag, 
    chat_prompt_template_with_axioms
)

from thesis.utils.retrievers import (
    TFIDFRetriever,
    BM25Retriever,
    SBERTRetriever,
    HiTRetriever,
    OnTRetriever
)

from thesis.utils.math_functools import (
  batch_cosine_similarity,
  batch_poincare_dist_with_adaptive_curv_k,
  entity_subsumption,
  concept_subsumption
)

import scispacy
import spacy

# bootstrap: search (simple retrieval)

common_map = Path("./embeddings/entity_mappings.json") # entity mappings
common_verbalisations = Path("./embeddings/verbalisations.json") # rdfs:label(s) & verbs
embeddings_dir = "./embeddings" # dir for embeddings

hit_model_path = Path('./models/snomed_models/HiT-mixed-SNOMED-25/final')
ont_model_path = Path("./models/snomed_models/OnTr-snomed25-uni")

tfidf_ret = TFIDFRetriever(
    verbalisations_fp=common_verbalisations,
    meta_map_fp=common_map
)

bm25_ret = BM25Retriever(
    verbalisations_fp=common_verbalisations,
    meta_map_fp=common_map
)

sbert_ret = SBERTRetriever(
  embeddings_fp=Path(f"./embeddings/sbert-plm-embeddings.npy"),
  meta_map_fp=Path("./embeddings/entity_mappings.json"),
  verbalisations_fp=Path("./embeddings/verbalisations.json"),
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_cosine_similarity
)

hit_retriever_w_dist = HiTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=hit_model_path,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

hit_retriever_entity_sub = HiTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=hit_model_path,
    score_fn=entity_subsumption
)

ont_retriever_w_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_model_path,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_retriever_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_model_path,
    score_fn=concept_subsumption
)

##################
#  BOOTSTRAP: LLM
##################

# define a prompt template, specifically for this use case:

GLOBAL_VERBALISATIONS: dict = load_json(Path("./data/snomed_axioms.json"))

def fetch_verbalisation(concept_iri: str) -> str:
    snomed_concept = GLOBAL_VERBALISATIONS[concept_iri]
    if not snomed_concept:
      return ""
    label = snomed_concept['label'] # rdfs_label
    verbalisations = snomed_concept.get("verbalization") or snomed_concept.get("verbalisations") or {}
    subclass_axioms = (verbalisations.get("subclass_of") or [])[:1]
    equiv_axioms = (verbalisations.get("equivalent_to") or [])[:1]
    if len(subclass_axioms) > 0:
        return subclass_axioms[0]
    if len(equiv_axioms) > 0:
        return equiv_axioms[0]
    # else:
    print("[WARNING] Cannot find subclass axioms or equiv axioms! Returning empty string!")
    return ""

def custom_prompt_with_axioms(question: str, axioms: list[str], **kwargs) -> str:
  """produce a biomedical MCQA prompt for RAG, discards additional kwargs (e.g. answer)"""
  if not axioms:
    axioms = []
  axiomatic_context = "\n".join(axiom for axiom in axioms)
  return (
    "You are a careful medical expert. The context can help, use it only if it directly supports the answer. "
    f"Helpful context: \n {axiomatic_context} \n\n"
    f"Here is the question: \n {question} \n\n"
    "Answer: "
  )

LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# provides a ranking for a pool of entities (drawn from multiple multiple mentions for the same question)
entity_selector_hit = ApproximateNearestNeighbourEntitySelector(hit_retriever_w_dist)
entity_selector_ont = ApproximateNearestNeighbourEntitySelector(ont_retriever_w_dist)

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
mistral_llm.register_prompt_template_fn("axiom_rag_chat", custom_prompt_with_axioms)

# BioMedical NER

ner_categories = [
    "CARDINAL",  
    "EVENT", 
    "FAC", 
    "GPE", 
    "LOC", 
    "NORP", 
    "ORDINAL", 
    "ORG", 
    "PERSON", 
    "PRODUCT", 
    "QUANTITY", 
    "LOC", 
    "MISC", 
    "ORG", 
    "PER"
]

import sys
from typing import Dict, List, Any

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

nlp = load_scispacy_model()

if not nlp:
    nlp = spacy.load("en_core_sci_sm")

def extract_med_entities(q: str):
    return [
      {
        "entity_literal": x.text
      } for x in (nlp(q)).ents
    ]

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

# BOOTSTRAP: LLM (END)

# # ideally, we would load from config (TODO: load cfgNode \w yacs or hydra)
# tests = QATestHarness(
#   Path("./data/benchmark.json"),
#   Path("./data/benchmark-questions-entities.json"),
#   Path("./data/benchmark-questions-entities.json")
# ).set_shuffle_question_options(True).set_permute_question_options(
#   True
# ).set_retrieval_k(100).set_append_k(10).set_top_k(1).set_use_rag(True).register_retriever(
#   ont_ret_snomed_25_w_dist # type: ignore
# ).register_entity_selector(
#   entity_selector # type: ignore
# ).register_llm(
#   mistral_llm # type: ignore
# )

# app

app = FastAPI(title="Serve SNOMED Verbalisation Indicies")

# modify to combat specific CORS issues
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://192.168.0.146",
    "http://192.168.0.146:5173",
    "http://192.168.0.146:8000",
    "http://192.168.0.112:5173",
    "http://192.168.0.112:8000",
    "http://192.168.0.112",
    "http://192.168.0.92:5173",
    "http://192.168.0.92:8000",
    "http://192.168.0.92"
]

# this is simply a test demo, so allow_origins=["*"]
# DO NOT DEPLOY THIS IN PROD WITH SETTINGS SUCH AS THESE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models for request and response

# request
class RetrievalRequest(BaseModel):
  query: str
  retrieval_method: str
  score_function: str
  top_k: int = 10
  weight: float = 0.35

# START CHANGES (SEPT 2025): 
# Here only for the purposes of compatability with the API
# that was written months prior

from thesis.utils.query_utils import QueryResult

class SingleRetrievalItem(BaseModel):
    rank: int
    id: str # iri
    score: float
    text: str # verbalisation

def my_map(fn: Callable, xs: list):
    return [fn(x) for x in xs]

# map :: (a -> b) -> [a] -> [b]
def map_query_result_to_retrieval_items(results: list[QueryResult]) -> list[SingleRetrievalItem]:
    return [SingleRetrievalItem(
        rank=result[0],
        id=result[1],
        score=result[2],
        text=result[3]
    ) for result in results]

# END CHANGES

# response:
class RetrievalResponse(BaseModel):
  results: list[QueryResult]

# Example response:
# response: list of results:
# [
#   {
#       "rank": 1,
#       "id": 4242,
#       "score": 0.1724,
#       "text": "Pain in the chest"
#   },
#   {
#       "rank": 2,
#       "id": 1337,
#       "score": 0.1892,
#       "text": "Broken collar bone"
#   },
#   ...
# ]

# IR @ <address>/search
@app.post("/search", response_model=RetrievalResponse)

def search(req: RetrievalRequest):

    if not req.query:
        raise HTTPException(status_code=400, detail="`query` cannot be empty.")
    
    if not req.retrieval_method:
        raise HTTPException(status_code=400, detail="`retrieval_method` cannot be empty.")
    
    if not req.score_function:
        raise HTTPException(status_code=400, detail="`score_function` cannot be empty.")
    
    if not req.top_k:
        raise HTTPException(status_code=400, detail="`top_k` cannot be empty.")

    centri_weight = 0.35 # default
    if req.weight:
        centri_weight = req.weight

    match req.retrieval_method:
        case "tf-idf":
            results = tfidf_ret.retrieve(req.query, top_k=req.top_k)
            response_items = map_query_result_to_retrieval_items(results)
            json_encoded_results = jsonable_encoder(response_items)
            return JSONResponse(content=json_encoded_results, status_code=200)
        case "bm-25":
            results = bm25_ret.retrieve(req.query, top_k=req.top_k)
            response_items = map_query_result_to_retrieval_items(results)
            json_encoded_results = jsonable_encoder(response_items)
            return JSONResponse(content=json_encoded_results, status_code=200)
        case "sbert":
            results = sbert_ret.retrieve(req.query, top_k=req.top_k, reverse_candidate_scores=True)
            response_items = map_query_result_to_retrieval_items(results)
            json_encoded_results = jsonable_encoder(response_items)
            return JSONResponse(content=json_encoded_results, status_code=200)
        case "hit":
            if req.score_function == "entity-subsumption":
                results = hit_retriever_entity_sub.retrieve(req.query, top_k=req.top_k, reverse_candidate_scores=True, model=hit_retriever_entity_sub._model, weight=centri_weight)
            else: # fallback to hyperbolic
                results = hit_retriever_w_dist.retrieve(req.query, top_k=req.top_k, reverse_candidate_scores=False, model=hit_retriever_w_dist._model)
            # build response
            response_items = map_query_result_to_retrieval_items(results)
            json_encoded_results = jsonable_encoder(response_items)
            return JSONResponse(content=json_encoded_results, status_code=200)
        case "ont":
            if req.score_function == "concept-subsumption":
                results = ont_retriever_w_con_sub.retrieve(req.query, top_k=req.top_k, reverse_candidate_scores=True, model=ont_retriever_w_con_sub._model, weight=centri_weight)
            else: # fallback to hyperbolic
                results = ont_retriever_w_dist.retrieve(req.query, top_k=req.top_k, reverse_candidate_scores=False, model=ont_retriever_w_dist._model)
            # build response
            response_items = map_query_result_to_retrieval_items(results)
            json_encoded_results = jsonable_encoder(response_items)
            return JSONResponse(content=json_encoded_results, status_code=200)
        case _:
            # fallthrough
            raise HTTPException(status_code=500, detail="unable to process request: hit fallthrough prior to any return statement.")


# OLD CODE FOR SEPERATED LLM MODULES

# import requests
# import sys

# def call_llm(server_url: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
#     payload = {
#         "prompt": prompt,
#         "max_new_tokens": max_new_tokens,
#         "temperature": temperature,
#         "top_p": top_p,
#         "top_k": top_k
#     }
#     try:
#         resp = requests.post(server_url, json=payload, timeout=60)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         sys.exit(f"Request failed: {e}")
#     data = resp.json()
#     return data.get("generated_text", "")

# END OF OLD CODE

# Request & Response Models for LLM interface:

class QuestionRequest(BaseModel):
  question: str
  retrieval_method: str
  score_function: str
  max_new_tokens: int = 128
  temperature: float = 0.3
  top_k: int = 40
  top_p: float = 0.85

class QuestionResponse(BaseModel):
  generated_text: str

# LLM Question @ <address>/question
@app.post("/question", response_model=QuestionResponse)
def question(req: QuestionRequest):

    question_entities = extract_med_entities(req.question)

    match req.retrieval_method:
        case "hit":
            entity_selector_hit.encode_and_rank_candidates(question_entities)
            results = entity_selector_hit.get_top_candidates()
            entity_results = map_query_result_to_retrieval_items(results)
        case "ont":
            entity_selector_ont.encode_and_rank_candidates(question_entities)
            results = entity_selector_ont.get_top_candidates()
            entity_results = map_query_result_to_retrieval_items(results)
        case _:
            raise HTTPException(status_code=500, detail="Invalid retrieval_method, hit fallthrough")    
    
    print("Entity Results: ")
    print(entity_results)

    axiom_verbalisations = [fetch_verbalisation(result.id) for result in entity_results]

    # temporary work around for obtaining the length .. hacky and inefficient!
    custom_prompt = mistral_llm.produce_template_prompt(
        "axiom_rag_chat", 
        {
            "question": req.question, 
            "axioms": axiom_verbalisations
        }
    )
    custom_prompt_length = len(custom_prompt)

    print("Custom Prompt: ")
    print(custom_prompt)

    llm_response = mistral_llm.generate_inject_template(
        # name of the template fn
        "axiom_rag_chat",
        # arguments to the template fn
        {
            "question": req.question, 
            "axioms": axiom_verbalisations
        },
        # arguments to LLM generate function (e.g. max_new_tokens, etc)
        max_new_tokens = req.max_new_tokens,
        do_sample = True,
        temperature = req.temperature,
        top_k = req.top_k,
        top_p = req.top_p
    ) # generate

    print("Full LLM Response: ")
    print(llm_response)
    print("\n#################\n")

    # remove the original question from the response

    processed_response = llm_response[custom_prompt_length:]

    print("Processed Response: ")
    print(processed_response)
    print("\n#################\n")

    # OLD CODE FOR SEPERATED LLM MODULES

    # response = call_llm(
    #     server_url="http://192.168.0.112:8000/generate",
    #     prompt=message,
    #     max_new_tokens=128,
    #     temperature=0.7,
    #     top_p=0.9,
    #     top_k=50
    # )

    # END OF OLD CODE

    json_encoded_response = jsonable_encoder(processed_response)
    return JSONResponse(content=json_encoded_response, status_code=200)

    # if we get here then something went wrong!
    raise HTTPException(status_code=500, detail="`something else went wrong! Oh noes!`")