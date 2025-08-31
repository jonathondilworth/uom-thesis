from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from pydantic import validate_call
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import coo_matrix, csr_matrix
from typing import Union, Callable, Union, Any, override, overload

from data_utils import parallel_tokenise
from math_functools import batch_poincare_dist_with_adaptive_curv_k, batch_euclidian_l2_distance
from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer
from query_utils import QueryResult

import numpy as np
import json
import pickle


def aggregate_posting_scores(query_weights, inverted):
    scores = {}
    for term, weight in query_weights.items():
        if term not in inverted:
            continue
        for doc_id, tfidf_score in inverted[term]:
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * tfidf_score
    return scores


def topk(scores: dict[int, float], k: int = 10):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def build_tf_idf_index(axiom_list: list[str], tfidf_dest: str, *args, **kwargs):
    vectoriser = TfidfVectorizer(**kwargs)
    doc_term_matrix = vectoriser.fit_transform(axiom_list)
    vocab = vectoriser.get_feature_names_out()
    # prep for storing to disk: create empty postings struct
    inverted_index: dict[str, list[tuple[int, float]]] = {term: [] for term in vocab}
    # see: https://matteding.github.io/2019/04/25/sparse-matrices/
    coo = coo_matrix(doc_term_matrix)
    # populate the inverted index
    for row, col, score in zip(coo.row, coo.col, coo.data):
        inverted_index[str(vocab[col])].append((int(row), float(score)))
    # order: desc
    for postings in inverted_index.values():
        postings.sort(key=lambda x: x[1], reverse=True)
    # save to disk
    with open(tfidf_dest, "wb") as fp:
        pickle.dump(
        {
            "vectorizer": vectoriser,
            "postings": postings, # type: ignore
            "verbalisations": axiom_list
        },
        fp,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return vectoriser, inverted_index


def build_bm_25_index(concept_list: list[str], bm_25_dest: str = "bm25-index.pkl", **kwargs):
    tokenised_concepts = parallel_tokenise(concept_list, **kwargs)
    bm25 = BM25Okapi(tokenised_concepts)
    with open(bm_25_dest, "wb") as fp:
        pickle.dump({
            "bm25": bm25,
            "verbalisations": concept_list,
        }, fp, protocol=pickle.HIGHEST_PROTOCOL
    )


class BaseRetriever(ABC):
    
  _verbalisations: list
  _meta_map: list

  @validate_call
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path):
    with open(verbalisations_fp, 'r', encoding='utf-8') as fp:
      self._verbalisations = json.load(fp)
    with open(meta_map_fp, 'r', encoding='utf-8') as fp:
       self._meta_map = json.load(fp)

  @abstractmethod
  def retrieve(self, query_string: str, *, top_k: int = 10, **kwargs) -> list[QueryResult]:
    pass



class BaseModelRetriever(BaseRetriever):
   
  _embeddings: np.ndarray
  _candidate_indicies: np.ndarray
  _model: Union[SentenceTransformer, HierarchyTransformer, OntologyTransformer]
  _score_fn: Callable

  @overload
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path): ...

  @overload
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path, *, score_fn: Callable | None = None): ...
    
  @overload
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path, *, score_fn: Callable | None = None, model_fp: Path | None = None): ...
    
  @overload
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path, *, score_fn: Callable | None = None, model_fp: Path | None = None, model_str: str | None = None): ...

  @validate_call
  def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path, *, score_fn: Callable | None = None, model_fp: Path | None = None, model_str: str | None = None):
    super().__init__(verbalisations_fp, meta_map_fp)
    self._embeddings = np.load(embeddings_fp, mmap_mode="r")
    self._candidate_indicies = np.arange(len(self._embeddings))
    if score_fn:
      self.register_score_function(score_fn)
    if model_fp:
      try:
        self.register_local_model(model_fp.expanduser().resolve())
      except FileNotFoundError:
        self.register_model(str(model_fp))
    elif (not model_fp and model_str):
      self.register_model(model_str)

  def register_score_function(self, score_fn: Callable):
    self._score_fn = score_fn

  def get_score_function_str_or_bool(self):
    if not self._score_fn:
        return False
    # else:
    return self._score_fn.__name__

  @override
  def retrieve(self, query_string: str, *, top_k: int | None = None, reverse_candidate_scores=False, **kwargs) -> list[QueryResult]:
    """
    TODO: 1. add docstring explaining why **kwargs is accepted and pass through to _score_fn
          2. add explaination of parameters
          3. types (args/return)
    ----------------------------------------
    TODO: method is too verbose (multiple elif branches, etc), look @ providing a more concise implementation
    """
    query_embedding = self._embed(query_string)
    scored_embeddings = self._score_fn(query_embedding, self._embeddings, **kwargs)
    if reverse_candidate_scores and top_k is not None:
      top_k_indicies = self._candidate_indicies[np.flip(np.argsort(scored_embeddings))[:top_k]]
    elif not reverse_candidate_scores and top_k is not None:
      top_k_indicies = self._candidate_indicies[np.argsort(scored_embeddings)[:top_k]]
    elif reverse_candidate_scores and top_k is None:
      top_k_indicies = self._candidate_indicies[np.flip(np.argsort(scored_embeddings))]
    elif not reverse_candidate_scores and top_k is None:
      top_k_indicies = self._candidate_indicies[np.argsort(scored_embeddings)]
    else:
      raise KeyError("Valid arguments for reverse_candidate_scores and top_k must be set.")
    results = []
    for rank, candidate_index in enumerate(top_k_indicies):
      candidate_score = scored_embeddings[candidate_index]
      candidate_meta_map = self._meta_map[candidate_index]
      candidate_verbalisation = candidate_meta_map['verbalisation']
      candidate_iri = candidate_meta_map['iri']
      results.append((rank, candidate_iri, candidate_score, candidate_verbalisation))
    return results

  @abstractmethod
  def register_model(self, model: str) -> None:
    pass

  @abstractmethod
  def register_local_model(self, model_fp: Path) -> None:
    pass

  @abstractmethod
  def _embed(self, query_string: str) -> np.ndarray:
    pass


class HiTRetriever(BaseModelRetriever):

  @override
  def register_model(self, model: str) -> None:
    self._model = HierarchyTransformer.from_pretrained(model)

  @override
  def register_local_model(self, model_fp: Path) -> None:
    self._model = HierarchyTransformer.from_pretrained(str(model_fp.expanduser().resolve()))  

  @override
  def _embed(self, query_string: str) -> np.ndarray:
    return (self._model.encode(
      [query_string]
    ).astype("float32"))[0] # type: ignore
  

class OnTRetriever(BaseModelRetriever):

  @override
  def register_model(self, model: str) -> None:
    self._model = OntologyTransformer.load(model)

  @override
  def register_local_model(self, model_fp: Path) -> None:
    self._model = OntologyTransformer.load(str(model_fp.expanduser().resolve()))

  @override
  def _embed(self, query_string: str) -> np.ndarray:
    return (self._model.encode_concept(
      [query_string]
    ).astype("float32"))[0] # type: ignore
  

class SBERTRetriever(BaseModelRetriever):

  @override
  def register_model(self, model: str) -> None:
    self._model = SentenceTransformer.load(model)

  @override
  def register_local_model(self, model_fp: Path) -> None:
    self._model = SentenceTransformer.load(str(model_fp.expanduser().resolve()))

  @override
  def _embed(self, query_string: str) -> np.ndarray:
    return (self._model.encode(
      [query_string]
    ).astype("float32"))[0] # type: ignore




# Everything below here is a mess (actally, this file more broadly is a mess), TODO: fix

class BM25Retriever(BaseRetriever):
    
    _bm25: BM25Okapi
    _tokenizer: Callable[[str], Sequence[str]]

    @validate_call
    def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, *, 
                 tokenizer: Callable[[str], Sequence[str]] | None = None, 
                 k1: float = 1.3, b: float = 0.7) -> None:
        # k1 should be tuned between 0.5 and 2 in increments of 0.1
        # b should be tuned between 0 and 1 in increments of 0.05
        super().__init__(verbalisations_fp, meta_map_fp)

        if tokenizer is None:
          tokenizer = lambda text: text.lower().split()
        self._tokenizer = tokenizer

        def _resolve_verbalisation(v):
          if isinstance(v, str):
            return v
          if isinstance(v, dict) and "verbalisation" in v:
            return v["verbalisation"]
          raise ValueError("Could not resolve a verbalisation string.")

        tokenised_corpus = [
          self._tokenizer(_resolve_verbalisation(v))
          for v in self._verbalisations
        ]

        self._bm25 = BM25Okapi(tokenised_corpus, k1=k1, b=b)

    def retrieve(self, query_string: str, *, top_k: int | None = None, **kwargs) -> list[QueryResult]:
      tokens = self._tokenizer(query_string)
      scores = self._bm25.get_scores(tokens)
      if top_k is not None:
        top_idx = np.argsort(scores)[::-1][:top_k]
      else:
        top_idx = np.argsort(scores)[::-1]
      results: list[QueryResult] = []
      for rank, idx in enumerate(top_idx):
        iri = self._meta_map[idx]["iri"]
        verbalisation = (
          self._verbalisations[idx] if isinstance(self._verbalisations[idx], str)
          else self._verbalisations[idx]["verbalisation"]
        )
        results.append(
          QueryResult(
              rank=rank,
              iri=iri,
              score=float(scores[idx]),
              verbalisation=verbalisation,
          )
        )
      return results


class TFIDFRetriever(BaseRetriever):
    
    _vectorizer: TfidfVectorizer
    _inverted_index: dict[str, list[tuple[int, float]]]
    _tfidf_matrix: csr_matrix
    _tokenizer: Callable[[str], Sequence[str]] | None

    @validate_call
    def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, *,
        lowercase: bool = True, stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 1),
        tokenizer: Callable[[str], Sequence[str]] | None = None,
        max_features: int | None = None,
    ) -> None:
        
        super().__init__(verbalisations_fp, meta_map_fp)

        def _resolve(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and "verbalisation" in v:
                return v["verbalisation"]
            raise ValueError("Unexpected verbalisation item: %r" % v)

        corpus: list[str] = [_resolve(v) for v in self._verbalisations]

        self._vectorizer = TfidfVectorizer(
          stop_words="english",
          use_idf=True,
          smooth_idf=True,
          # norm="l2"
          norm=None
        )
        doc_term_matrix = self._vectorizer.fit_transform(corpus)
        vocab = self._vectorizer.get_feature_names_out()
        inverted_index: dict[str, list[tuple[int, float]]] = {term: [] for term in vocab}
        coo = coo_matrix(doc_term_matrix)
        
        for row, col, score in zip(coo.row, coo.col, coo.data):
          inverted_index[str(vocab[col])].append((int(row), float(score)))
        
        for postings in inverted_index.values():
          postings.sort(key=lambda x: x[1], reverse=True)

        self._inverted_index = inverted_index

    def retrieve(self, query_string: str, *, top_k: int | None = None, **kwargs) -> list[QueryResult]:
        query_vec = self._vectorizer.transform([query_string])
        vocab = self._vectorizer.get_feature_names_out()
        q_weights = {
            vocab[col]: float(val)
            for col, val in zip(query_vec.indices, query_vec.data) # type: ignore
            if val > 0.0
        }
        tfidf_scores = aggregate_posting_scores(q_weights, self._inverted_index)
        if top_k:
          tfidf_top = topk(tfidf_scores, top_k)
        else:
          tfidf_top = topk(tfidf_scores, len(tfidf_scores))
        
        results: list[QueryResult] = []
        for rank, (doc_id, score) in enumerate(tfidf_top, 1):
          iri = self._meta_map[doc_id]['iri']
          verbalisation = self._meta_map[doc_id]['verbalisation']
          results.append(
            QueryResult(
              rank=rank,
              iri=iri,
              score=float(score),
              verbalisation=verbalisation,
            )
          )
        return results


def mixed_product_distance(d_hit: np.ndarray, d_ont: np.ndarray, d_sbert: np.ndarray, 
                           sigma: tuple[float, float, float] = (1.0, 1.0, 1.0),
                           to_similarity: bool = True, kernel: str = "exp") -> np.ndarray:
    
    sigma_hit, sigma_ont, sigma_sbert = sigma
    d2 = (sigma_hit * d_hit)**2 + (sigma_ont * d_ont)**2 + (sigma_sbert * d_sbert)**2
    if kernel == "dist":
      return np.sqrt(d2)
    if kernel == "exp":
      return np.exp(-np.sqrt(d2)) # rbf
    if kernel in {"inv", "inverse"}:
        return 1.0 / (1.0 + d2) # inverse‑quad
    raise ValueError("no valid kernel given")


class MixedModelRetriever(BaseRetriever):

    _hit_model: HierarchyTransformer
    _ont_model: OntologyTransformer
    _sbert_model: SentenceTransformer

    _hit_embs:   np.ndarray
    _ont_embs:   np.ndarray
    _sbert_embs: np.ndarray

    _sigma: np.ndarray

    def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, *,
        hit_model: HierarchyTransformer, hit_embeddings_fp: Path,
        ont_model: OntologyTransformer, ont_embeddings_fp: Path,
        sbert_model: SentenceTransformer, sbert_embeddings_fp: Path,
        sigma: tuple[float, float, float] = (1.0, 1.0, 1.0),
        kernel: str = "exp") -> None:

        super().__init__(verbalisations_fp, meta_map_fp)

        self._hit_model = hit_model
        self._ont_model = ont_model
        self._sbert_model = sbert_model

        self._hit_embs = np.load(hit_embeddings_fp, mmap_mode="r")
        self._ont_embs = np.load(ont_embeddings_fp, mmap_mode="r")
        self._sbert_embs = np.load(sbert_embeddings_fp, mmap_mode="r")

        assert len(self._hit_embs) == len(self._ont_embs) == len(self._sbert_embs), \
            "all embedding files must contain the same number of rows"

        self._candidate_indices = np.arange(len(self._hit_embs))
        self._sigma  = np.asarray(sigma, dtype=np.float32)
        self._kernel = kernel

    def set_sigma(self, sigma: tuple[float, float, float]) -> None:
        self._sigma = np.asarray(sigma, dtype=np.float32)

    def get_sigma(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in self._sigma) # type: ignore

    def retrieve(self, query_string: str, *, top_k: int | None = None, reverse_candidate_scores: bool = False, **kwargs) -> list[QueryResult]:

        q_hit = self._hit_model.encode([query_string], normalize_embeddings=False)[0]
        q_ont = self._ont_model.encode_concept([query_string])[0]
        q_sbert = self._sbert_model.encode([query_string], normalize_embeddings=True)[0]

        d_hit = batch_poincare_dist_with_adaptive_curv_k(q_hit, self._hit_embs, self._hit_model)
        d_ont = batch_poincare_dist_with_adaptive_curv_k(q_ont, self._ont_embs, self._ont_model)
        d_sbert = batch_euclidian_l2_distance(q_sbert, self._sbert_embs)

        scores = mixed_product_distance(
            d_hit=d_hit,
            d_ont=d_ont,
            d_sbert=d_sbert,
            sigma=tuple(self._sigma),
            to_similarity=True,
            kernel=self._kernel,
        )

        if reverse_candidate_scores and top_k is not None:
            top_idx = np.argsort(scores)[:top_k]
        elif not reverse_candidate_scores and top_k is not None:
            top_idx = np.argsort(-scores)[:top_k]
        elif reverse_candidate_scores and top_k is None:
            top_idx = np.argsort(scores)
        elif not reverse_candidate_scores and top_k is None:
            top_idx = np.argsort(-scores)
        else:
            raise KeyError("Invalid Argument Exception.")

        results: list[QueryResult] = []
        for rank, idx in enumerate(top_idx):
            meta = self._meta_map[idx]
            results.append(
                QueryResult(
                    rank = rank,
                    iri = meta["iri"],
                    score = float(scores[idx]),
                    verbalisation = meta["verbalisation"],
                )
            )
        return results
    
def custom_mixed_product_distance(d_ont_32: np.ndarray, d_ont_128: np.ndarray, d_sbert: np.ndarray, 
                                  sigma: tuple[float, float, float] = (1.0, 1.0, 1.0),
                                  to_similarity: bool = True, kernel: str = "exp") -> np.ndarray:
    
    sigma_hit, sigma_ont, sigma_sbert = sigma
    d2 = (sigma_hit * d_ont_32)**2 + (sigma_ont * d_ont_128)**2 + (sigma_sbert * d_sbert)**2
    if kernel == "dist":
      return np.sqrt(d2)
    if kernel == "exp":
      return np.exp(-np.sqrt(d2)) # rbf
    if kernel in {"inv", "inverse"}:
        return 1.0 / (1.0 + d2) # inverse‑quad
    raise ValueError("no valid kernel given")


class CustomMixedModelRetriever(BaseRetriever):

    _ont_model_32: OntologyTransformer
    _ont_model_128: OntologyTransformer
    _sbert_model: SentenceTransformer

    _ont_embs_32:   np.ndarray
    _ont_embs_128:   np.ndarray
    _sbert_embs: np.ndarray

    _sigma: np.ndarray

    def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, *,
        ont_model_32: OntologyTransformer, ont_32_embeddings_fp: Path,
        ont_model_128: OntologyTransformer, ont_128_embeddings_fp: Path,
        sbert_model: SentenceTransformer, sbert_embeddings_fp: Path,
        sigma: tuple[float, float, float] = (1.0, 1.0, 1.0),
        kernel: str = "exp") -> None:

        super().__init__(verbalisations_fp, meta_map_fp)

        self._ont_model_32 = ont_model_32
        self._ont_model_128 = ont_model_128
        self._sbert_model = sbert_model

        self._ont_embs_32 = np.load(ont_32_embeddings_fp, mmap_mode="r")
        self._ont_embs_128 = np.load(ont_128_embeddings_fp, mmap_mode="r")
        self._sbert_embs = np.load(sbert_embeddings_fp, mmap_mode="r")

        assert len(self._ont_embs_32) == len(self._ont_embs_128) == len(self._sbert_embs), \
            "all embedding files must contain the same number of rows"

        self._candidate_indices = np.arange(len(self._ont_embs_32))
        self._sigma  = np.asarray(sigma, dtype=np.float32)
        self._kernel = kernel

    def set_sigma(self, sigma: tuple[float, float, float]) -> None:
        self._sigma = np.asarray(sigma, dtype=np.float32)

    def get_sigma(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in self._sigma) # type: ignore

    def retrieve(self, query_string: str, *, top_k: int | None = None, reverse_candidate_scores: bool = False, **kwargs) -> list[QueryResult]:

        q_ont_32 = self._ont_model_32.encode_concept([query_string])[0]
        q_ont_128 = self._ont_model_128.encode_concept([query_string])[0]
        q_sbert = self._sbert_model.encode([query_string], normalize_embeddings=True)[0]

        d_ont_32 = batch_poincare_dist_with_adaptive_curv_k(q_ont_32, self._ont_embs_32, self._ont_model_32)
        d_ont_128 = batch_poincare_dist_with_adaptive_curv_k(q_ont_128, self._ont_embs_128, self._ont_model_128)
        d_sbert = batch_euclidian_l2_distance(q_sbert, self._sbert_embs)

        scores = custom_mixed_product_distance(
            d_ont_32=d_ont_32,
            d_ont_128=d_ont_128,
            d_sbert=d_sbert,
            sigma=tuple(self._sigma),
            to_similarity=True,
            kernel=self._kernel,
        )

        if reverse_candidate_scores and top_k is not None:
            top_idx = np.argsort(scores)[:top_k]
        elif not reverse_candidate_scores and top_k is not None:
            top_idx = np.argsort(-scores)[:top_k]
        elif reverse_candidate_scores and top_k is None:
            top_idx = np.argsort(scores)
        elif not reverse_candidate_scores and top_k is None:
            top_idx = np.argsort(-scores)
        else:
            raise KeyError("Invalid Argument Exception.")

        results: list[QueryResult] = []
        for rank, idx in enumerate(top_idx):
            meta = self._meta_map[idx]
            results.append(
                QueryResult(
                    rank = rank,
                    iri = meta["iri"],
                    score = float(scores[idx]),
                    verbalisation = meta["verbalisation"],
                )
            )
        return results


class RetrieverFactory:
   
    _common_map_fp: Path
    _common_verbalisations_fp: Path

    @validate_call
    def __init__(self, common_map_fp: Path, common_verbalisations_fp: Path):
       self._common_map_fp = common_map_fp
       self._common_verbalisations_fp = common_verbalisations_fp

    # bit of weird pattern actually, TODO: (i) reconsider this implementation, (ii) create a type def for all encoders instead of using `Any`
    def construct(self, encoder_class: Any, embeddings_fp: Path, model_str: str, score_fn: Callable, *args, **kwargs) -> Any:
        return encoder_class(
           embeddings_fp=embeddings_fp,
           meta_map_fp=self._common_map_fp,
           verbalisations_fp=self._common_verbalisations_fp,
           model_str=model_str,
           score_fn=score_fn,
           **kwargs
        )