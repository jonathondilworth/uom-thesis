from __future__ import annotations
from typing import Callable, Optional, override
from pathlib import Path
import math
import numpy as np
import torch

from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer
from sentence_transformers import SentenceTransformer

from thesis.utils.retrievers import (
    BaseRetriever
)

from copy import copy, deepcopy

"""
GPU accelerated retrievers for SBERT, HiT and OnT
caches matrix operations associated with hyperbolic dist calculations
within GPU memory, speeds up retrievers by ~x3:x12 compared to:
 28 minutes to run HiT RAG-QA vs 75 minutes with non-cached GPU
    vs multiple hours with CPU

  Only supports HiT/OnT with hyperbolic dist 
    (see implementation & math_functools for details)
"""

class BaseModelRetriever(BaseRetriever):

    _embeddings: np.ndarray # memmap
    _candidate_indicies: np.ndarray
    _model: SentenceTransformer | HierarchyTransformer | OntologyTransformer
    _score_fn: Callable

    # device-resident embeddings tensor
    _embeddings_cache: torch.Tensor
    _embeddings_norm_squared: torch.Tensor # ||vs||^2 (cached)
    _cache_hyperbolic_denominator_B: torch.Tensor  # 1 - (k * ||vs||^2 + eps)
    _curv_k: Optional[float] # model-specific section curvature   # ^ (cached)
    _eps: float

    _torch_device: Optional[str]
    _torch_dtype: Optional["torch.dtype"]
    _resident: bool

    def __init__(self, verbalisations_fp: Path, meta_map_fp: Path, embeddings_fp: Path, *, score_fn: Callable | None = None, model_fp: Path | None = None,
                model_str: str | None = None, resident: bool = True, torch_device: Optional[str] = None, torch_dtype: Optional["torch.dtype"] = None, 
                eps: float = 1e-7):
        
        super().__init__(verbalisations_fp, meta_map_fp)

        # host memmap -> float32 for BLAS/cuBLAS
        self._embeddings = np.load(embeddings_fp, mmap_mode="r").astype(np.float32, copy=False)
        self._candidate_indicies = np.arange(len(self._embeddings), dtype=np.int64)

        # torch setup (|| acts as a conditional branch)
        self._torch_device = torch_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._torch_dtype = torch_dtype or torch.float32

        self._resident = bool(resident)
        self._eps = float(eps)
        self._curv_k = None

        if score_fn: # register a custom score function if provided
            self.register_score_function(score_fn)
        else: # register the default (TODO: move to appropriate inheritance layer)
            self.register_score_function(self._poincare_geodesic_distance)

        if model_fp:
            try:
                self.register_local_model(model_fp.expanduser().resolve())
            except FileNotFoundError:
                self.register_model(str(model_fp))
        elif (not model_fp and model_str):
            self.register_model(model_str)

        if self._resident and self._curv_k:
            self._prepare_cache_for_poincare_embeddings()

    def register_model(self, model: str) -> None:
        raise NotImplementedError

    def register_local_model(self, model_fp: Path) -> None:
        raise NotImplementedError

    def _embed(self, query_string: str) -> np.ndarray:
        raise NotImplementedError

    # HiT/OnT specific, sets curv_k to None for SBERT & other models
    def _set_curvature_from_model(self):
        if isinstance(self._model, HierarchyTransformer):
            self._curv_k = float(self._model.get_circum_poincareball(self._model.embed_dim).c)
        elif isinstance(self._model, OntologyTransformer):
            self._curv_k = float(self._model.hit_model.get_circum_poincareball(self._model.hit_model.embed_dim).c)
        else:
            self._curv_k = None

    # HiT/OnT specific
    def _prepare_cache_for_poincare_embeddings(self):
        if not self._curv_k:
            raise RuntimeError("Not an appropriate retriever for use with hyperbolic distance metrics.")
        
        # embs_copy = deepcopy(self._embeddings)
        host = torch.from_numpy(self._embeddings).pin_memory() # pin embeddings
        # host = torch.from_numpy(embs_copy).pin_memory() # pin embeddings
        self._embeddings_cache = host.to(self._torch_device, dtype=self._torch_dtype, non_blocking=True).contiguous()
        with torch.inference_mode():
            self._embeddings_norm_squared = (self._embeddings_cache * self._embeddings_cache).sum(dim=1)
            self._cache_hyperbolic_denominator_B = 1.0 - (self._curv_k * self._embeddings_norm_squared + self._eps)

    # HiT/OnT specific & see `math_functools.py` for implementation details
    def _poincare_geodesic_distance(self, u_vec: np.ndarray, _unused=None, **kwargs):
        if not self._curv_k:
            raise RuntimeError("Poincare geodesics calc requires sectional curvature k from model (HiT/OnT).")
        if not self._resident:
            from math_functools import efficient_batch_poincare_distance_with_curv_k
            return efficient_batch_poincare_distance_with_curv_k(u_vec, self._embeddings, np.float64(self._curv_k))
        if self._embeddings_cache is None:
            self._prepare_cache_for_poincare_embeddings()    
        with torch.inference_mode():
            u = torch.as_tensor(u_vec, device=self._torch_device, dtype=self._torch_dtype)
            u_norm_squared = (u * u).sum()
            hyperbolic_denominator_A = 1.0 - (self._curv_k * u_norm_squared + self._eps)
            u_vs = torch.mv(self._embeddings_cache, u)
            l2_dist_squared = self._embeddings_norm_squared + u_norm_squared - 2.0 * u_vs # GEMV (see: math_functools.py)
            acosh_arg = 1.0 + (2.0 * self._curv_k * l2_dist_squared) / (hyperbolic_denominator_A * self._cache_hyperbolic_denominator_B)
            arg = torch.clamp_min(acosh_arg, 1.0) # domain of acosh: [1.0, inf)
            scaling_factor_k = 1.0 / math.sqrt(self._curv_k)
            return scaling_factor_k * torch.acosh(arg)

    def register_score_function(self, score_fn: Callable):
        self._score_fn = score_fn

    def retrieve(self, query_string: str, *, top_k: int | None = None, reverse_candidate_scores=False, **kwargs):
        q = self._embed(query_string).astype("float32")
        bank = self._embeddings_cache if self._embeddings_cache is not None else self._embeddings
        scores = self._score_fn(q, bank, **kwargs)
        n = scores.shape[0]
        # retrieve top_k results
        if top_k is None or top_k >= n:
            idx = self._argsort(scores, reverse_candidate_scores)
        else:
            idx = self._topk_indices(scores, top_k, reverse_candidate_scores)
        idx_host = idx.detach().cpu().numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx)
        sel_scores = scores[idx].detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)[idx_host]
        results = []
        for rank, candidate_index in enumerate(idx_host):
            mm = self._meta_map[int(candidate_index)]
            candidate_score = float(sel_scores[rank])
            results.append((rank, mm['iri'], candidate_score, mm['verbalisation']))
        return results

    def _argsort(self, scores, reverse: bool):
        if reverse:
            return torch.argsort(-scores)
        return torch.argsort(scores)

    def _topk_indices(self, scores, k: int, reverse: bool):
        if reverse:
            vals, inds = torch.topk(scores, k=k, largest=True, sorted=True)
        else:
            vals, inds = torch.topk(-scores, k=k, largest=True, sorted=True)
            inds = inds
        return inds


class GPUHiTRetriever(BaseModelRetriever):

  @override
  def register_model(self, model: str) -> None:
    self._model = HierarchyTransformer.from_pretrained(model)
    self._set_curvature_from_model()
    self._prepare_cache_for_poincare_embeddings()

  @override
  def register_local_model(self, model_fp: Path) -> None:
    self._model = HierarchyTransformer.from_pretrained(str(model_fp.expanduser().resolve()))
    self._set_curvature_from_model()
    self._prepare_cache_for_poincare_embeddings()

  @override
  def _embed(self, query_string: str) -> np.ndarray:
    return (self._model.encode(
      [query_string]
    ).astype("float32"))[0] # type: ignore
  

class GPUOnTRetriever(BaseModelRetriever):

  @override
  def register_model(self, model: str) -> None:
    self._model = OntologyTransformer.load(model)
    self._set_curvature_from_model()
    self._prepare_cache_for_poincare_embeddings()

  @override
  def register_local_model(self, model_fp: Path) -> None:
    self._model = OntologyTransformer.load(str(model_fp.expanduser().resolve()))
    self._set_curvature_from_model()
    self._prepare_cache_for_poincare_embeddings()

  @override
  def _embed(self, query_string: str) -> np.ndarray:
    return (self._model.encode_concept(
      [query_string]
    ).astype("float32"))[0] # type: ignore