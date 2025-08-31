from __future__ import annotations
from typing import Iterable
from hierarchy_transformers import HierarchyTransformer # type: ignore # <- temporary (TODO)
from OnT.OnT import OntologyTransformer
import numpy as np
import torch
import math


"""TODO: doc-comments, unit tests"""


def identity(x):
    return x


# specific to calculating nDCG
def add(a, b, key='dcg'):
    return a + b[key]


def batch_euclidian_l2_distance(u: np.ndarray, vs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u - vs, axis=1)


def l2_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.sum(x**2))


def batch_l2_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.asarray(np.sqrt(np.sum(x**2, axis=1)))


def inner_product(p_u: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    u = np.asarray(p_u, dtype=np.float32)
    v = np.asarray(p_v, dtype=np.float32)
    return np.inner(u, v)


def batch_inner_product(p_u: np.ndarray, p_vs: np.ndarray) -> np.ndarray:
    u = np.asarray(p_u, dtype=np.float32).ravel()
    vs = np.asarray(p_vs, dtype=np.float32)
    return vs.dot(u)


def cosine_similarity(u, v, normalised=True):
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return np.inner(u, v) if normalised else np.inner(u, v) / (l2_norm(u) * l2_norm(v))


def batch_cosine_similarity(p_u, p_vs, normalised=True):
    u  = np.asarray(p_u,  dtype=np.float32)
    vs = np.asarray(p_vs, dtype=np.float32)
    return batch_inner_product(u, vs) if normalised else batch_inner_product(u, vs) / (l2_norm(u) * batch_l2_norm(vs))


# https://medium.com/@heyamit10/understanding-numpy-ascontiguousarray-with-practical-examples-a71d639fe65a
def efficient_batch_poincare_distance_with_curv_k(u: np.ndarray, vs: np.ndarray, k: np.float32 | np.float64):
    vs = np.ascontiguousarray(vs, dtype=np.float64) # use float64 on CPU
    u = np.asarray(u, dtype=np.float64)
    eps_offset = 1e-7
    u_norm_sqd = np.dot(u, u) # more efficient (than element-wise: np.sum(u ** 2))
    vs_norm_sqd = np.matmul(vs, vs) # ^
    denom_term_A = 1 - (k * u_norm_sqd + eps_offset)
    denom_term_B = 1 - (k * vs_norm_sqd + eps_offset)
    # linalg gemv: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html
    # ^ operation covered in documentation from intel & ARM, so looks to be CPU optimised; also..
    # see: https://stackoverflow.com/questions/29752994/computing-all-pairs-distances-between-points-in-different-sets-with-cuda
    #   for now, I'm going to leave this operation as CPU bound... but may be worth offloading to GPU at some point..  ^ mind blow ^
    #   (if these operations can be efficiently offloaded to GPU, why don't any of the well known vector stores support hyperbolic embeddings?)
    vs_u = vs @ u # see ^ `||x-y||^2 = ||x||^2+||y||^2-2*<x,y>`
    l2_dist_sqd = vs_norm_sqd + u_norm_sqd - 2.0 * vs_u
    # now, we can easily calculate: d_k(x,y) = \frac{1}{\sqrt{k}} \ \text{arcosh} \left( 1 + \frac{2k\|x - y\|^2}{(1 - k \|x\|^2)(1 - k\|y\|^2)} \right)
    acosh_arg = np.maximum((1.0 + (2.0 * k * l2_dist_sqd) / (denom_term_A * denom_term_B)), 1.0) # bound to [1, inf)
    scaling_factor_k = 1.0 / np.sqrt(k)
    return scaling_factor_k * np.arccosh(acosh_arg)


def batch_poincare_distance_with_curv_k(u: np.ndarray, vs: np.ndarray, k: np.float64 | np.float32) -> np.float64 | np.float32:
    u_norm_sqd = np.sum(u**2)
    vs_norms_sqd = np.sum(vs**2, axis=1)
    l2_dist_sqd = np.sum((u - vs)**2, axis=1)
    offset = 1e-7 # tiny-offset: guard agaisnt division by zero & floating point arithmatic inaccuracies (eps)
    arg = 1 + ((2 * k * l2_dist_sqd) / ((1 - (k * u_norm_sqd + offset)) * (1 - (k * vs_norms_sqd + offset)))) # acosh
    arg = np.maximum(1.0, arg) # bounds check: domain of acosh is bound to [1, \inf)
    acosh_scaling = np.float64(1) / np.float64(np.sqrt(k)) # scaling factor: k
    return (acosh_scaling * np.arccosh(arg, dtype=np.float64)) # 1 / sqrt(k) * acosh(arg)


def batch_poincare_distance_with_curv_k_torch(u: np.ndarray, vs: np.ndarray, k: float, * , device: str | None = None, dtype: torch.dtype | None = None, batch_size: int | None = None) -> np.ndarray:
    """may (or may not) be more efficient than having the CPU deal with the calculation; the emebddings files are ~549MB"""
    if device is None: # .. what do we do about multi-GPU? DataParallel/FSDP neccesary here?
        device = "cuda:0" if torch.cuda.is_available() else "cpu" # <- default to 'cuda:0'
    if dtype is None: # torch.float32 for the sake of memory efficiency
        dtype = torch.float32 # precision is likely 'good enough'
    if batch_size is None or batch_size <= 0:
       batch_size = len(vs) # potentially huge (reccomended ~8k)
    eps = 1e-7 # (~) https://www.mathworks.com/help/matlab/ref/double.eps.html
    scaling_factor_k = 1.0 / math.sqrt(k)
    u_t = torch.as_tensor(u, device=device, dtype=dtype)
    u_norm_sqd = (u_t * u_t).sum()
    outs = []
    with torch.inference_mode():
        # batching: (..or chunking)
        for start in range(0, len(vs), batch_size):
            vs_chunk = vs[start:start + batch_size] # iterative batch/chunking
            vs_t = torch.as_tensor(vs_chunk, device=device, dtype=dtype)
            vs_norms_sqd = (vs_t * vs_t).sum(dim=1)
            l2_dist_sqd = ((vs_t - u_t) ** 2).sum(dim=1)
            denom = (1.0 - (k * u_norm_sqd + eps)) * (1.0 - (k * vs_norms_sqd + eps))
            arg = 1.0 + (2.0 * k * l2_dist_sqd) / denom
            arg = torch.clamp_min(arg, 1.0)  # acosh domain: [1, +inf)
            d = scaling_factor_k * torch.acosh(arg)
            outs.append(d.detach().to("cpu"))
    return torch.cat(outs, dim=0).numpy()


def batch_poincare_dist_with_adaptive_curv_k(u: np.ndarray, vs:np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    if isinstance(model, HierarchyTransformer):
        k = np.float64(model.get_circum_poincareball(model.embed_dim).c)
    elif isinstance(model, OntologyTransformer):
        hierarchy_model = model.hit_model
        hierarchy_poincare_ball = hierarchy_model.get_circum_poincareball(hierarchy_model.embed_dim)
        k = np.float64(hierarchy_poincare_ball.c)    
    else:
        raise Exception("Hyperbolic distance should only be only calculated in B^n or H^n")
    return np.asarray(batch_poincare_distance_with_curv_k(u, vs, k))


def batch_poincare_dist_with_adaptive_curv_k_torch(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    if not torch.cuda.is_available():
       print("Warning: explicit call to torch-based batch poincare dist fn, but CUDA is not available.")
       batch_poincare_dist_with_adaptive_curv_k(u, vs, model=model, **kwargs)
    # else: (CUDA is available)
    if isinstance(model, HierarchyTransformer):
        k = np.float64(model.get_circum_poincareball(model.embed_dim).c)
    elif isinstance(model, OntologyTransformer):
        hierarchy_model = model.hit_model
        hierarchy_poincare_ball = hierarchy_model.get_circum_poincareball(hierarchy_model.embed_dim)
        k = np.float64(hierarchy_poincare_ball.c)
    else:
        raise Exception("Hyperbolic distance should only be only calculated in B^n or H^n")
    # happy path:
    device = kwargs.get("device", "cuda:0")
    dtype = kwargs.get("dtype", torch.float32)
    batch_size = kwargs.get("batch_size", None) # => batch_size := len(vs)
    return np.asarray(batch_poincare_distance_with_curv_k_torch(u, vs, float(k), device=device, dtype=dtype, batch_size=batch_size))


# pretty sure `vs` needs to be pinned for this to improve the performance; otherwise we're going to flood the PCIe bus
def gpu_cpu_batch_poincare_dist_with_adaptive_curv_k(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    if torch.cuda.is_available():
        return batch_poincare_dist_with_adaptive_curv_k_torch(u, vs, model, **kwargs)
    # else:
    return batch_poincare_dist_with_adaptive_curv_k(u, vs, model, **kwargs)
    

def subsumption_score_hit(hit_transformer: HierarchyTransformer, child_emb: np.ndarray | torch.Tensor, parent_emd: np.ndarray | torch.Tensor, centri_weight: float = 1.0):
    child_emb_t = torch.Tensor(child_emb)
    parent_emb_t = torch.Tensor(parent_emd)
    dists = hit_transformer.manifold.dist(child_emb_t, parent_emb_t)
    child_norms = hit_transformer.manifold.dist0(child_emb_t)
    parent_norms = hit_transformer.manifold.dist0(parent_emb_t)
    return -(dists + centri_weight * (parent_norms - child_norms))


def subsumption_score_ont(ontology_transformer: OntologyTransformer, child_emb: np.ndarray | torch.Tensor, parent_emb: np.ndarray | torch.Tensor, weight_lambda: float = 1.0):
    child_emb_t = torch.Tensor(child_emb)
    parent_emb_t = torch.Tensor(parent_emb)
    return ontology_transformer.score_hierarchy(child_emb_t, parent_emb_t, weight_lambda)


def entity_subsumption(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer, *, weight: float = 0.4, **kwargs):
    return np.asarray(subsumption_score_hit(model, u, vs, centri_weight=weight))


def concept_subsumption(u: np.ndarray, vs: np.ndarray, model: OntologyTransformer, *, weight: float = 0.4, **kwargs):
    return np.asarray(subsumption_score_ont(model, u, vs, weight_lambda=weight))


def dcg_exp_relevancy_at_pos(relevancy: int, rank_position: int) -> float:
    if relevancy <= 0:
      return float(0.0)
    numerator = (2**relevancy) - 1
    denominator = math.log2(rank_position + 1)
    return float(numerator / denominator)


def compute_ndcg_at_k(results: list[tuple[int, str, float, str]], targets_with_dcg_exp: list[dict], k: int = 20) -> float:
    relevance_map = {target['iri']: target['relevance'] for target in targets_with_dcg_exp}
    dcg = 0.0
    for rank, (idx, iri, score, label) in enumerate(results[:k], start=1):
        rel = relevance_map.get(iri, 0)
        dcg += dcg_exp_relevancy_at_pos(rel, rank)
    ideal_dcg = sum(target['dcg'] for target in targets_with_dcg_exp[:k])
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def average_precision_binary(rels: Iterable[int]) -> float:
    rels = list(int(r > 0) for r in rels)
    num_rel = sum(rels)
    if num_rel == 0:
        return 0.0
    hits = 0
    cum_prec = 0.0
    for k, r in enumerate(rels, start=1):
        if r:
            hits += 1
            cum_prec += hits / k
    return cum_prec / num_rel


def pr_points_from_binary(rels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    rels = np.asarray(list(int(r > 0) for r in rels), dtype=int)
    num_rel = int(rels.sum())
    if num_rel == 0:
        return np.array([]), np.array([])
    hits = 0
    ps, rs = [], []
    for k, r in enumerate(rels, start=1):
        if r:
            hits += 1
            ps.append(hits / k)
            rs.append(hits / num_rel)
    return np.asarray(rs, dtype=float), np.asarray(ps, dtype=float)


def interpolate_precision(recall: np.ndarray, precision: np.ndarray, recall_grid: np.ndarray) -> np.ndarray:
    if recall.size == 0:
        return np.zeros_like(recall_grid, dtype=float)
    # sort recall, ensuring monotonicity
    order = np.argsort(recall)
    r = recall[order] # recall in asc (smallest -> largest)
    p = precision[order] # precision @ i : \forall i \in r(ecall)
    # non-increasing precision (cumulative maxima) from right to left
    p_right_max = p.copy()
    # i <- arg (position) \in prec, reversed in desc
    for i in range(len(p_right_max) - 2, -1, -1):
        if p_right_max[i] < p_right_max[i + 1]:
            p_right_max[i] = p_right_max[i + 1]
    # ^ yields (recall, max(precision)) @ k : \forall k \in \{r_0, r_1, \cdot r_n\} \leftarrow \text{recall}
    # i.e. p, r : p -> max(p), r -> r \forall r \in R (r*)
    interp = np.zeros_like(recall_grid, dtype=float)
    for i, rg in enumerate(recall_grid):
        # find first index where recall >= rg
        idx = np.searchsorted(r, rg, side="left")
        if idx < len(p_right_max):
            interp[i] = p_right_max[idx]
        else:
            interp[i] = 0.0
    return interp


def macro_pr_curve(all_query_rels: list[Iterable[int]], recall_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """macro-averaged, interpolated PR curve; returns: recall grid, macro precision"""
    recall_grid = np.linspace(0.0, 1.0, recall_points)
    acc = np.zeros_like(recall_grid, dtype=float)
    Q = len(all_query_rels)
    if Q == 0:
        return recall_grid, acc
    for rels in all_query_rels:
        r, p = pr_points_from_binary(rels)
        acc += interpolate_precision(r, p, recall_grid)
    macro_p = acc / Q
    return recall_grid, macro_p

