from pathlib import Path
import json
import copy
import latextable
from latextable import texttable
import statistics
from sklearn.metrics import auc as sk_auc
import numpy as np
# ^ python imports
# module/lib imports:
from thesis.utils.retrievers import (
    TFIDFRetriever,
    BM25Retriever,
    SBERTRetriever,
    HiTRetriever,
    OnTRetriever,
    MixedModelRetriever,
    CustomMixedModelRetriever
)
from thesis.utils.math_functools import (
  batch_cosine_similarity,
  batch_euclidian_l2_distance,
  batch_poincare_dist_with_adaptive_curv_k,
  entity_subsumption,
  concept_subsumption,
  macro_pr_curve
)
from thesis.utils.data_utils import load_json
from thesis.utils.query_utils import QueryObjectMapping, QueryResult
from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer
from sentence_transformers import SentenceTransformer

# TEMP: TODO fix

from functools import reduce
from typing import Any
import math


def obj_max_depth(x: int, obj: Any, key: str = "depth") -> int:
  return x if x > obj[key] else obj[key]


def discounted_cumulative_gain():
  pass


def dcg_exp_relevancy_at_pos(relevancy: int, rank_position: int) -> float:
  if relevancy <= 0:
    return float(0.0)
  numerator = (2**relevancy) - 1
  denominator = math.log2(rank_position + 1)
  return float(numerator / denominator)


def add(a, b, key='dcg'):
  return a + b[key]


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

# END TEMP

embeddings_dir = "./embeddings"

common_map = Path(f"{embeddings_dir}/entity_mappings.json")
common_verbalisations = Path(f"{embeddings_dir}/verbalisations.json")

verbalisations = load_json(common_verbalisations)

## Lexical Baseline Retrievers (TF-IDF, BM25):

tfidf_ret = TFIDFRetriever(common_verbalisations, common_map)
bm25_ret = BM25Retriever(common_verbalisations, common_map, k1=1.3, b=0.7)

# SBERT

sbert_plm_hf_string = "all-MiniLM-L12-v2"

sbert_ret_plm_w_cosine_sim = SBERTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/sbert-plm-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_cosine_similarity
)

sbert_ret_plm_w_euclid_dist = SBERTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/sbert-plm-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_euclidian_l2_distance
)

# HiT SNOMED 25 (Full)

hit_snomed_25_model_fp = './models/snomed_models/HiT-mixed-SNOMED-25/final'
hit_SNOMED25_model_path = Path(hit_snomed_25_model_fp)

hit_ret_snomed_25_w_hyp_dist = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_SNOMED25_model_path,
  score_fn=batch_poincare_dist_with_adaptive_curv_k
)

hit_ret_snomed_25_w_ent_sub = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_SNOMED25_model_path,
  score_fn=entity_subsumption
)

# OnT GALEN

ont_galen_23_pred_model_fp = "./models/models/prediction/OnTr-all-MiniLM-L12-v2-GALEN"
ont_galen_pred_model_path = Path(ont_galen_23_pred_model_fp)

ont_ret_galen_pred_w_hyp_dist = OnTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/ont-galen-23-pred-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=ont_galen_pred_model_path,
  score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_ret_galen_pred_w_con_sub = OnTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/ont-galen-23-pred-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=ont_galen_pred_model_path,
  score_fn=concept_subsumption
)

# ANATOMY

ont_anatomy_23_pred_model_fp = "./models/models/prediction/OnTr-all-MiniLM-L12-v2-ANATOMY"
ont_anatonmy_pred_model_path = Path(ont_anatomy_23_pred_model_fp)

ont_ret_anatomy_pred_w_hyp_dist = OnTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/ont-anatomy-23-pred-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=ont_anatonmy_pred_model_path,
  score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_ret_anatomy_pred_w_con_sub = OnTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/ont-anatomy-23-pred-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=ont_anatonmy_pred_model_path,
  score_fn=concept_subsumption
)

# Gene Ontology (GO)

ont_gene_ontology_23_pred_model_fp = "./models/models/prediction/OnTr-all-MiniLM-L12-v2-GO"
ont_go_pred_model_path = Path(ont_gene_ontology_23_pred_model_fp)

ont_ret_go_pred_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-go-23-pred-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_go_pred_model_path,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_ret_go_pred_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-go-23-pred-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_go_pred_model_path,
    score_fn=concept_subsumption
)

# SNOMED CT 2025 (Full)

ontr_snomed_25_uni_model_fp = './models/snomed_models/OnTr-snomed25-uni'
ont_snomed_25_updated_model_path = Path(ontr_snomed_25_uni_model_fp)

ont_ret_snomed_25_updtd_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_snomed_25_updated_model_path,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_ret_snomed_25_updtd_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-25-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_snomed_25_updated_model_path,
    score_fn=concept_subsumption
)

# SNOMED CT 2025 (M-64)

ontr_snomed_minified_model_fp = './models/snomed_models/OnTr-minified-64'
ontr_snomed_minified_model_fp = Path(ontr_snomed_minified_model_fp)

ontr_ret_snomed_minified_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_model_fp,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ontr_ret_snomed_minified_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_model_fp,
    score_fn=concept_subsumption
)

# SNOMED CT 2025 (M-128)

ontr_snomed_minified_128_model_fp = './models/snomed_models/OnTr-m-128'
ontr_snomed_minified_128_model_fp = Path(ontr_snomed_minified_128_model_fp)

ontr_ret_snomed_minified_128_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_128_model_fp,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ontr_ret_snomed_minified_128_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_128_model_fp,
    score_fn=concept_subsumption
)

# SNOMED CT 2025 (M-32)

ontr_snomed_minified_32_model_fp = './models/snomed_models/OnTr-m-32'
ontr_snomed_minified_32_model_fp = Path('./models/snomed_models/OnTr-m-32')

ontr_ret_snomed_minified_32_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-32-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_32_model_fp,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ontr_ret_snomed_minified_32_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-32-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_32_model_fp,
    score_fn=concept_subsumption
)

## product manifold retriever

hit_emb_fp   = Path("./embeddings/hit-snomed-25-embeddings.npy")
ont_emb_fp   = Path("./embeddings/ont-snomed-25-embeddings.npy")
sbert_emb_fp = Path("./embeddings/sbert-plm-embeddings.npy")

product_hit_model = HierarchyTransformer.from_pretrained('./models/snomed_models/HiT-mixed-SNOMED-25/final')
product_ont_model = OntologyTransformer.load('./models/snomed_models/OnTr-snomed25-uni')
product_sbert_model = SentenceTransformer("all-MiniLM-L12-v2")

mixed_ret = MixedModelRetriever(
    verbalisations_fp = common_verbalisations,
    meta_map_fp = common_map,
    hit_model = product_hit_model,
    hit_embeddings_fp = hit_emb_fp,
    ont_model = product_ont_model,
    ont_embeddings_fp = ont_emb_fp,
    sbert_model = product_sbert_model,
    sbert_embeddings_fp = sbert_emb_fp,
    sigma = (1.0, 1.0, 1.0),
    kernel = "exp",
)

## Tripple Mini OnT-Mini + SBERT Product Manifold

ont_m_32_emb_fp    = Path(f"{embeddings_dir}/ont-snomed-minified-32-embeddings.npy")
ont_m_128_emb_fp   = Path(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy")
sbert_emb_fp       = Path(f"{embeddings_dir}/sbert-plm-embeddings.npy")

product_ont_model_32  = OntologyTransformer.load('./models/snomed_models/OnTr-m-32')
product_ont_model_128 = OntologyTransformer.load('./models/snomed_models/OnTr-m-128')
product_sbert_model   = SentenceTransformer("all-MiniLM-L12-v2")

mixed_ret_mini = CustomMixedModelRetriever(
    verbalisations_fp = common_verbalisations,
    meta_map_fp = common_map,
    ont_model_32 = product_ont_model_32,
    ont_32_embeddings_fp = ont_m_32_emb_fp,
    ont_model_128 = product_ont_model_128,
    ont_128_embeddings_fp = ont_m_128_emb_fp,
    sbert_model = product_sbert_model,
    sbert_embeddings_fp = sbert_emb_fp,
    sigma = (1.0, 1.0, 0.35),
    kernel = "exp",
)

#### EXPERIMENTAL RUNS

# load query objects via ORM-style QueryObjectMapping:
data_query_mapping = QueryObjectMapping(Path("./data/eval_dataset_50_ORIGINAL.json"))
equiv_queries, subsumpt_queries = data_query_mapping.get_queries()

# specify the 'cutoff_depth' (i.e. 'd' parameter)
global_cutoff_depth = 5
# global_cutoff_depth = 100 # acts as inf (there are no relations that are > 100 depth)

# copy the original data (as we re-use later)
oov_single_target_queries = copy.deepcopy(subsumpt_queries)
for q in oov_single_target_queries:
    q._ancestors = []
    q._parents = []

oov_match_all = copy.deepcopy(subsumpt_queries)

# set up the 'models dict' ready for experimental runs (single target):

models_dict_multi_target = {
  # baselines
  "BoW Lexical TFIDF": tfidf_ret,
  "BoW Lexical BM25": bm25_ret,
  # baseline contextual
  "SBERT SNOMED25 cos-sim": sbert_ret_plm_w_cosine_sim,
  "SBERT SNOMED25 d_l2": sbert_ret_plm_w_euclid_dist,
  # HiT models
  "HiT SNOMED25(F) d_k": hit_ret_snomed_25_w_hyp_dist,
  "HiT SNOMED25(F) s_e": hit_ret_snomed_25_w_ent_sub,
  # OnT transferability (prediction)
  "OnTr GALEN(P) d_k": ont_ret_galen_pred_w_hyp_dist,
  "OnTr GALEN(P) s_c": ont_ret_galen_pred_w_con_sub,
  "OnTr ANATOMY(P) d_k": ont_ret_anatomy_pred_w_hyp_dist,
  "OnTr ANATOMY(P) s_c": ont_ret_anatomy_pred_w_con_sub,
  "OnTr GO(P) d_k": ont_ret_go_pred_w_hyp_dist,
  "OnTr GO(P) s_c": ont_ret_go_pred_w_con_sub,
  # OnT SNOMED models
  "OnTr SNO-25(F-64) d_k": ont_ret_snomed_25_updtd_w_hyp_dist,
  "OnTr SNO-25(F-64) s_c": ont_ret_snomed_25_updtd_w_con_sub,
  "OnTr SNO-25(M-32) d_k": ontr_ret_snomed_minified_32_w_hyp_dist,
  "OnTr SNO-25(M-32) s_c": ontr_ret_snomed_minified_32_w_con_sub,
  "OnTr SNO-25(M-64) d_k": ontr_ret_snomed_minified_w_hyp_dist,
  "OnTr SNO-25(M-64) s_c": ontr_ret_snomed_minified_w_con_sub,
  "OnTr SNO-25(M-128) d_k": ontr_ret_snomed_minified_128_w_hyp_dist,
  "OnTr SNO-25(M-128) s_c": ontr_ret_snomed_minified_128_w_con_sub,
  # Mixed Model Space (as before)
  "Mixed SNO-25(F-HOS) d_m": mixed_ret,
  "Mixed SNO-25(M-32-128) d_m": mixed_ret_mini
}

# OOV (TARGET + ANCESTORS) [weighted subsumption retrieval, \lambda = 0.35] [50]

# PREP TABLE START #
experiment_three_table = texttable.Texttable()
experiment_three_table.set_deco(texttable.Texttable.HEADER)
experiment_three_table.set_precision(2)
experiment_three_table.set_cols_dtype(['t', 't', 't', 'f', 'f', 'f', 'f', 'f'])
experiment_three_table.set_cols_align(["l", "l", "l", "c", "c", "c", "c", "c"])
experiment_three_table.header(["Model", "Variant", "Metric", "mAP", "MRR*", "nDCG@10", "PR-AUC", "R@100"])
# END-PREP TABLE #

ks      = [1, 3, 5, 10, 100, len(verbalisations)]
MAX_K   = max(ks)

all_results = {}
macro_avg_PR_AUC_data = {}

for model_name, model in models_dict_multi_target.items():
    
    # init accumulators
    results = {
      "MRR": 0.0, # Mean Reciprical Rank
      "MAP": 0.0, # Mean Average Precision
      **{f"Hits@{k}": 0.0 for k in ks},
      **{f"P@{k}": 0.0 for k in ks}, # Precision@k
      **{f"R@{k}": 0.0 for k in ks}, # Recall@k
      **{f"F1@{k}": 0.0 for k in ks}, # F1@k
      **{f"nDCG@{k}": 0.0 for k in ks}, # normalised Discounted Cumlative Gain @ k
      "MR": 0.0, # Mean Rank
      "aRP": 0.0  # R-Precision
    }
    # AUC-PR, Median Rank & Coverage are calculated during the test procedure
    hit_count = 0 # for coverage
    total_possible_hits = 0 # for coverage := hit_count / total_possible_hits .. essentially: recall@k, when k = MAX_K
    all_ranks = [] # for median rank
    # @depricationWarning : AUC-PR, previous implementation was rough approximation
    per_query_rels_for_PR = []

    for q_idx, query in enumerate(oov_match_all):
        
        qstr = query.get_query_string()
        gold_targets = query.get_unique_sorted_subsumptive_targets(key="depth", reverse=False, depth_cutoff=global_cutoff_depth) # [*parents, *ancestors]
        g_target_iris = set([x["iri"] for x in gold_targets])
        num_targets = len(g_target_iris)
        total_possible_hits += num_targets
        average_precision = 0.0
        hit_count_this_query = 0
        hit_count_lt_or_eq_num_targets = 0

        ranked_results: list[QueryResult] = [] # empty lists (are unlikely to exist) but are treated as full misses
        
        # TODO: replace with match (?) - i.e. switch
        if isinstance(model, HiTRetriever):
          if model._score_fn == entity_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=0.35)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, OnTRetriever):
          if model._score_fn == concept_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=0.35)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, SBERTRetriever):
          if model._score_fn == batch_cosine_similarity:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True)
          else:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False)
        #
        elif isinstance(model, BM25Retriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        elif isinstance(model, TFIDFRetriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        elif isinstance(model, MixedModelRetriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        elif isinstance(model, CustomMixedModelRetriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        else:
           raise ValueError("No appropriate retriever has been set.")

        retrieved_iris = [iri for (_, iri, _, _) in ranked_results] # type: ignore

        # (macro) PR-AUC
        rel_binary = []
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
              rel_binary.append(1)
            else:
              rel_binary.append(0)
        per_query_rels_for_PR.append((rel_binary, num_targets))

        # MRR & Mean Rank (on the first hit)
        rank_pos = None
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
                rank_pos = rank_idx
                results["MRR"] += 1.0 / rank_idx
                results["MR"] += rank_idx
                break
        
        # Average Precision (this query), for use in calculating mAP
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count += 1
              hit_count_this_query += 1
              average_precision += hit_count_this_query / rank_idx
        average_precision /= num_targets
        results["MAP"] += average_precision

        # R-Precision (this query)
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count_lt_or_eq_num_targets += 1
           if rank_idx == num_targets: # then we need to calculate the precision @ this index
              results["aRP"] += hit_count_lt_or_eq_num_targets / num_targets
              break

        # include a penalty to appropriately offset the MR
        # rather than artifically inflating the performance
        # by simply dropping queries that do not contain 
        # (unlikely in this case)
        if rank_pos is None:
            results["MR"] += MAX_K + 1 # penalty: rank := MAX_K + 1

        for k in ks:
            hit = 1 if (rank_pos is not None and rank_pos <= k) else 0
            results[f"Hits@{k}"] += hit
            top_k_results = set(retrieved_iris[:k])
            total_hits_at_k = len(g_target_iris.intersection(top_k_results))
            p_at_k = total_hits_at_k / k # Precision@K
            results[f"P@{k}"] += p_at_k
            r_at_k = total_hits_at_k / num_targets
            results[f"R@{k}"] += r_at_k
            if (p_at_k + r_at_k) > 0:
               results[f"F1@{k}"] += 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)
            iDCG, targets_with_dcg = query.get_targets_with_dcg(type="exp", depth_cutoff=global_cutoff_depth)
            results[f"nDCG@{k}"] += compute_ndcg_at_k(ranked_results, targets_with_dcg, k) # type: ignore

        final_rank = rank_pos if rank_pos is not None else MAX_K + 1
        all_ranks.append(final_rank)

    # (macro) PR-AUC
    R_grid, P_macro = macro_pr_curve(per_query_rels_for_PR, recall_points=101)
    macro_pr_auc = float(np.trapz(P_macro, R_grid))

    # normalise over queries & compute coverage
    N = len(oov_match_all)
    normalized = {metric: value / N for metric, value in results.items()}
    normalized['Cov'] = (hit_count / total_possible_hits) # calculate the coverage of this model
    normalized['Med'] = statistics.median(all_ranks) # median rank
    # area under precision-recall curve (trapezodial rule)
    recall_at_k_xs    = [normalized[f"R@{k}"] for k in ks]
    # check for monotonic recall
    if any(r2 < r1 for r1, r2 in zip(recall_at_k_xs, recall_at_k_xs[1:])):
        raise ValueError(f"Recall must be non-decreasing for PR-AUC")
    precision_at_k_xs = [normalized[f"P@{k}"] for k in ks]
    normalized["AUC"] = float(sk_auc(recall_at_k_xs, precision_at_k_xs))
    normalized["MacroPR_AUC"] = macro_pr_auc

    print(f"Model: {model_name}")
    print(f" mAP: \t  {normalized['MAP']:.2f}") # Mean Average Precision
    print(f" MRR*:   {normalized['MRR']:.2f}") # MRR at first hit ranks
    print(f" nDCG@10: {normalized['nDCG@10']:.2f}") # nDCG@10
    print(f" PR-AUC:  {normalized['AUC']:.2f}") # area under precision-recall curve
    print(f" mPR-AUC: {normalized['MacroPR_AUC']:.2f}") # PR-AUC (macro averaged)
    print(f" R@100:   {normalized['R@100']:.2f}") # Recall@100
    print("-"*60)

    all_results[model_name] = normalized

    model_metric_string = model_name.split()

    experiment_three_table.add_row([
      model_metric_string[0],
      model_metric_string[1],
      model_metric_string[2],
      normalized['MAP'], 
      normalized['MRR'], 
      normalized['nDCG@10'], 
      normalized['MacroPR_AUC'], # normalized['AUC'],
      normalized['R@100']
    ])

    macro_avg_PR_AUC_data[model_name] = {
      "recall": R_grid.tolist(),
      "precision": P_macro.tolist()
    }

output_file = '../data/oov_entity_mentions_multi_relevant_targets_weight_w035_50_queries.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"All results saved to {output_file}")

output_macro_pr_auc_file = '../data/oov_entity_mentions_multi_target_WEIGHTED_w035_50_q__PR_AUC_POINTS_4_PLOT.json'
with open(output_macro_pr_auc_file, 'w') as f:
    json.dump(macro_avg_PR_AUC_data, f, indent=2)

print(f"Macro PR AUC plot data dumped to {output_macro_pr_auc_file}")

print(f"Printing table: \n\n")

print(experiment_three_table.draw())

print("\n\n Printing LaTeX: \n\n")

print(latextable.draw_latex(
    table=experiment_three_table, 
    caption="Performance of fetching multiple relevant entities using OOV mentions with lambda=0.35 (50 Queries)",
    use_booktabs=True, position="H", caption_above=True, caption_short="Multi target performance of OOV mentions, lambda=0.35",
    label="tab:multi-target-oov-weighted"
  )
)