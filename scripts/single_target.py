from pathlib import Path
import json
import copy
import latextable
from latextable import texttable
import statistics
from sklearn.metrics import auc as sk_auc
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
  concept_subsumption
)
from thesis.utils.data_utils import load_json
from thesis.utils.query_utils import QueryObjectMapping
from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer
from sentence_transformers import SentenceTransformer

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

models_dict_single_target = {
  # BASELINES
  "BoW TFIDF": tfidf_ret,
  "BoW BM25": bm25_ret,
  # BASELINE CONTEXTUAL EMBEDDINGS
  "SBERT cos-sim": sbert_ret_plm_w_cosine_sim,
  # HiT (Full)
  "HiT SNO-25(F)": hit_ret_snomed_25_w_hyp_dist,
  # OnT Transfer Models
  "OnTr GALEN(P)": ont_ret_galen_pred_w_hyp_dist,
  "OnTr ANATOMY(P)": ont_ret_anatomy_pred_w_hyp_dist,
  "OnTr GO(P)": ont_ret_go_pred_w_hyp_dist,
  # OnT SNOMED Models (Full, batch_size=64, Mini, batch_size=[32,64,128])
  "OnTr SNO-25(F-64)": ont_ret_snomed_25_updtd_w_hyp_dist,  
  "OnTr SNO-25(M-32)": ontr_ret_snomed_minified_32_w_hyp_dist,  
  "OnTr SNO-25(M-64)": ontr_ret_snomed_minified_w_hyp_dist,
  "OnTr SNO-25(M-128)": ontr_ret_snomed_minified_128_w_hyp_dist,
  # Mixed Model Space (Product Manifold):
  "Mixed D_m(F)": mixed_ret,      # HiT(F) x OnT(F) x SBERT
  "Mixed D_m(M)": mixed_ret_mini  # OnT(M-32) x OnT(M-128) x SBERT <- SBERT down-weighted (1.0, 1.0, 0.35)
} 

# OOV QUERIES (SINGLE TARGET) [50]

# PREP TABLE START #
experiment_one_table = texttable.Texttable()
experiment_one_table.set_deco(texttable.Texttable.HEADER)
experiment_one_table.set_precision(2)
experiment_one_table.set_cols_dtype(['t', 't', 'f', 'f', 'f', 'f', 'f', 'f', 'f'])
experiment_one_table.set_cols_align(["l", "l", "c", "c", "c", "c", "c", "c", "c"])
experiment_one_table.header(["Model", "Variant", "MRR", "H@1", "H@3", "H@5", "Med", "MR", "R@100"])
# END-PREP TABLE #

ks      = [1, 3, 5, 100, len(verbalisations)]
MAX_K   = max(ks)

all_results = {}

for model_name, model in models_dict_single_target.items():
    
    # init accumulators
    results = {
      "MRR": 0.0, # Mean Reciprical Rank
      **{f"H@{k}": 0.0 for k in ks}, # Hits@k
      **{f"P@{k}": 0.0 for k in ks}, # Precision@k
      **{f"R@{k}": 0.0 for k in ks}, # Recall@k
      **{f"F1@{k}": 0.0 for k in ks}, # F1@k
      "MR": 0.0 # Mean Rank
    }
    # PR-AUC, Median Rank & Coverage are calculated during the test procedure
    hit_count = 0 # for coverage
    all_ranks = [] # for median rank

    for q_idx, query in enumerate(oov_single_target_queries):
        
        qstr = query.get_query_string()
        gold_iri = query.get_target_iri()

        ranked_results = [] # empty lists (are unlikely to exist) but are treated as full misses
        
        # TODO: replace with match (?) - i.e. switch
        if isinstance(model, HiTRetriever):
          if model._score_fn == entity_subsumption:
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=True, model=model._model, weight=0.0)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, OnTRetriever):
          if model._score_fn == concept_subsumption:
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=True, model=model._model, weight=0.0)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, SBERTRetriever):
          if model._score_fn == batch_cosine_similarity:
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=True)
          else:
            ranked_results = model.retrieve(qstr, top_k=None, reverse_candidate_scores=False)
        #
        elif isinstance(model, BM25Retriever):
          ranked_results = model.retrieve(qstr, top_k=None)
        #
        elif isinstance(model, TFIDFRetriever):
          ranked_results = model.retrieve(qstr, top_k=None)
        #
        elif isinstance(model, MixedModelRetriever):
           ranked_results = model.retrieve(qstr, top_k=None)
        #
        elif isinstance(model, CustomMixedModelRetriever):
           ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        else:
           raise ValueError("No appropriate retriever has been set.")

        retrieved_iris = [iri for (_, iri, _, _) in ranked_results] # type: ignore

        # MRR & MeanRank
        rank_pos = None
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri == gold_iri:
                rank_pos = rank_idx
                results["MRR"] += 1.0 / rank_idx
                results["MR"] += rank_idx
                break
        
        # for calculating coverage
        if rank_pos is not None:
           hit_count += 1

        # include a penalty to appropriately offset the MR
        # rather than artifically inflating the performance
        # by simply dropping queries that do not contain 
        if rank_pos is None:
            results["MR"] += MAX_K + 1 # penalty: rank := MAX_K + 1

        for k in ks:
            hit = 1 if (rank_pos is not None and rank_pos <= k) else 0
            results[f"H@{k}"] += hit # Hits@K
            p_at_k = hit / k # Precision@K
            results[f"P@{k}"] += p_at_k
            r_at_k = 1 if (rank_pos is not None and rank_pos <= k) else 0
            results[f"R@{k}"] += r_at_k
            if (p_at_k + r_at_k) > 0:
               results[f"F1@{k}"] += 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)

        final_rank = rank_pos if rank_pos is not None else MAX_K + 1
        all_ranks.append(final_rank)

    # normalise over queries & compute coverage
    N = len(oov_single_target_queries)
    normalized = {metric: value / N for metric, value in results.items()}
    normalized['Cov'] = (hit_count / N) # calculate the coverage of this model
    normalized['Med'] = statistics.median(all_ranks) # median rank
    # area under precision-recall curve (trapezodial rule)
    recall_at_k_xs    = [normalized[f"R@{k}"] for k in ks]
    # check for monotonic recall
    if any(r2 < r1 for r1, r2 in zip(recall_at_k_xs, recall_at_k_xs[1:])):
        raise ValueError(f"Recall must be non-decreasing for PR-AUC")
    precision_at_k_xs = [normalized[f"P@{k}"] for k in ks]
    normalized["AUC"] = float(sk_auc(recall_at_k_xs, precision_at_k_xs))

    print(f"Model: {model_name}")
    print(f"  MRR:    {normalized['MRR']:.2f}")
    for k in [1, 3, 5]:
        print(f"  H@{k}:    {normalized[f'H@{k}']:.2f}")
    print(f"  Med:    {normalized['Med']:.1f}")
    print(f"  MR:     {normalized['MR']:.1f}")
    print(f"  R@100:  {normalized['R@100']}")
    print("-"*60)
    
    model_metric_string = model_name.split()
    experiment_one_table.add_row([model_metric_string[0], model_metric_string[1], 
                                  normalized['MRR'], 
                                  normalized['H@1'], normalized['H@3'], normalized['H@5'], 
                                  normalized['Med'], normalized['MR'], normalized['R@100']])

    all_results[model_name] = normalized

output_file = './data/oov_entity_mentions_single_target_ANN_50_queries.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"All results saved to {output_file}")

print(f"Printing table: \n\n")

print(experiment_one_table.draw())

print("\n\n Printing LaTeX: \n\n")

print(latextable.draw_latex(
    table=experiment_one_table, 
    caption="Single target retrieval performance of OOV entity mentions measured across multiple models (50 Queries)", 
    use_booktabs=True, position="H", caption_above=True, caption_short="Single target performance of OOV mentions", 
    label="tab:single-target-oov"
  )
)