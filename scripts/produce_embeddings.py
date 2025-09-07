from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import numpy as np

from thesis.utils.data_utils import load_json, save_json
from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer

print("Preparing data for indexing/encoding...")

data_dir = "./data"
embeddings_dir = "./embeddings"

if not (Path(data_dir).exists()):
  print("[WARNING] No data directory exists. The notebook will fail. Review the README.md, or the docs dir.")

# if an embeddings dir has not yet been created, create one    
Path(embeddings_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

# generated during SNOMED CT processing
entity_lexicon_fp = Path(f"{data_dir}/preprocessed_entity_lexicon.json")

# list of the verbalisations (label text, or deeponto verbs)
verbalisation_list_fp = Path(f"{embeddings_dir}/verbalisations.json")
# each index of the entity_map points to a tuple: (index, label, verbalisation, iri)
entity_map_fp = Path(f"{embeddings_dir}/entity_map.json")
# compiles a list of the above mappings (handy when it comes to argsort)
entity_mappings_list_fp = Path(f"{embeddings_dir}/entity_mappings.json")

entity_lexicon = load_json(entity_lexicon_fp)
iris = entity_lexicon.keys()

entity_map = {}
entity_verbalisation_list = []
list_of_entity_mappings = []

for entity_idx, entity_iri in enumerate(tqdm(iris)):
    entity_map[str(entity_idx)] = {
        "mapping_id": str(entity_idx),
        "label": entity_lexicon[entity_iri].get('name'), # type: ignore
        "verbalisation": strip_parens(str(entity_lexicon[entity_iri].get('name'))).lower(), # type: ignore
        "iri": entity_iri
    }
    entity_verbalisation_list.append(strip_parens(str(entity_lexicon[entity_iri].get('name'))).lower()) # type: ignore
    list_of_entity_mappings.append(entity_map[str(entity_idx)])

save_json(verbalisation_list_fp, entity_verbalisation_list)
save_json(entity_map_fp, entity_map)
save_json(entity_mappings_list_fp, list_of_entity_mappings)

print("Completed verbalisation extraction!")

# SBERT

sbert_plm_hf_string = "all-MiniLM-L12-v2"
sbert_plm_encoder = SentenceTransformer.load(sbert_plm_hf_string)
sbert_plm_embeddings = sbert_plm_encoder.encode(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")
np.save(f"{embeddings_dir}/sbert-plm-embeddings.npy", sbert_plm_embeddings)

# HiT SNOMED 25 (Full)

hit_snomed_25_model_fp = '../models/snomed_models/HiT-mixed-SNOMED-25/final'
hit_snomed_25_encoder = HierarchyTransformer.from_pretrained(hit_snomed_25_model_fp)
hit_snomed_25_embeddings = hit_snomed_25_encoder.encode(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/hit-snomed-25-embeddings.npy", hit_snomed_25_embeddings)

# OnT GALEN

ont_galen_23_pred_model_fp = "../models/models/prediction/OnTr-all-MiniLM-L12-v2-GALEN"
ont_galen_23_pred_encoder = OntologyTransformer.load(ont_galen_23_pred_model_fp)
ont_galen_23_pred_embeddings = ont_galen_23_pred_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-galen-23-pred-embeddings.npy", ont_galen_23_pred_embeddings)

# ANATOMY

ont_anatomy_23_pred_model_fp = "../models/models/prediction/OnTr-all-MiniLM-L12-v2-ANATOMY"
ont_anatomy_23_pred_encoder = OntologyTransformer.load(ont_anatomy_23_pred_model_fp)
ont_anatomy_23_pred_embeddings = ont_anatomy_23_pred_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-anatomy-23-pred-embeddings.npy", ont_anatomy_23_pred_embeddings)

# Gene Ontology (GO)

ont_gene_ontology_23_pred_model_fp = "../models/models/prediction/OnTr-all-MiniLM-L12-v2-GO"
ont_gene_ontology_23_pred_encoder = OntologyTransformer.load(ont_gene_ontology_23_pred_model_fp)
ont_gene_ontology_23_pred_embeddings = ont_gene_ontology_23_pred_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-go-23-pred-embeddings.npy", ont_gene_ontology_23_pred_embeddings)

# SNOMED CT 2025 (Full)

ontr_snomed_25_uni_model_fp = '../models/snomed_models/OnTr-snomed25-uni'
ontr_snomed_25_uni_encoder = OntologyTransformer.load(ontr_snomed_25_uni_model_fp)
ont_snomed_25_embeddings = ontr_snomed_25_uni_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-snomed-25-embeddings.npy", ont_snomed_25_embeddings)

# SNOMED CT 2025 (M-64)

ontr_snomed_minified_model_fp = '../models/snomed_models/OnTr-minified-64'
ontr_snomed_encoder = OntologyTransformer.load(ontr_snomed_minified_model_fp)
ont_snomed_minified_embeddings = ontr_snomed_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-snomed-minified-embeddings.npy", ont_snomed_minified_embeddings)

# SNOMED CT 2025 (M-128)

ontr_snomed_minified_128_model_fp = '../models/snomed_models/OnTr-m-128'
ontr_snomed_m_128_encoder = OntologyTransformer.load(ontr_snomed_minified_128_model_fp)
ont_snomed_minified_128_embeddings = ontr_snomed_m_128_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy", ont_snomed_minified_128_embeddings)

# SNOMED CT 2025 (M-32)

ontr_snomed_minified_32_model_fp = '../models/snomed_models/OnTr-m-32'
ontr_snomed_m_32_encoder = OntologyTransformer.load(ontr_snomed_minified_32_model_fp)
ont_snomed_minified_32_embeddings = ontr_snomed_m_32_encoder.encode_concept(
    entity_verbalisation_list,
    batch_size=128,
    show_progress_bar=True
).astype("float32")
np.save(f"{embeddings_dir}/ont-snomed-minified-32-embeddings.npy", ont_snomed_minified_32_embeddings)

print("Embeddings saved!!")
