from __future__ import annotations

from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import json
import re
import random
import numpy as np
import torch

# GLOBALS

_regex_parens = re.compile(r"\s*\([^)]*\)") # for parentheses removal (prevents leakage)

# FUNCS

def my_set_seed(seed_value: int):
  random.seed(seed_value) # python
  np.random.seed(seed_value) # numpy
  torch.manual_seed(seed_value) # pytorch
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value) # single GPU, fails silently
    torch.cuda.manual_seed_all(seed_value) # multi GPU, fails silently


def strip_parens(s: str) -> str:
    return _regex_parens.sub("", s)


def load_json(file_path: Path) -> dict[str, str]:
    with file_path.open('r', encoding='utf-8') as fp:
        return json.load(fp)


def save_json(file_path: Path, payload: dict | list, encoding: str = "utf-8", indentation: int = 4) -> None:
    with open(file_path, "w", encoding=encoding) as fp:
        json.dump(payload, fp, indent=indentation)


def load_concepts_to_list(concepts_file_path: Path) -> list[str]:
    return list(load_json(concepts_file_path).values())


def naive_tokenise(seq: str) -> list[str]:
    return seq.lower().split()


# ~800k concepts, assuming 16 cores, 32 threads; 
# 800k / 32 = 25k (note: subprocesses != threads)
def parallel_tokenise(seq_list: list[str], workers: int, chunksize: int = 25000) -> list[list[str]]:
    with Pool(workers) as pool:
        return list(pool.map(naive_tokenise, seq_list, chunksize=chunksize))


def get_dataset(dataset_key: str, benchmark_data: dict) -> str:
    return benchmark_data[dataset_key]


def get_question_obj(question_id: str, dataset: dict) -> dict:
    return dataset[question_id]


def get_dataset_question_mapping(dataset_key: str, benchmark_data: dict) -> dict[int, str]:
    data = get_dataset(dataset_key, benchmark_data)
    mapping = {}
    for question_index, question_id in enumerate(data):
        mapping[question_index] = question_id
    return mapping


def get_question_str(question_id: str, dataset: dict) -> str:
    return dataset[question_id]['question']


def get_question_opts(question_id: str, dataset: dict) -> dict:
    return dataset[question_id]['options']


def get_question_ans(question_id: str, dataset: dict) -> str:
    return dataset[question_id]['answer']


def get_dataset_names(benchmark_data: dict) -> list[str]:
    return list(benchmark_data.keys())


def get_question_count(dataset_name: str, benchmark_data: dict) -> int:
    return len(benchmark_data[dataset_name])


def get_random_question_sample(benchmark_data: dict, allowable_datasets: list[str] = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']) -> dict:
    random_dataset_name = allowable_datasets[random.randint(0, len(allowable_datasets) - 1)]
    dataset_question_mapping = get_dataset_question_mapping(random_dataset_name, benchmark_data)
    # ^ provides a map from custom indicies used to access questions specific to each dataset
    dataset_questons_xs = benchmark_data[random_dataset_name]
    random_question_index = dataset_question_mapping[random.randint(0, len(dataset_question_mapping) - 1)]
    return dataset_questons_xs[random_question_index]


def xs_of_all_questions(benchmark_data: dict, allowable_datasets: list[str] = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']) -> list[dict]:
    xs = []
    for dataset_name in allowable_datasets:
        dataset_question_mappings = get_dataset_question_mapping(dataset_name, benchmark_data)
        question_list = benchmark_data[dataset_name]
        for itr, mapping_idx in dataset_question_mappings.items():
            xs.append(question_list[mapping_idx])
    return xs


def get_question_entity_mentions(entity_mention_data: dict, dataset: str, question_id: str):
    for question in entity_mention_data:
        if question['source_dataset'] == dataset and question['question_id'] == question_id:
            return question['entities'] # warning: it is possible to be []


def merge_entity_mentions(benchmark_data: dict, biomed_entities: dict, head_entities: dict, allowable_datasets: list[str] = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']):
    xs = []
    for dataset_name in allowable_datasets:
        print(f"Processing {dataset_name} ... ")
        dataset_question_mappings = get_dataset_question_mapping(dataset_name, benchmark_data)
        question_list = benchmark_data[dataset_name]
        for itr, mapping_idx in tqdm(dataset_question_mappings.items()):
            question_entities_biomedical = get_question_entity_mentions(biomed_entities, dataset_name, mapping_idx)
            question_entities_head = get_question_entity_mentions(head_entities, dataset_name, mapping_idx)
            question_list[mapping_idx]['entities'] = []
            question_list[mapping_idx]['entities'].extend(question_entities_biomedical)
            question_list[mapping_idx]['entities'].extend(question_entities_head)
            xs.append(question_list[mapping_idx])
    return xs

