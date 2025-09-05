from __future__ import annotations

from pathlib import Path
from typing import Any, Union, NamedTuple
from functools import reduce
from pydantic import validate_call
from collections.abc import Mapping, Sequence

import json
import copy
import math


def make_signature(obj):
  if isinstance(obj, Mapping):
    return tuple((key, make_signature(obj[key])) for key in sorted(obj))
  elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
    return tuple(make_signature(item) for item in obj)
  # else:
  return obj


def unique_unhashable_object_list(obj_xs: list[dict]) -> list[dict]:
    object_signatures = set()
    unique_obj_list = []
    for obj in obj_xs:
      signature = make_signature(sorted(obj.items()))
      if signature not in object_signatures:
        object_signatures.add(signature)
        unique_obj_list.append(obj)
    return unique_obj_list


def obj_max_depth(x: int, obj: Any, key: str = "depth") -> int:
  return x if x > obj[key] else obj[key]


def dcg_exp_relevancy_at_pos(relevancy: int, rank_position: int) -> float:
  if relevancy <= 0:
    return float(0.0)
  numerator = (2**relevancy) - 1
  denominator = math.log2(rank_position + 1)
  return float(numerator / denominator)


def dcg_linear_relevancy_at_pos(relevancy: int, rank_position: int) -> float:
  if relevancy <= 0:
    return float(0.0)
  numerator = relevancy
  denominator = math.log2(rank_position + 1)
  return float(numerator / denominator)


def accumulate(a, b, key='dcg'):
  return a + b[key]


class Query:
  
  _query_obj_repr: dict
  _query_string: str
  _target: dict[str, Union[str, int]]
  _entity_mention: dict[str, Union[str, int]]

  def __init__(self, query_obj_repr: dict):
    self._query_obj_repr = query_obj_repr
    self._query_string = query_obj_repr['entity_mention']['entity_literal']
    self._target = query_obj_repr['target_entity']
    self._entity_mention = query_obj_repr['entity_mention']

  def get_query_string(self) -> str:
    return self._query_string
  
  def get_target_iri(self) -> str:
    return str(self._target['iri'])
  
  def get_target_label(self) -> str:
    return str(self._target['rdfs:label'])
  
  def get_target_depth(self) -> int:
    return int(self._target['depth'])
  
  def get_target(self) -> dict:
    return self._target
  

class EquivQuery(Query):
  
  _equiv_class_expression: Union[str, None]

  def __init__(self, query_obj_repr: dict):
    super().__init__(query_obj_repr)


class SubsumptionQuery(Query):
  
  _parents: list
  _ancestors: list

  def __init__(self, query_obj_repr: dict):
    super().__init__(query_obj_repr)
    self._set_parents()
    self._set_ancestors()

  def _set_parents(self):
    if self._query_obj_repr['parent_entities'] and len(self._query_obj_repr['parent_entities']) > 0:
      self._parents = self._query_obj_repr['parent_entities']
    else:
      self._parents = []

  def _set_ancestors(self):
    if self._query_obj_repr['ancestors'] and len(self._query_obj_repr['ancestors']) > 0:
      self._ancestors = self._query_obj_repr['ancestors']
    else:
      self._ancestors = []

  def get_parents(self) -> list:
    return self._parents
  
  def get_ancestors(self) -> list:
    return self._ancestors
  
  def get_all_subsumptive_targets(self) -> list:
    return [self._target, *self._parents, *self._ancestors]
  
  def get_unique_subsumptive_targets(self) -> list:
    return unique_unhashable_object_list(
      self.get_all_subsumptive_targets()
    )

  def get_sorted_subsumptive_targets(self, key="depth", reverse=False, depth_cutoff=3) -> list:
    xs = self.get_all_subsumptive_targets()
    xs.sort(key=lambda x: x[key], reverse=reverse)
    return xs[:depth_cutoff]
  
  def get_unique_sorted_subsumptive_targets(self, key="depth", reverse=False, depth_cutoff=3) -> list:
    return unique_unhashable_object_list(
      self.get_sorted_subsumptive_targets(key=key, reverse=reverse, depth_cutoff=depth_cutoff)
    )
  
  def get_targets_with_dcg(self, type="exp", depth_cutoff=3, **kwargs) -> tuple[float, list[dict]]:
    # get targets (target, parents, ancestors) ordered in ascending via depth
    targets_asc_depth = self.get_unique_sorted_subsumptive_targets(key="depth", depth_cutoff=depth_cutoff)
    # increase depth by 1 (offsetting zero-based index)
    targets_w_offset = [
      {**x, "depth": x["depth"] + 1}
      for x in targets_asc_depth
    ]
    # find the max depth (to calculate relevancy): (max_depth - depth_at_pos_k) + zero_based_offset
    # which we refer to as: relevancy := ascent height + zero_based_offset
    max_target_depth = reduce(
      obj_max_depth, 
      targets_w_offset, 
      0
    )
    # calculate relevance for each target (node/parent/ancestor) 
    targets_with_rel = [
      {**x, "relevance": (max_target_depth - x["depth"]) + 1}
      for x in targets_w_offset
    ]
    # ensure targets are sorted by relevance
    targets_with_rel.sort(key=lambda x: x['relevance'], reverse=True)
    # calculate dcg:
    if type == "linear":
      targets_with_dcg = [
        {**x, "dcg": dcg_linear_relevancy_at_pos(x['relevance'], rank)}
        for rank, x in enumerate(targets_with_rel, start=1)
      ]
    else:
      targets_with_dcg = [
        {**x, "dcg": dcg_exp_relevancy_at_pos(x['relevance'], rank)}
        for rank, x in enumerate(targets_with_rel, start=1)
      ]
    iDCG = reduce(accumulate, targets_with_dcg, 0)
    self._idcg = iDCG
    self._targets_with_dcg = targets_with_dcg
    return iDCG, targets_with_dcg

  def get_ideal_dcg(self, type="exp"):
    if self._idcg:
      return self._idcg
    iDCG, targets = self.get_targets_with_dcg()
    return iDCG
  

class QueryObjectMapping:

  _loaded: bool
  _data_file_path: Path
  _data: list
  _equiv_queries: list
  _subsumpt_queries: list

  @validate_call
  def __init__(self, json_data_fp: Path):
    self._loaded = False
    self._data_file_path = json_data_fp
    self._load()
    self._map()

  def _load(self) -> None:
    with self._data_file_path.open('r', encoding='utf-8') as fp:
      self._data = json.load(fp)
    self._loaded = True

  @validate_call
  def load_from_path(self, json_data_fp: Path) -> None:
    # overwrite an existing file path
    self._data_file_path = json_data_fp
    self._load()

  # equivalence_retrieval: bool = True, subsumption_retrieval: bool = False
  def _map(self) -> None:
    equiv_queries = []
    subsumpt_queries = []
    for query_obj_repr in self._data:
      # if the query obj within the data contains an equiv class
      if len(query_obj_repr['equivalent_classes']) > 0:
        # we're dealing with an equiv query
        equiv_queries.append(EquivQuery(query_obj_repr))
      else:
        subsumpt_queries.append(SubsumptionQuery(query_obj_repr))
    self._equiv_queries = equiv_queries
    self._subsumpt_queries = subsumpt_queries

  def get_queries(self) -> tuple[list, list]:
    return (self._equiv_queries, self._subsumpt_queries)
  
  def get_subsumpt_queries_with_no_transformations(self):
    tmp_subsumpt_queries = copy.deepcopy(self._subsumpt_queries)
    result_queries = []
    for query in tmp_subsumpt_queries:
      if query._entity_mention['transformed_entity_literal_for_type_alignment'] == "":
        result_queries.append(query)
    return result_queries
  
  def get_subsumpt_queries_with_transformations_only(self):
    tmp_subsumpt_queries = copy.deepcopy(self._subsumpt_queries)
    result_queries = []
    for query in tmp_subsumpt_queries:
      if query._entity_mention['transformed_entity_literal_for_type_alignment'] != "":
        result_queries.append(query)
    return result_queries


class QueryResult(NamedTuple):
    rank: int
    iri: str
    score: float
    verbalisation: str