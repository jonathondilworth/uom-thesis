#!/usr/bin/env python3
"""
train_only_ont.py
-----------------
Train the OntologyTransformer **only** and write everything the evaluation
phase needs to disk.

Run:
    python train_only_ont.py --config_file ./config.yaml
"""
from __future__ import annotations
import json, logging, os, shutil, sys
from datetime import datetime

import click
import torch
from deeponto.utils import create_path, load_file, set_seed
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from yacs.config import CfgNode

from src.hierarchy_transformers.datasets import load_local_dataset
from src.hierarchy_transformers.evaluation import OnTEvaluator
from src.hierarchy_transformers.losses import HierarchyTransformerLoss, LogicalConstraintLoss
from src.hierarchy_transformers.models.hierarchy_transformer import (
    OntologyTransformer,
    HierarchyTransformer,
    HierarchyTransformerTrainer,
)

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True), required=True)
def main(config_file: str):
    # 0. housekeeping
    set_seed(8888)
    cfg: CfgNode = CfgNode(load_file(config_file))
    model_name = cfg.model_path.split(os.path.sep)[-1]
    exp_dir = f"experiments/OnTr-{model_name}-{cfg.dataset_name}"
    create_path(exp_dir)
    shutil.copy2(config_file, os.path.join(exp_dir, "config.yaml"))

    # 1. data + model
    dataset = load_local_dataset(cfg.dataset_path)
    base_model = HierarchyTransformer.from_pretrained(cfg.model_path)
    model = OntologyTransformer(base_model, cfg.role_emd_mode, cfg.role_model_mode)

    # 2. losses
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=cfg.hit_loss.clustering_loss_weight,
        clustering_loss_margin=cfg.hit_loss.clustering_loss_margin,
        centripetal_loss_weight=cfg.hit_loss.centripetal_loss_weight,
        centripetal_loss_margin=cfg.hit_loss.centripetal_loss_margin,
    )
    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=cfg.train_batch_size,
        data_exist=dataset["train_exist"],
        data_conj=dataset["train_conj"],
        conj_weight=cfg.logical_loss.conj_weight,
        exist_weight=cfg.logical_loss.exist_weight,
        existence_loss_kind=cfg.existence_loss_kind,
    )

    # 3. validation evaluator (used only to discover best centri_weight)
    val_evaluator = OnTEvaluator(
        ont_model=model,
        query_entities=dataset["val"]["query_sentences"],
        answer_ids=dataset["val"]["answer_ids"],
        all_entities=dataset["concept_names"]["name"],
        batch_size=cfg.eval_batch_size,
    )

    # 4. trainer
    training_args = SentenceTransformerTrainingArguments(
        output_dir=exp_dir,
        num_train_epochs=int(cfg.num_train_epochs),
        learning_rate=float(cfg.learning_rate),
        per_device_train_batch_size=int(cfg.train_batch_size),
        per_device_eval_batch_size=int(cfg.eval_batch_size),
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        metric_for_best_model="MRR",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    trainer = HierarchyTransformerTrainer(
        model=model.hit_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        loss=logical_loss,
        evaluator=val_evaluator,
    )
    trainer.train()

    # 5. collect info for the evaluation stage
    best_centri_weight = float(val_evaluator.results.iloc[-1]["centri_weight"])
    payload = {
        "best_centri_weight": best_centri_weight,
        "test_query_entities": dataset["test"]["query_sentences"],
        "test_answer_ids": dataset["test"]["answer_ids"],
        "all_concept_names": dataset["concept_names"]["name"],
        "eval_batch_size": cfg.eval_batch_size,
    }
    with open(os.path.join(exp_dir, "eval_payload.json"), "w") as f:
        json.dump(payload, f)
    logger.info("Saved eval_payload.json with best_centri_weight=%.4f", best_centri_weight)

    # 6. save final model
    final_dir = os.path.join(exp_dir, "final")
    model.save(final_dir)
    logger.info("Model saved to %s", final_dir)


if __name__ == "__main__":
    main()