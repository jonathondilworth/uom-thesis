# Processing SNOMED Data & Training Embedding Models: HiT & OnT

**!! NOTE: ALL OF THESE STEPS ARE NOW AUTOMATED UNDER `make` !!**

**Required Dependencies**

*(system level, maybe more to add)*

1. JDK: `sudo apt install openjdk-17-jdk`
2. SNOMED-OWL-TOOLKIT: `wget https://github.com/IHTSDO/snomed-owl-toolkit/releases/download/5.3.0/snomed-owl-toolkit-5.3.0-executable.jar`
3. HierarchyTransformer: `git clone git@github.com:KRR-Oxford/HierarchyTransformers.git`
4. OntologyTransformer: `git clone <GITHUB_URL>`

*then:*

5. `mkdir data`
6. `mv <SNOMED_FILE_PATH>.zip ./data/.`

**GENERATING SNOMED .OWL FILE**

1. Run:

*(replace the directory structure, IRI and VERSION to suit your specific requirements)*

```
java -Xms4g --add-opens java.base/java.lang=ALL-UNNAMED -jar snomed-owl-toolkit-5.3.0-executable.jar \
  -rf2-to-owl \
  -rf2-snapshot-archives ./data/SnomedCT_InternationalRF2_PRODUCTION_20250701T120000Z.zip \
  -iri http://snomed.info/sct/900000000000207008 \
  -version 20250701 
```

2. `mv ./ontology-*.owl ./data/snomedct-international.owl`

3. If your enviornment doesn't already support DeepOnto, install it:

```
pip install deeponto
```

`mkdir scripts`

`mv <ONTO_VERBALISATION_AND_PROCESSING_AND_TESTING_SCRIPTS> .`

5. Run the test script under `./load_taxonomy.py` *(script for obtaining the entity lexicon for use with axiom retrieval methods & local hit training later)*:

**TODO: modify `load_taxonomy.py` to accept flags**
**TODO: modify these instructions to:**
`load_taxonomy.py --entity-lexicon --lexicon-name="snomed_entity_lexicon" --data-dir="./data"`
*OR: for a build/deployment script (Dockerfile, .sh or ansible script) **replace arguments with config values*

**load_taxonomy.py does the following:**

  * Loads the **asserted** EL ontology into DeepOnto
  * *(optionally)* generates the training data for HiT
  * produces the entity lexicon for use with retrieval, training & QA

```sh
python ./load_taxonomy.py
```

  * run `python preprocess_entity_lexicon.py`

**Do this to change the shape and labelling of the lexicon to allow for the HiT Trainer to function correctly.**

```sh
python preprocess_entity_lexicon.py --input ./data/snomed_entity_lexicon.json --output ./data/clean_cased_snomed_entity_lexicon.json
```

## Train HiT

**!! NOTE: This is now accomplished by simply running `make hit-data && make hit-train`. !!**

This bit is a little bit involved... *(you may just want to pull down a pre-trained model from HF as they're provided through the HiT HF data card or via zenodo; though, if you want to train your own model, you'll need to make some modifications which are similar to using data downloaded from zenodo)*

1. You will have generated (if followed step 4.1) training data `multi` and `mixed` for HiT
    * Move these into a `data` dir
    * Move or copy the `entity-lexicon` into both `data/multi` and `data/mixed`
    * *(optionally)* perform additional preprocessing *(though, I assumed the tokenizer should handle this?)*, such as:
        * lower-case the training data
        * *You may want to modify owl:Thing entry in the lexicon*, etc
2. **Modifying the `scripts/training_sft.py` & `scripts/training_hit.py` and setting up config:**
    * Since we're not using HF to pull down an old dataset (we want to use the newest version of SNOMED), we need to swap out the `load_hf_dataset` function within the trainer/evaluater with: `load_zenodo_dataset` fn:

```python
# scripts/training_sft.py
# --> first some additional imports <--
from pathlib import Path
import json

# ... we also require our lexicon for this method:
# TODO: replace with proper pathing
json_path = Path('./data/hit_dataset/mixed/entity_lexicon.json')
with json_path.open('r', encoding='utf-8') as f:
    data_dict = json.load(f)
lexicon = data_dict

# ... now swap out the `pair_dataset` hf call:
pair_dataset = load_zenodo_dataset(
    "./data/hit_dataset/mixed", 
    lexicon, 
    negative_type="random", example_type="pair"
)
```

3. Modifying `scripts/training_hit.py`:

```python
# scripts/training_hit.py
# this step is essentially the same as the above, but...
# for the re-training step (approximate mapping to poincare ball/hyperbolic space)
# we require both the `pair_dataset` and `triplet_dataset`:

triplet_dataset = load_zenodo_dataset("./data/hit_dataset/mixed", lexicon, negative_type="random", example_type="triplet")
pair_dataset = load_zenodo_dataset("./data/hit_dataset/mixed", lexicon, negative_type="random", example_type="pair")
```

*(You may run into some minor type consistency issues, but from this point, everything is fix-able in a short time, couple of minutes)*

4. Update the configuration/config files in the respective `training_hit` and `training_sft` folders to utilise your preferred parameters **(but as importantly, your dataset!)**, you ought to consider checking the paper to see what the configurable parameters were set to for training.

  4.1. For standard fine-tuning *(this will take a while to run)*, the paper states that they: "largely adhered to the default settings of the huggingface trainer, but maintained the same training batch size as used in the hierarchy re-training".

  4.2. For hierarchy re-training:
    * hyperbolic clustering loss margin: 5.0
    * hyperbolic centripetal loss margin: 0.1
    * epochs: 20
    * batch-size: 256
    * 500 warm-up steps in validation
    * initial learning rate of $10^{-5}$
    * AdamW optimizer
    * Model selection after each epoch guided by performance on validation set.

*Changes:*

```python
# --> START CHANGE BETWEEN STEP 0. AND STEP 1. : LOAD LEXICON <--
# TODO: replace with proper abs path
json_path = Path('./data/hit_dataset/mixed/entity_lexicon.json')
with json_path.open('r', encoding='utf-8') as f:
    data_dict = json.load(f)
lexicon = data_dict
# --> END CHANGE <--

# 1. Load dataset and pre-trained model
# --> START CHANGE : SWITCH DATASET LOADER WITH YOUR PREFERRED SETTINGS FOR TYPES <--
# pair_dataset = load_hf_dataset(config.dataset_path, config.dataset_name + "-Pairs") 
pair_dataset = load_zenodo_dataset("./data/hit_dataset/mixed", lexicon, negative_type="random", example_type="pair")
# --> END CHANGE <--
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=config.model_path,
    num_labels=2
)
```

5. Now that the neccesary changes are made, you can train the base (s)BERT model first:

*(standard fine-tuning step)*

`python HierarchyTransformers/scripts/training/sft/training_sft.py --config_file ./HierarchyTransformers/scripts/training/sft/config_sft.yaml`

And then re-train using the specified hyperbolic clustering & centripetal loss (`training_hit.py`):

`python HierarchyTransformers/scripts/training/hit/training_hit.py --config_file ./HierarchyTransformers/scripts/training/hit/config_hit.yaml `

*(obtaining results from evaluating hit for best weight scoring in retrieval and QA later):*

`mkdir ./hit_eval_output`

`python HierarchyTransformers/scripts/evaluation/hit/eval_hit.py --config_file HierarchyTransformers/scripts/evaluation/hit/config_hit.yaml --output_path ./hit_eval_output/`

**RESULTS**

SFT training:

*After 50k steps, overfitting the training set? Best model @ 13k steps (I have two checkpoints, one at 50k steps and the other at 13k steps).*

```
{'eval_loss': 0.19461706280708313, 'eval_runtime': 1186.6744, 'eval_samples_per_second': 1879.449, 'eval_steps_per_second': 3.672, 'epoch': 2.99}                                                                                         
{'loss': 0.0108, 'grad_norm': 1.6663192510604858, 'learning_rate': 7.180469123982767e-09, 'epoch': 3.0}              
{'train_runtime': 137048.7539, 'train_samples_per_second': 93.649, 'train_steps_per_second': 0.366, 'train_loss': 0.016753383883762078, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████| 50136/50136 [38:04:08<00:00,  2.73s/it]
100%|████████████████████████████████████████████████████████████████████████████| 4357/4357 [19:53<00:00,  3.65it/s]
INFO:__main__:threshold                0.500000
precision                0.948391
recall                   0.628907
f1                       0.756293
accuracy                 0.963153
accuracy_on_negatives    0.996578
Name: testing, dtype: float64
```

SFT eval results on another training run (distributed across two GPUs):

```
	threshold	precision	recall	f1	accuracy	accuracy_on_negatives
testing	0.5	0.9165084362030029	0.6617592573165894	0.7685742378234863	0.9637704491615295	0.9939715266227722
```

HiT **(only, no prior SFT'ing)** - *early attempt*:

```tsv
centri_weight	threshold	precision	recall	f1	accuracy	accuracy_on_negatives
validation	1.3	-23.16	0.9239423274993896	0.8821823596954346	0.9025796055793762	0.9826874732971193	0.9927380084991456
testing	1.3	-23.16	0.9227519035339355	0.8820744156837463	0.9019548296928406	0.9825665354728699	0.9926156997680664
```

**HiT re-training a SFT'ed model (SFT @ 13k steps): *(still running)***

Eval on HiT @ 10 epochs:

```
INFO:hierarchy_transformers.evaluation.hit_eval:Evaluate with grid search on hyperparameters `best_centri_weight` (centripetal score weight) and `best_threshold` (overall threshold).████████████████████| 4357/4357 [03:34<00:00, 47.18it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4342/4342 [00:03<00:00, 1096.72it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4357/4357 [00:03<00:00, 1098.43it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4372/4372 [00:03<00:00, 1098.34it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4388/4388 [00:03<00:00, 1099.50it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4403/4403 [00:04<00:00, 1098.56it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4418/4418 [00:03<00:00, 1120.80it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4434/4434 [00:03<00:00, 1230.38it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4449/4449 [00:03<00:00, 1210.19it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4479/4479 [00:03<00:00, 1144.75it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4533/4533 [00:04<00:00, 1096.70it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4588/4588 [00:04<00:00, 1094.17it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4711/4711 [00:04<00:00, 1093.85it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 4988/4988 [00:04<00:00, 1171.08it/s]
Thresholding: 100%|██████████████████████████████████████████████████████████████| 5341/5341 [00:04<00:00, 1230.52it/s]
INFO:hierarchy_transformers.evaluation.hit_eval:Eval results: {'centri_weight': 1.2, 'threshold': -21.72, 'precision': 0.9382465481758118, 'recall': 0.9719857573509216, 'f1': 0.9548181891441345, 'accuracy': 0.9916374087333679, 'accuracy_on_negatives': 0.9936025738716125}
{'eval_loss': 0.1064, 'eval_centri_weight': 1.2, 'eval_threshold': -21.72, 'eval_precision': 0.9382465481758118, 'eval_recall': 0.9719857573509216, 'eval_f1': 0.9548181891441345, 'eval_accuracy': 0.9916374087333679, 'eval_accuracy_on_negatives': 0.9936025738716125, 'eval_runtime': 1439.7086, 'eval_samples_per_second': 1408.299, 'eval_steps_per_second': 1.376, 'eval_cluster_loss': 0.1054, 'eval_centri_loss': 0.001, 'epoch': 10.0}
```

Training @ 10 epochs:

```
{'loss': 0.0044, 'grad_norm': 0.5304040908813477, 'learning_rate': 5.01459323824617e-06, 'cluster_loss': 0.0027, 'centri_loss': 0.0017, 'epoch': 10.0}
{'loss': 0.004, 'grad_norm': 0.341686874628067, 'learning_rate': 5.007989963021659e-06, 'cluster_loss': 0.0024, 'centri_loss': 0.0016, 'epoch': 10.02}
{'loss': 0.0034, 'grad_norm': 0.9817683696746826, 'learning_rate': 5.0013866877971475e-06, 'cluster_loss': 0.0019, 'centri_loss': 0.0015, 'epoch': 10.03}
```

## ELNormalize:

**!! NOTE: This is now accomplished by simply running `make ont-data`. !!**

```sh
cd OnT
mkdir data
cp <PATH_TO_SNOMED_CT>.owl ./data/.
```

*You might want to increase the amount of available memory for the JVM right about now..*

```sh
# 32gb
export JAVA_TOOL_OPTIONS="-Xmx32g"
```

**Run:**

```sh
python ./normalize/ELNormalizedData.py --input ./data/snomedct-international.owl --output ./data/snomedct-el-normalised-compendium-dataset
```

Now Run `tree ./data/snomedct-el-normalised-compendium-dataset` to ensure you have the following directory structure:

```sh
.
├── concept_names.json
├── OnT
│   ├── concept_names.json
│   ├── role_names.json
│   ├── test.json
│   ├── train_conj.jsonl
│   ├── train_exist.jsonl
│   ├── train.jsonl
│   ├── train_nf1.jsonl
│   ├── train_nf2.jsonl
│   ├── train_nf3.jsonl
│   ├── train_nf4.jsonl
│   └── val.json
├── prediction
│   ├── classes.json
│   ├── relations.json
│   ├── test
│   │   ├── nf1.npy
│   │   ├── nf2.npy
│   │   ├── nf3.npy
│   │   └── nf4.npy
│   ├── train
│   │   ├── class_ids.npy
│   │   ├── disjoint.npy
│   │   ├── nf1.npy
│   │   ├── nf2.npy
│   │   ├── nf3.npy
│   │   ├── nf4.npy
│   │   ├── role_chain.npy
│   │   ├── role_inclusion.npy
│   │   └── top.npy
│   └── val
│       ├── nf1.npy
│       ├── nf2.npy
│       ├── nf3.npy
│       └── nf4.npy
└── role_names.json
```

**Pre-processing:** If you would like to pre-process the data to:

* Remove branch (parenthesised high level concepts) names from concepts
    * e.g. `example (branch name)` -> `example `
    * flag: `--strip-parentheses`, `default=True`
* Convert all concepts to lower case, flag: `--to-lower-case`, `default=False`
* Remove duplicate white-space, flag: `--collapse-whitespace`, `default=False`

Run: `python utils/ont_preprocess_el_dir.py --base-dir ./data/snomedct-el-normalised-compendium-with-preprocessing --strip-parentheses --to-lower-case --collapse-whitespace`

9. *(optionally)* Train OnT:

**!! NOTE: This is now accomplished by simply running `make ont-train`. !!**

*config.yaml:*

```yaml
train_batch_size: 32 # was 64
eval_batch_size: 16 # was 32
```

*hit_trainer.py, for distributed GPU workloads:*

```python
class HierarchyTransformerTrainer(SentenceTransformerTrainer):
    # ...
    def compute_loss(
            # ... params
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
            loss_dict = super().compute_loss(
                model=model, inputs=inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

            # <-- START: CHANGES
            if (isinstance(loss_dict, torch.Tensor)):
                return loss_dict
            # <-- END: CHANGES
            
            outputs = None
            
            if return_outputs:
                loss_dict, outputs = loss_dict
            
            if "conj_loss" in loss_dict:
                self.log(
                # ... *(rest of file)*
```

Add the guard:

```python
if (isinstance(loss_dict, torch.Tensor)):
    return loss_dict
```

The training loop will now run sucessfully (and looking at wandb, the loss curves look fine across all loss functions).

Running `train_ont.py`:

```sh
# configure options for distributed GPUs
config accelerate
# the output directory is determined within the training script
python train_ont.py --config_file ./config.yaml 
# or:
accelerate launch train_ont.py --config_file ./config.yaml 
# you might run into problems \w distributing the load again!
```

**config.yaml:**

```yaml
# dataset from local path
dataset_path: "./data/ont_dataset" 
dataset_name: "" # TODO <---

# pre-trained model from Hugging Face
# model_path: "sentence-transformers/all-MiniLM-L6-v2"
model_path: "sentence-transformers/all-MiniLM-L12-v2"
# model_path: "sentence-transformers/all-mpnet-base-v2"

# training config
num_train_epochs: 1
train_batch_size: 32 # UPDATE: had to change from 64 -> 32 due to VRAM issues
eval_batch_size: 16 # scaled with the batch size
learning_rate: 1e-5
role_emd_mode: "sentenceEmbedding" 
role_model_mode: "rotation" 
existence_loss_kind: "hit" 
hit_loss: 
  clustering_loss_weight: 1.0
  clustering_loss_margin: 3.0
  centripetal_loss_weight: 1.0
  centripetal_loss_margin: 0.5
logical_loss:
  conj_weight: 1.0  
  exist_weight: 1.0
```

**Issues with Evaluation on 48GB of VRAM:**

*Note: due to VRAM constraints, the batch size was reduced from 64 to 32. The evaluation batch size was scaled in alignment with the training batch size (from 32 to 16).*

*Since the training+evaluation script is fairly compute hungry, training and evaluation were conducted seperately. It's important to note that the evaluator appeared to utilise more data (predictably) when it executed immediately after training; though it failed to load the model due to an exception referencing remote code saftey (see below) and then ran out of memory.*

Error: `ERROR:sentence_transformers.trainer:Could not load the best model from experiments/OnTr-all-MiniLM-L12-v2-OnT-preprocessed-REMOTE/checkpoint-39504. Error: HierarchyTransformer.__init__() got an unexpected keyword argument 'trust_remote_code'`

```sh
RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript (most recent call last):
  File "/root/miniforge3/envs/msc-env/lib/python3.12/site-packages/geoopt/manifolds/stereographic/math.py", line 869, in fallback_function
):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
        ~~~~~~~~~~~ <--- HERE
    )
  File "/root/miniforge3/envs/msc-env/lib/python3.12/site-packages/geoopt/manifolds/stereographic/math.py", line 514, in _mobius_add
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
                                           ~~~~~~~~~~~~~~~ <--- HERE
    denom = 1 - 2 * k * xy + k**2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
RuntimeError: CUDA out of memory. Tried to allocate 34.64 GiB. GPU 0 has a total capacity of 79.18 GiB of which 7.38 GiB is free. Including non-PyTorch memory, this process has 71.79 GiB memory in use. Of the allocated memory 36.47 GiB is allocated by PyTorch, and 34.59 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

*Evaluation was migrated to its own script, the results reported below may have some discrepencies compared to results obtained during prior training.*

**Results, as reported by the evaluator:**

*(from running `python train_ont.py --config ./config.yaml` where the model being tuned is MiniLM-all-L12-v2, without any prior sft'ing)*

```
INFO:__main__:Test‑set metrics with centri_weight=0.9
  axiom_kind: combined
centri_weight: 0.9
         H@1: 0.18408116671326064
        H@10: 0.4705409561084708
       H@100: 0.699538719597428
         MRR: 0.275200949495547
          MR: 5779.212880129221
      median: 13.0
         AUC: 0.9923656030206802
```

Assuming H@k and MRR are percentages (as is stated in the paper), the results are reformated below:

```
axiom_kind: combined
centri_weight: 0.9
         H@1: 18.4%
        H@10: 47.1%
       H@100: 70.0%
         MRR: 27.5%
          MR: 5779
```

```
          H@K     MRR     MR
OnTr   18/47/70    28    5779
```

**Takeaway:** SNOMED~CT is actually quite large; and the model struggles to learn the appropriate mappings for NF1 with the current learning rate, batch size and training strategy. Given the limited time and compute availability (80gb of VRAM); we can chop down the size of SNOMED~CT to produce 'miniature' models and see whether that improves performance. Though, it may be worth accessing H200 architecture (141GB of VRAM per card) to try training the full model in the future.