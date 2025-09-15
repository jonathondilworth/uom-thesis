# Hierarchical Knowledge Retrieval using Transformer-based Ontology Embeddings (HiT, OnT) in SNOMED CT

## Usage

Host system requirements for running containerised experiments:

* Docker Engine
* Installation of NVIDIA GPU drivers (CUDA >= 12.1)
* NVIDIA Container Toolkit

Requirements for deploying to bare metal:

* Ubuntu >= 22.04 (tested on 22.04 and 24.04 LTS)

#### Run via Docker

End-to-End Experimental Run:

```
make docker-build
make docker-all
```

*For specific commands and script call sequence, see [Makefile](./Makefile) and [scripts](./scripts).*

#### Run via Bare Metal (with Makefile)

Possibly the simplest means of running end-to-end experiments; first spin up a fresh cloud instance, clone the repo, and run:

```
make
```

**See [Makefile](./Makefile) for a breakdown of bare metal make commands.**

### Manual Installation

**Requirements:**

* JDK: `sudo apt install openjdk-17-jdk`
* Maven: `sudo apt install maven`
* ROBOT: `git clone git@github.com:ontodev/robot.git` *see: (https://robot.obolibrary.org/)[https://robot.obolibrary.org/]*
* snomed-owl-toolkit: `git clone git@github.com:IHTSDO/snomed-owl-toolkit.git` and:
    * `cd snomed-owl-toolkit`
    * `wget https://github.com/IHTSDO/snomed-owl-toolkit/releases/download/5.3.0/snomed-owl-toolkit-5.3.0-executable.jar`

**For manual build processes, see [docs](./docs/).**

#### Tests

Simply run:

```
pytest
```



## Models (Encoders)

Models can be conviently downloaded using `gdown` or via the `make models` command. Encoders leveraged by the experiments include: HiT SNOMED Full, OnT SNOMED Full, OnT ANATOMY (prediction), OnT GALEN (prediction), OnT GO (prediction), OnT Miniature SNOMED (M-32, M-64, M-128).

### SNOMED-CT Tuned Models

Manual download link: [https://drive.google.com/file/d/1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT/view](https://drive.google.com/file/d/1cQOqFVOHqBKkSirepzF7ga6mRYPP-LnT/view)

### Pre-trained OnT Encoders

Manual download link: [https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view](https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view)

See [the original OnT paper](https://www.arxiv.org/abs/2507.14334) and the associated [GitHub](https://github.com/HuiYang1997/OnT).

### Training Models

To re-train the models used within the thesis (first ensure the requisite training data has been generated with `make hit-data` and/or `make ont-data`, then) modify the config provided under `./lib/hierarchy_transformers/scripts/config.yaml` and `./lib/OnT/config.yaml` (changing the batch size to `32`, `64`, or `128`), then run: `make hit-train` or `make ont-train`. Training for additional models is only supported via bare metal deployments.



## Experiments

The use of `Docker` is reccomended for running experiments locally. We strongly discourage using the bare metal deployment scripts on a local machine with pre-existing environments (as these scripts will automatically update your package manager, download external dependencies and could possibly break existing miniconda configurations).

To run the experiments with docker, ensure docker is installed, along with CUDA >= 12.1 drivers and nvidia container toolkit and run:

```
make docker-build
make docker-all
```

Should you wish to deploy to a fresh GPU cloud instance, simply cloning the repo and running `make` will configure the environment, download all neccesary dependencies, rebuild datasets, prepare the neccesary data for model re-training, obtain pre-trained encoder models, automatically produce embeddings for evaluation and will re-run all experiments (dumping experimental results under `./data`). Warning: This can take some time, but is the reccomended approach when deploying to bare metal.

To obtain precise experimental results, fully aligned with those reported in the paper, you will need to include your NHS TRUD API key within `.env`. In this instance you should run the following commands:

```
make init
echo "NHS_API_KEY=${MY_NHS_API_KEY_GOES_HERE}" >> .env
make docker-eval # for docker
make eval # for bare metal deployments
```

Failing to include the `NHS_API_KEY` in `.env` will cause the scripts to fallback to use a publicly available version of SNOMED~CT, provided under [Zenodo](https://zenodo.org/records/14036213). In such cases, you will obtain similar results to those reported in the paper, with some small deviations.

### Single & Multi Target Knowledge Retrieval Experiments

1. First build the docker container: `make docker-build`
2. Initialise the neccesary environment variables: `make docker-init`
    * Note: should you wish to use the latest SNOMED CT release, include your NHS TRUD API key under `NHS_API_KEY=<API_KEY_HERE>` within `.env`.
3. Download, then process SNOMED: `make docker-snomed`
4. Download, then process MIRAGE: `make docker-mirage`
5. Deterministically sample and rebuild the evaluation dataset: `make docker-sample && make docker-eval`
6. Obtain the neccesary models: `make docker-models`
7. Produce the neccesary embeddings for use in the experiments: `make docker-embeddings`
8. Run single target experiments: `make docker-single-target`
9. Run multiple target experiments: `make docker-multi-target`

See Usage. For additional experimental intution, see the included [notebook](./notebooks/retrieval-notebook.ipynb).

### RAG Experiments

The retrieval augmented generation experiments make use of the `transformers` library from HuggingFace. To run these experiments, you must first accept the conditions to use the models (BioMistral and Mistral7B) at the following links:

* [https://huggingface.co/BioMistral/BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B), and
* [https://huggingface.co/mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

Additionally, you must authorise via an account-based token using `hf auth login` (the `make env` will have already installed the HF cli):

```
hf auth login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `hf auth whoami` to get more information or `hf auth logout` if you want to log out.
    Setting a new token will erase the existing one.
    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): <ENTER_YOUR_ACCOUNT_TOKEN_HERE>
```

Prior to running RAG experiments, axiom verbalisations are required, to generate these, run the following comand:

`make axiom-verb` (bare metal) or `make docker-axioms` (docker)

Now you can simply run any of the following commands:

* `make no-rag` or `make docker-no-rag`: runs the No-RAG baseline experiments.
* `make sbert-rag` or `make docker-sbert-rag`: runs the SBERT RAG baseline experiments.
* `make hit-rag` or `make docker-hit-rag`: runs the HiT RAG experiments.
* `make ont-rag` or `make docker-ont-rag`: runs the OnT RAG experiments.



## Web Interface

To interface with the embedding store in an intuitive fashion, we provide a simple web interface. The web application leverages a react-based front-end and a FastAPI python wrapper around the project utilities found under `./src/thesis/utils`, specifically the `llm_utils.py` and `retrievers.py`. The purpose of the web interface is to demonstrate the application of transformer-based (language model-based) ontology embeddings to web-search and for retrieval augmented generation.

#### Screenshots

![img][./docs/imgs/search.png]

![img][./docs/imgs/rag.png]

For additional screenshots, see [the included images](./docs/imgs).

#### Usage

To launch the web interface UI, you'll need to have `npm` installed (see [Downloading and installing node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)).

```
cd src/frontend_web_ui
npm install
npm run dev
```

Additionally, to launch the backend FastAPI service, you must have already aquired the models (`make models`, or `make docker-models`), have FastAPI within your local python environment (`pip install fastapi`) and have produced the embeddings (`make embeddings or `/`make docker-embeddings`); then, run the following command from the project root:

```
conda activate knowledge-retrieval-env
uvicorn thesis.app.serve_single_host:app --host 0.0.0.0 --port 8000
```



## License

